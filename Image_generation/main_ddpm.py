import copy
import json
import os
import optuna
import pickle

import torch
import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10, CelebA
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import tqdm, trange
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from diffusion import extract
from model import UNet
from score.both import get_inception_and_fid_score
from score.statistic import get_mu_sigma
from dataclasses import dataclass
from torch import optim
from ACProp.optimizers import ACProp
from Adan import Adan
from lion_pytorch import Lion
from mechanic_pytorch import mechanize
from DDIM import generator



@dataclass
class TrainingConfig:
    train = True # train from scratch
    eval = False # load ckpt.pt and evaluate FID and IS
    device = torch.device('cuda:0') 

    # UNET
    ch = 128 # base channel of UNet
    ch_mult = [1, 2, 2, 2] # channel multiplier
    attn = [1] # add attention to these levels
    num_res_blocks = 2 # resblock in each level
    dropout = 0.1 # dropout rate of resblock
    
    # Gaussian Diffusion
    beta_1 = 1e-4 # start beta value
    beta_T = 0.02 # end beta value
    T = 1000 # total diffusion steps
    mean_type = 'epsilon' # predict variable
    var_type = 'fixedlarge' # variance type
    
    # Training
    lr = 2e-4 # target learning rate
    batch_size = 128 # batch size
    optimizer = Adan # optimizer
    noise_schedule = 'linear'
    grad_clip = 1. # gradient norm clipping
    total_epochs = 2000 # total training epochs, when load the pre-trained checkpoints to be changed
    img_size = 32 # image size
    num_subset = 5000 # the number of images in the created subset
    warmup = 200 # learning rate warmup
    num_workers = 4 # workers of Dataloader
    ema_decay = 0.9999 # ema decay rate
    parallel = False # multi gpu training
    seed = torch.manual_seed(42)

    # Logging & Sampling
    logdir = '/home/jhxiu/ddpm_torch/logs/DDPM_CIFAR10_EPS' # log directory
    sample_size = 64 # sampling size of images
    sample_epoch = 200 # frequency of sampling

    # Saving images with ddim 
    ddim_sample = True # use ddim to accelerate sample
    eta = 0.0
    method = "linear"
    result_only = True
    nrow = 4
    steps = 100
    show = False
    image_save_path = '/home/jhxiu/ddpm_torch/logs/DDPM_CIFAR10_EPS/sample'
    to_grayscale = False
    interval = 50

    # Evaluation
    save_epoch = 50 # frequency of saving checkpoints, 0 to disable during training
    eval_epoch = 400 # frequency of evaluating model, 0 to disable during training
    num_images = 50000 # the number of generated images for evaluation
    fid_use_torch = False # calculate IS and FID on gpu
    fid_cache = 'stats/cifar10.train.npz' # FID cache


config = TrainingConfig()
device = config.device
print(config.total_epochs,config.save_epoch,config.eval_epoch,config.num_images,config.batch_size)

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(epoch):
    return min(epoch, config.warmup) / config.warmup


def evaluate(sampler, model, epoch):
    model.eval()
    with torch.no_grad():
        if config.ddim_sample:
            config.epoch = epoch
            config.model = UNet(T=config.T, ch=config.ch, ch_mult=config.ch_mult, attn=config.attn,
                                num_res_blocks=config.num_res_blocks, dropout=config.dropout)
            images = generator(config)
            
        else:
            images = []
            desc = "generating images"
            for i in trange(0, config.num_images, config.batch_size, desc=desc):
                batch_size = min(config.batch_size, config.num_images - i)
                x_T = torch.randn((batch_size, 3, config.img_size, config.img_size))
                batch_images = sampler(x_T.to(device)).cpu()
                images.append((batch_images + 1) / 2) # convert the pixel value into 0-1
            images = torch.cat(images, dim=0).numpy()
            with open('generated_images.pkl', 'wb') as file:
                pickle.dump(images, file)
        model.train()
        (IS, IS_std), FID = get_inception_and_fid_score(images, config.fid_cache, num_images=None,
            use_torch=config.fid_use_torch, verbose=True,device=config.device)
        # print((IS, IS_std), FID, images)
        return (IS, IS_std), FID, images


def train():
    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    # select the part of the entire dataset
    # config.seed
    # samples_num = config.num_subset
    # random_indices = torch.randperm(len(dataset))[:samples_num]
    # subset_dataset = torch.utils.data.Subset(dataset, random_indices)
    # dataloader = torch.utils.data.DataLoader(
    #     subset_dataset, batch_size=config.batch_size, shuffle=True,
    #     num_workers=config.num_workers, drop_last=True)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True)
    
    datalooper = infiniteloop(dataloader)

    # calculate the mu and sigma
    if not os.path.exists(config.fid_cache):
        images = []
        for i in range(len(dataloader)):
            batch_images = next(datalooper).to(device).cpu()
            images.append((batch_images + 1) / 2) # convert the pixel value into 0-1
        images = torch.cat(images, dim=0).numpy()
        m, s = get_mu_sigma(
            images, num_images=config.num_subset,
            use_torch=config.fid_use_torch, verbose=True)
        fid_statistic = {}
        fid_statistic['mu'] = m
        fid_statistic['sigma'] = s
        print(m,s)
        np.savez('stats/fid_statistic.npz', **fid_statistic)

   
    # model setup
    net_model = UNet(
        T=config.T, ch=config.ch, ch_mult=config.ch_mult, attn=config.attn,
        num_res_blocks=config.num_res_blocks, dropout=config.dropout)
    ema_model = copy.deepcopy(net_model)
    optimizer = config.optimizer(net_model.parameters(), lr=config.lr)
    # optimizer = mechanize(config.optimizer)(net_model.parameters(), lr=config.lr)  # use mechanic
    sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    
    trainer = GaussianDiffusionTrainer(
        net_model, config.T, beta_schedule = config.noise_schedule).to(device)
    
    
    net_sampler = GaussianDiffusionSampler(
        net_model, config.beta_1, config.beta_T, config.T, config.img_size,
        config.mean_type, config.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, config.beta_1, config.beta_T, config.T, config.img_size,
        config.mean_type, config.var_type).to(device)
    if config.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    model_path = os.path.join(config.logdir,"ckpt_.pt") # load the retrained model in the last time
    print(model_path)
    if os.path.exists(model_path):
        model_dict = torch.load(model_path)
        net_model.load_state_dict(model_dict["net_model"])
        ema_model.load_state_dict(model_dict["ema_model"])
        sched.load_state_dict(model_dict["sched"])
        optimizer.load_state_dict(model_dict["optim"])

    if not os.path.exists(os.path.join(config.logdir, 'sample')):
        os.makedirs(os.path.join(config.logdir, 'sample'))
    x_T = torch.randn(config.sample_size, 3, config.img_size, config.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:config.sample_size]) + 1) / 2
    writer = SummaryWriter(config.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
 
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # print the size of weight parameters
    size = []
    for name, parameters in net_model.named_parameters():
        if parameters.requires_grad and len(list(parameters.size()))>3 and parameters.size()[3]==3:
            if parameters.size not in size:
                size.append(parameters.data.size()[:2])
    print(size)
    statis = {i:0 for i in size}
    for element in size:
        
        statis[element]+=1
    print(statis)



    # generate a coefficient tensor
    config.seed

    kernel = torch.ones(3, 3).unsqueeze(0).unsqueeze(0).to(config.device)
    c_tensor = kernel.repeat(256,256,1,1)
    # sum = c_tensor.sum(dim=tuple(range(1,len(c_tensor.size()))),keepdim=True)
    # coefficient = c_tensor/sum
    coefficient = c_tensor  # for adaptive learning of coefficient only by gradients 

   
    # train!
    for epoch in range(config.total_epochs):
        loss_epoch = []
        with trange(len(dataloader), dynamic_ncols=True) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for i in pbar:
                optimizer.zero_grad()
                # print(batch)
                x_0 = next(datalooper).to(device)
                # x_0 = x_0.to(device)
                # print(x_0)
                loss = trainer(x_0).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), config.grad_clip)
                tem = optimizer.step(i,coefficient,device=config.device)
                coefficient = tem
                sched.step() 
                ema(net_model, ema_model, config.ema_decay)

                # log
                writer.add_scalar('loss', loss, i)
                pbar.set_postfix(loss='%.3f' % loss)
                loss_epoch.append(loss.detach().item())
        print(torch.tensor(loss_epoch).mean())     
        # sample
        if (epoch % config.sample_epoch == 0) or ((epoch+1) == config.total_epochs):
        # if (epoch > 0 and epoch % config.sample_epoch == 0) or ((epoch+1) == config.total_epochs): # when load the precious checkpoints
            net_model.eval()
            with torch.no_grad():
                x_0 = ema_sampler(x_T)
                grid = (make_grid(x_0) + 1) / 2
                path = os.path.join(
                    config.logdir, 'sample', '%d.png' % epoch)
                save_image(grid, path)
                writer.add_image('sample', grid, epoch)
            net_model.train()

        # save
        if (epoch % config.save_epoch == 0) or ((epoch+1) == config.total_epochs):
        # if (epoch > 0 and epoch % config.sample_epoch == 0) or ((epoch+1) == config.total_epochs): # when load the precious checkpoints
            ckpt = {
                'net_model': net_model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'sched': sched.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch,
                'x_T': x_T,
            }
            torch.save(ckpt, os.path.join(config.logdir, 'ckpt_{}.pt'.format(epoch)))

        # evaluate
        if (epoch % config.eval_epoch == 0) or ((epoch+1) == config.total_epochs):
        # if (epoch > 0 and epoch % config.sample_epoch == 0) or ((epoch+1) == config.total_epochs): # when load the precious checkpoints
            net_IS, net_FID, _ = evaluate(net_sampler, net_model, epoch)
            metrics = {
                'IS': float(net_IS[0].astype(float)), # numpy convert to float type, self
                'IS_std': float(net_IS[1].astype(float)),
                'FID': net_FID   
            }
            pbar.write(
                "%d/%d " % (epoch, config.total_epochs) +
                ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
            for name, value in metrics.items():
                writer.add_scalar(name, value, epoch)
            writer.flush()
            with open(os.path.join(config.logdir, 'eval.txt'), 'a') as f:
                metrics['epoch'] = epoch
                f.write(json.dumps(metrics) + "\n")
            
    
    writer.close()
    # return net_FID, net_IS  # for optuna optimization
    

def eval():
    # model setup
    model = UNet(
        T=config.T, ch=config.ch, ch_mult=config.ch_mult, attn=config.attn,
        num_res_blocks=config.num_res_blocks, dropout=config.dropout)
    sampler = GaussianDiffusionSampler(
        model, config.beta_1, config.beta_T, config.T, img_size=config.img_size,
        mean_type=config.mean_type, var_type=config.var_type).to(device)
    if config.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(config.logdir, 'ckpt_2000.pt'))
    model.load_state_dict(ckpt['net_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(config.logdir, 'samples.png'),
        nrow=16)

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(config.logdir, 'samples_ema.png'),
        nrow=16)


# Optuna to HPO
# def objective(trail):
#     batch_size = trail.suggest_categorical("batch_size", [16, 32, 64, 128])
#     optimizer = trail.suggest_categorical("optimizer", ["Adam","RMSprop","SGD", "ACProp", "Adan", "Lion"])
#     noise_schedule = trail.suggest_categorical("noise_schedule", ["linear","cosine","sigmoid"])
#     dict_opt = {"ACProp":ACProp,"Adan":Adan,"Lion":Lion}
#     config.batch_size = batch_size

#     if optimizer in ["Adam","RMSprop","SGD"]: 
#         optimizer = getattr(optim, optimizer)
#     else:
#         optimizer = dict_opt[optimizer]
#     config.optimizer = optimizer
#     config.noise_schedule = noise_schedule
     
#     trainer = train()
#     return trainer


# if __name__ == '__main__':

#     study_name = "ddpm_final_4"  # Unique identifier of the study.
#     storage_name = "sqlite:///{}.db".format(study_name)
#     # study = optuna.create_study(study_name= study_name, storage=storage_name, directions=["minimize", "maximize"])  # create a new study
#     study = optuna.create_study(study_name=study_name, storage=storage_name, directions=["minimize", "maximize"], load_if_exists=True) # resume a study, resume的时候不要忘记注释掉上一条命令
#     df = study.trials_dataframe(attrs=("number", "params", "state"))
#     optuna.visualization.plot_pareto_front(study, target_names=["FID", "IS"])
#     # study.optimize(objective, n_trials=20)
#     # with open("sampler.pkl", "wb") as fout:
#     #     pickle.dump(study.sampler, fout)
#     # print(study.best_params)
#     # print(study.best_trials)
#     # print(study.best_trials)
#     # df = study.trials_dataframe(attrs=("number", "params", "state"))
#     # # optuna.visualization.plot_contour(study)
#     # # optuna.visualization.plot_param_importances(study).show()
#     # # optuna.visualization.plot_optimization_history(study).show()
#     # # optuna.visualization.plot_slice(study).show()
#     # print(df)


def main():
    train()

if __name__ == '__main__':
    main()

