from .utils.engine import DDPMSampler, DDIMSampler
from .model.UNet import UNet
import torch
import os
from .utils.tools import save_sample_image, save_image
from argparse import ArgumentParser
from tqdm import tqdm, trange


def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])

    # generator param
    parser.add_argument("-bs", "--batch_size", type=int, default=16)

    # sampler param
    parser.add_argument("--result_only", default=False, action="store_true")
    parser.add_argument("--interval", type=int, default=50)

    # DDIM sampler param
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic"])

    # save image param
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)
    parser.add_argument("--to_grayscale", default=False, action="store_true")

    args = parser.parse_args()
    return args


@torch.no_grad()
def generator(args):
    device = torch.device(args.device)
    path = os.path.join(args.logdir, 'ckpt_{}.pt'.format(args.epoch))
    print(path)
    cp = torch.load(path)
    # cp = torch.load(args.checkpoint_path)
    # load trained model
    # model = UNet(**cp["config"]["Model"])
    # model.load_state_dict(cp["model"])
    model = args.model
    model.load_state_dict(cp['net_model'])
    model.to(device)
    model = model.eval()

    
    sampler = DDIMSampler(model, (args.beta_1, args.beta_T), args.T).to(args.device)
    images = []
    desc = "generating images"
    for i in trange(0, args.num_images, args.batch_size, desc=desc):
        batch_size = min(args.batch_size, args.num_images - i)
        # generate Gaussian noise
        z_t = torch.randn((batch_size, 3, args.img_size, args.img_size), device=args.device)
        extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)
        batch_images = sampler(z_t, only_return_x_0=args.result_only, interval=args.interval, **extra_param).cpu()
        images.append((batch_images + 1) / 2) # convert the pixel value into 0-1
    images = torch.cat(images, dim=0).numpy()
    # if args.result_only:
    #     save_image(x, nrow=args.nrow, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)
    # else:
    #     save_sample_image(x, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)
    return images # self

if __name__ == "__main__":
    args = parse_option()
    generator(args)
