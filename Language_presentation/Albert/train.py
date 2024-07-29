"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""

""" Training Config & Helper Classes  """

import os
import json
from typing import NamedTuple
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import pickle

class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 8
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, data_iter, test_iter, optimizer, evaluate_mlm, evaluate_sop, save_dir, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.data_iter = data_iter # iterator to load train data
        self.test_iter = test_iter # iterator to load test data
        self.evaluate_mlm = evaluate_mlm
        self.evaluate_sop = evaluate_sop
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device # device name

    def train(self, get_loss, model_file=None, data_parallel=False):

        
        record = {"train":{"loss_mlm":[],"loss_sop":[]},"test":{"accuracy_mlm":[],"accuracy_sop":[]}} # record the result of all epochs
        """ Train Loop """
        self.model.train() # train mode
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum_mlm, loss_sum_sop = 0., 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                loss_mlm, loss_sop = get_loss(model, batch, global_step) # mean() for Data Parallelism
                loss_sum_mlm += loss_mlm.detach().cpu()
                loss_sum_sop += loss_sop.detach().cpu()
                loss = loss_mlm+loss_sop
                loss.backward()
                self.optimizer.step()

                global_step += 1
                
                iter_bar.set_description('Iter (loss_sum_mlm=%5.3f loss_sum_sop=%5.3f)'%(loss_mlm, loss_sop))

            record["train"]["loss_mlm"].append(loss_sum_mlm/(i+1))  # record average loss of mlm in one epoch
            record["train"]["loss_sop"].append(loss_sum_sop/(i+1)) # record average loss of sop in one epoch
            # self.save(e)
            # model_file = os.path.join("saved",'model_steps_'+str(e)+'.pt')
            results = self.eval(self.evaluate_mlm,self.evaluate_sop,model_file,data_parallel=False)
            self.model.train()
            record["test"]["accuracy_mlm"].append(results["mlm"])
            record["test"]["accuracy_sop"].append(results["sop"])
            print('Epoch %d/%d : Average Loss_mlm %5.3f Average Loss_sop %5.3f'%(e+1, self.cfg.n_epochs, loss_sum_mlm/(i+1), loss_sum_sop/(i+1)))
        
        with open("dis_record.pkl",'wb') as file:
            pickle.dump(record, file)



    def eval(self, evaluate_mlm, evaluate_sop, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = {'mlm':0,'sop':0} # prediction results
        iter_bar = tqdm(self.test_iter, desc='Iter (loss=X.XXX)')
        accuracy_mlm_sum, accuracy_sop_sum = 0., 0.
        for i, batch in enumerate(iter_bar):
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                accuracy_mlm, accuracy_sop = evaluate_mlm(model, batch), evaluate_sop(model, batch)
                print(accuracy_mlm, accuracy_sop)
                accuracy_mlm_sum += accuracy_mlm 
                accuracy_sop_sum += accuracy_sop
            iter_bar.set_description('Iter(acc_mlm=%5.3f acc_sop=%5.3f)'%(accuracy_mlm,accuracy_sop))
        results['mlm'] = accuracy_mlm_sum/len(iter_bar)
        results['sop'] = accuracy_sop_sum/len(iter_bar)
        return results

    def load(self, model_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))
        
    

