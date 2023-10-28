#!/usr/bin/env python3

"""code to train STERLING representations from spot data"""

import torch
torch.multiprocessing.set_sharing_strategy('file_system') #https://github.com/pytorch/pytorch/issues/11201
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pickle
import glob
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F
import tensorboard as tb
import cv2
from tqdm import tqdm

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime
import argparse
import yaml
import os
from sklearn import metrics

webhook_url = "https://hooks.slack.com/services/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"

from scipy.signal import periodogram
from scripts.models import VisualEncoder, IPTEncoder, VisualEncoderTiny
from scripts.utils import process_feet_data, get_transforms

import albumentations as A
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from termcolor import cprint

FEET_TOPIC_RATE = 24.0
LEG_TOPIC_RATE = 24.0
IMU_TOPIC_RATE = 200.0

class TerrainDataset(Dataset):
    def __init__(self, pickle_files_root,
                 data_stats=None, img_augment=False):
        self.pickle_files_paths = glob.glob(pickle_files_root + '/*.pkl')
        self.label = pickle_files_root.split('/')[-2]
        self.data_stats = data_stats
        
        if self.data_stats is not None:
            self.min, self.max = data_stats['min'], data_stats['max']
            self.mean, self.std = data_stats['mean'], data_stats['std']
        
        if img_augment:
            cprint('Using image augmentations', 'green', attrs=['bold', 'underline'])
            self.transforms = get_transforms()
        else:
            self.transforms = None
    
    def __len__(self):
        return len(self.pickle_files_paths)
    
    def __getitem__(self, idx):
        with open(self.pickle_files_paths[idx], 'rb') as f:
            data = pickle.load(f)
        imu, feet, leg = data['imu'], data['feet'], data['leg']
        patches = data['patches']
    
        # process the feet data to remove the mu and std values for non-contacting feet
        feet = process_feet_data(feet)
        
        imu = imu[:, :-4] # donot include the orientation data
        # select only columns 0, 1, 5
        imu = imu[:, [0, 1, 5]] # angular_x, angular_y, linear_z

        imu = periodogram(imu, fs=IMU_TOPIC_RATE, axis=0)[1]
        leg = periodogram(leg, fs=LEG_TOPIC_RATE, axis=0)[1]
        feet = periodogram(feet, fs=FEET_TOPIC_RATE, axis=0)[1]
        imu, leg, feet = imu[-201:, :], leg[-25:, :], feet[-25:, :]
        
        # normalize the imu data
        # if self.mean is not None and self.std is not None:
        if self.data_stats is not None:
            # #minmax normalization
            imu = (imu - self.min['imu']) / (self.max['imu'] - self.min['imu'] + 1e-7)
            leg = (leg - self.min['leg']) / (self.max['leg'] - self.min['leg'] + 1e-7)
            feet = (feet - self.min['feet']) / (self.max['feet'] - self.min['feet'] + 1e-7)
            
        # sample 2 values between 0 and num_patches-1
        patch_1_idx, patch_2_idx = np.random.choice(len(patches), 2, replace=False)

        patch1, patch2 = patches[patch_1_idx], patches[patch_2_idx]
        
        # convert BGR to RGB
        patch1, patch2 = cv2.cvtColor(patch1, cv2.COLOR_BGR2RGB), cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB)
        
        # apply the transforms
        if self.transforms is not None:
            patch1 = self.transforms(image=patch1)['image']
            patch2 = self.transforms(image=patch2)['image']
        
        # normalize the image patches
        patch1 = np.asarray(patch1, dtype=np.float32) / 255.0
        patch2 = np.asarray(patch2, dtype=np.float32) / 255.0
        
        # transpose
        patch1, patch2 = np.transpose(patch1, (2, 0, 1)), np.transpose(patch2, (2, 0, 1))
        
        return np.asarray(patch1), np.asarray(patch2), imu, leg, feet, self.label, idx

# create pytorch lightning data module
class STERLINGDataModule(pl.LightningDataModule):
    def __init__(self, data_config_path, batch_size=64, num_workers=4):
        super().__init__()
        
        # read the yaml file
        cprint('Reading the yaml file at : {}'.format(data_config_path), 'green')
        self.data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)
        self.data_config_path = '/'.join(data_config_path.split('/')[:-1])

        self.batch_size, self.num_workers = batch_size, num_workers
        
        self.mean, self.std = {}, {}
        self.min, self.max = {}, {}
        
        # load the train and val datasets
        self.data_statistics_pkl_path = self.data_config_path + '/data_statistics.pkl'
        if 'all' in data_config_path:
            cprint('This is the pretraining step...', 'green')
            self.data_statistics_pkl_path = self.data_config_path + '/data_statistics_all.pkl'
        cprint('data_statistics_pkl_path : {}'.format(self.data_statistics_pkl_path), 'green')
        self.load()
        cprint('Train dataset size : {}'.format(len(self.train_dataset)), 'green')
        cprint('Val dataset size : {}'.format(len(self.val_dataset)), 'green')
        
        
    def load(self):
        # check if the data_statistics.pkl file exists
        if os.path.exists(self.data_statistics_pkl_path):
            cprint('Loading the mean and std from the data_statistics pickle file', 'green')
            data_statistics = pickle.load(open(self.data_statistics_pkl_path, 'rb'))
            # self.mean, self.std = data_statistics['mean'], data_statistics['std']
            # self.min, self.max = data_statistics['min'], data_statistics['max']
            
        else:
            # find the mean and std of the train dataset
            cprint('data_statistics pickle file not found!', 'yellow')
            cprint('Finding the mean and std of the train dataset', 'green')
            self.tmp_dataset = ConcatDataset([TerrainDataset(pickle_files_root) for pickle_files_root in self.data_config['train']])
            self.tmp_dataloader = DataLoader(self.tmp_dataset, batch_size=128, num_workers=10, shuffle=True)
            cprint('the length of the tmp_dataloader is : {}'.format(len(self.tmp_dataloader)), 'green')
            # find the mean and std of the train dataset
            imu_data, leg_data, feet_data = [], [], []
            for _, _, imu, leg, feet, _, _ in tqdm(self.tmp_dataloader):
                imu_data.append(imu.cpu().numpy())
                leg_data.append(leg.cpu().numpy())
                feet_data.append(feet.cpu().numpy())
            imu_data = np.concatenate(imu_data, axis=0)
            leg_data = np.concatenate(leg_data, axis=0)
            feet_data = np.concatenate(feet_data, axis=0)
            
            self.mean['imu'], self.std['imu'] = np.mean(imu_data, axis=0), np.std(imu_data, axis=0)
            self.min['imu'], self.max['imu'] = np.min(imu_data, axis=0), np.max(imu_data, axis=0)
            
            self.mean['leg'], self.std['leg'] = np.mean(leg_data, axis=0), np.std(leg_data, axis=0)
            self.min['leg'], self.max['leg'] = np.min(leg_data, axis=0), np.max(leg_data, axis=0)
            
            self.mean['feet'], self.std['feet'] = np.mean(feet_data, axis=0), np.std(feet_data, axis=0)
            self.min['feet'], self.max['feet'] = np.min(feet_data, axis=0), np.max(feet_data, axis=0)
            
            cprint('Mean : {}'.format(self.mean), 'green')
            cprint('Std : {}'.format(self.std), 'green')
            cprint('Min : {}'.format(self.min), 'green')
            cprint('Max : {}'.format(self.max), 'green')
            
            # save the mean and std
            cprint('Saving the mean, std, min, max to the data_statistics pickle file', 'green')
            data_statistics = {'mean': self.mean, 'std': self.std, 'min': self.min, 'max': self.max}
            
            pickle.dump(data_statistics, open(self.data_statistics_pkl_path, 'wb'))
            
        # load the train data
        self.train_dataset = ConcatDataset([TerrainDataset(pickle_files_root, data_stats=data_statistics, img_augment=True) for pickle_files_root in self.data_config['train']])
        self.val_dataset = ConcatDataset([TerrainDataset(pickle_files_root, data_stats=data_statistics) for pickle_files_root in self.data_config['val']])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True, 
                          drop_last= True if len(self.train_dataset) % self.batch_size != 0 else False,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True, 
                          drop_last= True if len(self.val_dataset) % self.batch_size != 0 else False,
                          pin_memory=True)

class STERLINGModel(pl.LightningModule):
    def __init__(self, lr=3e-4, latent_size=64, scale_loss=1.0/32, lambd=3.9e-6, 
                 weight_decay=1e-6, l1_coeff=0.5, rep_size=64):
        super(STERLINGModel, self).__init__()
        
        self.save_hyperparameters(
            'lr',
            'latent_size',
            'weight_decay',
            'l1_coeff',
            'rep_size',
        )
        
        self.best_val_loss = 1000000.0
        
        self.lr, self.latent_size, self.scale_loss, self.lambd, self.weight_decay = lr, latent_size, scale_loss, lambd, weight_decay
        self.l1_coeff = l1_coeff
        self.rep_size = rep_size
        
        # visual encoder architecture
        self.visual_encoder = VisualEncoderTiny(latent_size=rep_size)
        self.ipt_encoder = IPTEncoder(latent_size=rep_size)
        
        self.projector = nn.Sequential(
            nn.Linear(rep_size, latent_size), nn.PReLU(),
            nn.Linear(latent_size, latent_size)
        )
        
        # coefficients for vicreg loss
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0
        
        self.max_acc = None
        
    def forward(self, patch1, patch2, inertial_data, leg, feet):
        v_encoded_1 = self.visual_encoder(patch1.float())
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2.float())
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)        
        
        # i_encoded = self.inertial_encoder(inertial_data.float())
        i_encoded = self.ipt_encoder(inertial_data.float(), leg.float(), feet.float())
        
        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)
        zi = self.projector(i_encoded)
        
        return zv1, zv2, zi, v_encoded_1, v_encoded_2, i_encoded
    
    def vicreg_loss(self, z1, z2):
        repr_loss = F.mse_loss(z1, z2)

        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        cov_x = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_y = (z2.T @ z2) / (z2.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div_(z1.shape[1]) + self.off_diagonal(cov_y).pow_(2).sum().div_(z2.shape[1])
  
        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss
    
    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def all_reduce(self, c):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)
            
    def training_step(self, batch, batch_idx):
        patch1, patch2, inertial, leg, feet, label, _ = batch

        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial, leg, feet)
        
        # compute viewpoint invariance vicreg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        # compute visual-inertial vicreg loss
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)
        
        loss = self.l1_coeff * loss_vpt_inv + (1.0-self.l1_coeff) * loss_vi
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_vpt_inv', loss_vpt_inv, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_vi', loss_vi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        patch1, patch2, inertial, leg, feet, label, _ = batch
        
        # combine inertial and leg data
        # inertial = torch.cat((inertial, leg, feet), dim=-1)
        
        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial, leg, feet)
        
        # compute viewpoint invariance vicreg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        # compute visual-inertial vicreg loss
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)
        
        loss = self.l1_coeff * loss_vpt_inv + (1.0-self.l1_coeff) * loss_vi
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_vpt_inv', loss_vpt_inv, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_vi', loss_vi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
    
    def on_validation_batch_start(self, batch, batch_idx):
        # save the batch data only every other epoch or during the last epoch
        if self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs-1:
            patch1, patch2, inertial, leg, feet, label, sampleidx = batch
        
            with torch.no_grad():
                _, _, _, zv1, zv2, zi = self.forward(patch1, patch2, inertial, leg, feet)
            zv1, zi = zv1.cpu(), zi.cpu()
            patch1 = patch1.cpu()
            label = np.asarray(label)
            sampleidx = sampleidx.cpu()
            
            if batch_idx == 0:
                self.visual_encoding = [zv1]
                self.inertial_encoding = [zi]
                self.label = label
                self.visual_patch = [patch1]
                self.sampleidx = [sampleidx]
            else:
                self.visual_encoding.append(zv1)
                self.inertial_encoding.append(zi)
                self.label = np.concatenate((self.label, label))
                self.visual_patch.append(patch1)
                self.sampleidx.append(sampleidx)
    
    def on_validation_end(self):
        if (self.current_epoch % 10 == 0 or \
            self.current_epoch == self.trainer.max_epochs-1) and \
                torch.cuda.current_device() == 0:
                    
            val_loss = self.trainer.callback_metrics["val_loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_models()
                
            self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            self.inertial_encoding = torch.cat(self.inertial_encoding, dim=0)
            self.visual_patch = torch.cat(self.visual_patch, dim=0)
            self.sampleidx = torch.cat(self.sampleidx, dim=0)
            
            # randomize index selections
            idx = np.arange(self.visual_encoding.shape[0])
            np.random.shuffle(idx)
            
            # limit the number of samples to 2000
            ve = self.visual_encoding#[idx[:2000]]
            vi = self.inertial_encoding#[idx[:2000]]
            vis_patch = self.visual_patch#[idx[:2000]]
            ll = self.label#[idx[:2000]]
            
            data = torch.cat((ve, vi), dim=-1)
            
            if self.current_epoch % 10 == 0:
                self.logger.experiment.add_embedding(mat=data[idx[:2500]], label_img=vis_patch[idx[:2500]], global_step=self.current_epoch, metadata=ll[idx[:2500]], tag='visual_encoding')
            del self.visual_patch, self.visual_encoding, self.inertial_encoding, self.label
    
    def save_models(self, path_root='./models/'):
        cprint('saving models...', 'yellow', attrs=['bold'])
        if not os.path.exists(path_root): 
            cprint('creating directory: ' + path_root, 'yellow')
            os.makedirs(path_root)
        else:
            cprint('directory already exists: ' + path_root, 'red')
        
        # save the visual encoder
        torch.save(self.visual_encoder.state_dict(), os.path.join(path_root, 'visual_encoder.pt'))
        cprint('visual encoder saved', 'green')
        
        # save the proprioceptive encoder
        torch.save(self.ipt_encoder.state_dict(), os.path.join(path_root, 'proprioceptive_encoder.pt'))
        cprint('proprioceptive encoder saved', 'green')
        cprint('All models successfully saved', 'green', attrs=['bold'])
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train representations using the STERLING framework')
    parser.add_argument('--batch_size', '-b', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--l1_coeff', type=float, default=0.5, metavar='L1C',
                        help='L1 loss coefficient (1)')
    parser.add_argument('--num_gpus','-g', type=int, default=2, metavar='N',
                        help='number of GPUs to use (default: 8)')
    parser.add_argument('--latent_size', type=int, default=64, metavar='N',
                        help='Size of the common latent space (default: 128)')
    parser.add_argument('--data_config_path', type=str, default='spot_data/data_config.yaml')
    args = parser.parse_args()
    
    model = STERLINGModel(lr=args.lr, latent_size=args.latent_size, l1_coeff=args.l1_coeff)
    dm = STERLINGDataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="sterling_logs/")
    
    print("Training the representation learning model...")
    trainer = pl.Trainer(devices=args.num_gpus,
                         max_epochs=args.epochs,
                         log_every_n_steps=10,
                         strategy='ddp',
                        #  num_sanity_val_steps=0,
                         logger=tb_logger,
                         sync_batchnorm=True,
                         gradient_clip_val=10.0,
                         gradient_clip_algorithm='norm',
                         deterministic=True,
                         )

    # fit the model
    trainer.fit(model, dm)
    
    
    

