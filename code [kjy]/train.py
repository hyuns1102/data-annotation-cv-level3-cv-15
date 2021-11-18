import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

from datetime import datetime
import wandb
import myconfig

import numpy as np
import random
import augmentation as A
from sklearn.metrics import f1_score
from deteval import calc_deteval_metrics

def seed_everything(seed):
    """
    random seed를 고정하기위한 함수
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/merge_dataset'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'trained_models'))
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=15)
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def get_optimizer(optimizer_option, model, learning_rate):
    optimizer = None

    if optimizer_option.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_option.lower() == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    elif optimizer_option.lower() == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_option.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_option.lower() == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif optimizer_option.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    elif optimizer_option.lower() == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)
    elif optimizer_option.lower() == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    return optimizer

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, seed, train_json='train_1', valid_json=None,
                isshuffle=False, optimizer_option='adam', isWandb=False):
    
    seed_everything(seed)

    my_transform = A.ComposedTransformation(
            rotate_range=30, crop_aspect_ratio=1.0, crop_size=(0.2, 0.2),
            hflip=True, vflip=True, random_translate=False,
            resize_to=512,
            min_image_overlap=0.9, min_bbox_overlap=0.99, min_bbox_count=1, allow_partial_occurrence=True,
            max_random_trials=1000,
        )
    train_dataset = SceneTextDataset(data_dir, split=train_json, image_size=image_size, crop_size=input_size, transform=None)
    train_dataset = EASTDataset(train_dataset)
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=isshuffle, num_workers=num_workers)

    valid_dataset = None
    valid_num_batches = -1
    valid_loader = None
    if valid_json != None:
        valid_dataset = SceneTextDataset(data_dir, split=valid_json, image_size=image_size, crop_size=input_size, transform=None)
        valid_dataset = EASTDataset(valid_dataset)
        valid_num_batches = math.ceil(len(valid_dataset) / batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=isshuffle, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    model = EAST()
    model.to(device)
    
    optimizer = get_optimizer(optimizer_option, model, learning_rate)    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    best_epoch = -1
    best_iou_loss = 9999999
    
    for epoch in range(max_epoch):
        train_epoch_loss, train_epoch_start = 0, time.time()
        train_epoch_extra_info = dict()
        train_epoch_extra_info['angle_loss'] = 0
        train_epoch_extra_info['cls_loss'] = 0
        train_epoch_extra_info['iou_loss'] = 0

        model.train()

        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                train_epoch_loss += loss_val

                ################################################################
                if extra_info['angle_loss'] != None:
                    train_epoch_extra_info['angle_loss'] += extra_info['angle_loss']

                if extra_info['cls_loss'] != None:
                    train_epoch_extra_info['cls_loss'] += extra_info['cls_loss']

                if extra_info['iou_loss'] != None:
                    train_epoch_extra_info['iou_loss'] += extra_info['iou_loss']
                ################################################################

                pbar.update(1)
                val_dict = {
                    'Angle loss': extra_info['angle_loss'],
                    'Cls loss': extra_info['cls_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)
        scheduler.step()
        
        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            train_epoch_loss / train_num_batches, timedelta(seconds=time.time() - train_epoch_start)))

        if valid_json != None:
            valid_epoch_loss, valid_epoch_start = 0, time.time()
            valid_epoch_extra_info = dict()
            valid_epoch_extra_info['angle_loss'] = 0
            valid_epoch_extra_info['cls_loss'] = 0
            valid_epoch_extra_info['iou_loss'] = 0

            model.eval()
            with tqdm(total=valid_num_batches) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
                    pbar.set_description('[Epoch {}]'.format(epoch + 1))

                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_val = loss.item()
                    valid_epoch_loss += loss_val

                    ################################################################
                    if extra_info['angle_loss'] != None:
                        valid_epoch_extra_info['angle_loss'] += extra_info['angle_loss']

                    if extra_info['cls_loss'] != None:
                        valid_epoch_extra_info['cls_loss'] += extra_info['cls_loss']

                    if extra_info['iou_loss'] != None:
                        valid_epoch_extra_info['iou_loss'] += extra_info['iou_loss']
                    ################################################################

                    pbar.update(1)
                    val_dict = {
                        'Angle loss': extra_info['angle_loss'],
                        'Cls loss': extra_info['cls_loss'],
                        'IoU loss': extra_info['iou_loss']
                    }
                    pbar.set_postfix(val_dict)
            scheduler.step()
        
            print('Mean loss: {:.4f} | Elapsed time: {}'.format(
                valid_epoch_loss / valid_num_batches, timedelta(seconds=time.time() - valid_epoch_start)))

        if isWandb:
            if valid_json == None:
                wandb.log({'train_loss' : train_epoch_loss/train_num_batches, 'train_angle_loss' : train_epoch_extra_info['angle_loss']/train_num_batches,
                            'train_cls_loss' : train_epoch_extra_info['cls_loss']/train_num_batches, 'train_iou_loss' : train_epoch_extra_info['iou_loss']/train_num_batches,
                            'learning_rate' : learning_rate})
            else:
                wandb.log({'train_loss' : train_epoch_loss/train_num_batches, 'train_angle_loss' : train_epoch_extra_info['angle_loss']/train_num_batches,
                            'train_cls_loss' : train_epoch_extra_info['cls_loss']/train_num_batches, 'train_iou_loss' : train_epoch_extra_info['iou_loss']/train_num_batches,
                            'val_loss' : valid_epoch_loss/valid_num_batches, 'val_angle_loss' : valid_epoch_extra_info['angle_loss']/valid_num_batches,
                            'val_cls_loss' : valid_epoch_extra_info['cls_loss']/valid_num_batches, 'val_iou_loss' : valid_epoch_extra_info['iou_loss']/valid_num_batches,
                            'learning_rate' : learning_rate})
        
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            cur_iou_loss = (valid_epoch_extra_info['iou_loss'] / valid_num_batches)
            
            if best_iou_loss > cur_iou_loss:
                try:
                    pre_modelname = osp.join(model_dir, f'epoch_{best_epoch}_{best_iou_loss}.pth')
                    if os.path.exists(pre_modelname):
                        os.remove(pre_modelname)
                    if os.path.exists(osp.join(model_dir, 'latest.pth')):
                        os.remove(osp.join(model_dir, 'latest.pth'))
                except Exception as error:
                    print(error)

                best_epoch = epoch + 1
                best_iou_loss = cur_iou_loss
                ckpt_fpath = osp.join(model_dir, f'epoch_{best_epoch}_{best_iou_loss}.pth')
                torch.save(model.state_dict(), ckpt_fpath)
                ckpt_fpath = osp.join(model_dir, 'latest.pth')
                torch.save(model.state_dict(), ckpt_fpath)

def main(args):
    do_training(**args.__dict__)

Index = [0]
def my_main(args=None):
    Index[0] += 1
    with wandb.init() as run:
        run.name = run.config.name + f"_{Index[0]}"
        hparams = run.config
        
        do_training(hparams['data_dir'], hparams['model_dir'], hparams['device'], hparams['image_size'], hparams['input_size'], hparams['num_workers'],
                    hparams['batch_size'], hparams['learning_rate'], hparams['max_epoch'], hparams['save_interval'], seed=hparams['seed'],
                    train_json=hparams['train_dataset'], valid_json=hparams['valid_dataset'], isshuffle=hparams['dataset_shuffle'],
                    optimizer_option=hparams['optimizer'], isWandb=True)

        run.finish()

if __name__ == '__main__':
    args = parse_args()
    # main(args)
    
    sweep_id = wandb.sweep(myconfig.my_config, project="OCR")
    wandb.agent(sweep_id, function=my_main, count=1)