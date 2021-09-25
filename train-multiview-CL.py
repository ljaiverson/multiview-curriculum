# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2021-02-23 08:47:57
# @Last Modified by:   Jun Luo
# @Last Modified time: 2021-03-03 13:17:24

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

import numpy as np

from utils.datasets import DualViewDataset
from utils.models import DualViewVGG16
from utils.evaluator import getAUC

import platform
import os
import argparse
import shutil

def os_checking():
    if platform.system().upper() == 'WINDOWS':
        torch.multiprocessing.freeze_support()
        print(platform.system())

def load_model(model, state_dict, view=None):
    own_state = model.state_dict()

    # 1. filter out unnecessary keys
    view_num = view[1]
    temp = {('feature_extractor_%s.' % view_num )+ '.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if k.startswith('features')}
    state_dict ={**temp, **{('classifier_v%s.' % view_num )+ '.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if k.startswith('classifier')}}
    
    print("==> Loading the following parameters...")
    for k in state_dict:
        print(k)
    # 2. overwrite entries in the existing state dict
    own_state.update(state_dict) 
    # 3. load the new state dict
    model.load_state_dict(own_state)
    return

def get_parser(add_help=False):
    parser = argparse.ArgumentParser(
        add_help=add_help,
        description='Multiview classification on the elbow fracture dataset')

    # region arguments yapf: disable
    parser.add_argument('--lr', '-l',
        default=0.0001,
        type=float,
        help='Learning Rate')
    parser.add_argument('--batch_size', '-b',
        default=64,
        type=int,
        help='batch_size')
    parser.add_argument('--num_epochs', '-e',
        default=30,
        type=int,
        help='Number of epochs')
    parser.add_argument('--data_dir', '-d',
        default='data',
        type=str,
        help='Data folder')
    parser.add_argument('--weights_folder', '-wf',
        default='weights',
        type=str,
        help='Trained weights folder')
    parser.add_argument('--checkpoints_folder', '-cf',
        default='checkpoints',
        type=str,
        help='Trained checkpoints folder')
    parser.add_argument('--pretrained', '-pt',
        default='single-view',
        choices=["single-view", ""],
        type=str,
        help='Use which kind of pretrained (from "single-view" / No pretrain (""))')
    parser.add_argument('--epoch_equal', '-ee',
        default=16,
        type=int,
        help="Epoch number after which sampling prob's. are 1/N w.r.t. CL")

    parser.set_defaults(print_log=False)
    return parser

def main():
    os_checking()
    parser = get_parser()
    args = parser.parse_args()
    img_rows, img_cols = 299, 299
    end_epoch = args.num_epochs - 1

    if not os.path.exists(args.weights_folder):
        os.makedirs(args.weights_folder)
    if not os.path.exists(args.checkpoints_folder):
        os.makedirs(args.checkpoints_folder)

    ######################################## Data ########################################
    torch.cuda.empty_cache()
    train_dataset = DualViewDataset(img_rows, img_cols, args.data_dir, phase='train', use_transformer=True, 
                                curriculum='knowledge', epochs_to_equal_prob=args.epoch_equal)
    # Sampling-based CL, train_loader defined in Epochs for loop
    val_dataset = DualViewDataset(img_rows, img_cols, args.data_dir, phase='val', use_transformer=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    class_names = list(train_dataset.get_stats().keys())
    print('Class names:', class_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######################################## Model ########################################
    model = DualViewVGG16(len(class_names), pretrained=True) # pretrained on ImageNet
    criterion = nn.CrossEntropyLoss()
    criterion_v1 = nn.CrossEntropyLoss()
    criterion_v2 = nn.CrossEntropyLoss()

    # homogeneous transfer learning
    if args.pretrained.upper() == 'SINGLE-VIEW':
        for i, view in enumerate(['frontal', 'lateral']):
            model_sv_fn = os.path.join(args.weights_folder, '%s.pth' % (view))
            assert os.path.exists(model_sv_fn), "Using pretrained single view model. However, single view model does not exist!"
            load_model(model, torch.load(model_sv_fn)['net'], view='V%d' % (i+1))
        print('\n********** Finished loading pretrained single-view models **********\n')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    val_auc_list = []

    ######################################## Training ########################################
    for epoch in range(args.num_epochs):
        sampler = WeightedRandomSampler(weights=train_dataset.get_ps(),
                                        num_samples=len(train_dataset),
                                        replacement=False)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False, 
                                                   sampler=sampler,
                                                   num_workers=4)

        train(model, optimizer, criterion, criterion_v1, criterion_v2, train_loader, device, epoch, end_epoch)
        val(model, val_loader, device, val_auc_list, args.checkpoints_folder, epoch, end_epoch)
        train_dataset.update_ps()

    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('==> Epoch %s is the best model' % (index+1))
    best_model_fn = os.path.join(args.checkpoints_folder, 'ckpt_%d.pth' % (index+1))
    if args.pretrained == '':
        best_weights_copy_to = 'multiview_CL.pth'
    elif args.pretrained.upper() == 'SINGLE-VIEW':
        best_weights_copy_to = 'multiview_TL_CL.pth'
    best_weights_copy_to = os.path.join(args.weights_folder, best_weights_copy_to)

    shutil.copyfile(best_model_fn, best_weights_copy_to)
    print('\nModel saved at %s' % (best_weights_copy_to))

def train(model, optimizer, criterion, criterion_v1, criterion_v2, train_loader, device, epoch, end_epoch):
    model.train()
    for batch_idx, (x1, x2, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs, pred1, pred2 = model(x1.to(device), x2.to(device))
        targets = targets.squeeze().long().to(device)
        try:
            num_targets = len(targets)
            targets = targets.long().resize_(num_targets)
        except TypeError:
            num_targets = 1
            targets = targets.long().resize_(num_targets)
        loss = criterion(outputs, targets) + criterion_v1(pred1, targets) + criterion_v2(pred2, targets)
        loss.backward()
        optimizer.step()
        print('******* Epoch [%3d/%3d], Phase: Training, Batch [%4d/%4d], loss = %.8f *******' % 
                (epoch+1, end_epoch+1, batch_idx+1, len(train_loader), loss.item()))

def val(model, val_loader, device, val_auc_list, weights_folder, epoch, end_epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (x1, x2, targets) in enumerate(val_loader):
            outputs, _1, _2 = model(x1.to(device), x2.to(device))
            targets = targets.squeeze().long().to(device)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            try:
                num_targets = len(targets)
                targets = targets.float().resize_(num_targets)
            except TypeError:
                num_targets = 1
                targets = targets.float().resize_(num_targets)
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)
            print('******* Epoch [%3d/%3d], Phase: Validation, Batch [%4d/%4d] *******' % 
                (epoch+1, end_epoch+1, batch_idx+1, len(val_loader)))

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, 'multi-class')
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    print('\n******* Epoch [%3d/%3d], Validation AUC: %.5f *******\n' % (epoch+1, end_epoch+1, auc))

    path = os.path.join(weights_folder, 'ckpt_%d.pth' % (epoch+1))
    torch.save(state, path)



if __name__ == '__main__':
    main()