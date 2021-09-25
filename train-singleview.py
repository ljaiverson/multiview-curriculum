# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2021-01-18 02:26:12
# @Last Modified by:   Jun Luo
# @Last Modified time: 2021-02-21 08:29:22

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import numpy as np

from utils.datasets import SingleViewDataset
from utils.evaluator import getAUC

import platform
import os
import argparse
import shutil

def os_checking():
    if platform.system().upper() == 'WINDOWS':
        torch.multiprocessing.freeze_support()
        print(platform.system())

def get_parser(add_help=False):
    parser = argparse.ArgumentParser(
        add_help=add_help,
        description='Single view classification on the elbow fracture dataset')

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
    parser.add_argument('--view', '-v',
        default='frontal',
        type=str,
        help='What view')

    parser.set_defaults(print_log=False)
    return parser

def main():
    os_checking()
    parser = get_parser()
    args = parser.parse_args()
    img_rows, img_cols = 299, 299
    end_epoch = args.num_epochs - 1

    args.checkpoints_folder = os.path.join(args.checkpoints_folder, args.view)
    if not os.path.exists(args.weights_folder):
        os.makedirs(args.weights_folder)
    if not os.path.exists(args.checkpoints_folder):
        os.makedirs(args.checkpoints_folder)

    ######################################## Data ########################################
    torch.cuda.empty_cache()

    train_dataset = SingleViewDataset(img_rows, img_cols, args.data_dir, 'train', args.view, use_transformer=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataset = SingleViewDataset(img_rows, img_cols, args.data_dir, 'val', args.view, use_transformer=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    class_names = list(train_dataset.get_stats().keys())
    print('Class names:', class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######################################## Model ########################################
    model = models.vgg16(pretrained=True) # pretrained on ImageNet
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, len(class_names))
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    val_auc_list = []

    ######################################## Training ########################################
    for epoch in range(args.num_epochs):
        train(model, optimizer, criterion, train_loader, device, epoch, end_epoch)
        val(model, val_loader, device, val_auc_list, args.checkpoints_folder, epoch, end_epoch)
    
    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('==> Epoch %s is the best model' % (index+1))
    best_model_fn = os.path.join(args.checkpoints_folder, 'ckpt_%d.pth' % (index+1))
    best_weights_copy_to = '%s.pth' % (args.view)
    best_weights_copy_to = os.path.join(args.weights_folder, best_weights_copy_to)

    shutil.copyfile(best_model_fn, best_weights_copy_to)
    print('\nModel saved at %s' % (best_weights_copy_to))

def train(model, optimizer, criterion, train_loader, device, epoch, end_epoch):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = targets.squeeze().long().to(device)
        try:
            num_targets = len(targets)
            targets = targets.long().resize_(num_targets)
        except TypeError:
            num_targets = 1
            targets = targets.long().resize_(num_targets)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        print('******* Epoch [%3d/%3d], Phase: Training, Batch [%4d/%4d], loss = %.8f *******' % 
                (epoch+1, end_epoch+1, batch_idx+1, len(train_loader), loss.item()))

def val(model, val_loader, device, val_auc_list, weights_folder, epoch, end_epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            targets = targets.squeeze().long().to(device)
            try:
                num_targets = len(targets)
                targets = targets.float().resize_(num_targets, 1)
            except TypeError:
                num_targets = 1
                targets = targets.float().resize_(num_targets, 1)
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