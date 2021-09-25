# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2021-01-18 10:27:15
# @Last Modified by:   Jun Luo
# @Last Modified time: 2021-02-14 17:08:44

import torch
import torch.nn as nn
from torchvision import models

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

import numpy as np

from utils.datasets import DualViewDataset, SingleViewDataset
from utils.models import DualViewVGG16

import platform
import os
import argparse

def os_checking():
    if platform.system().upper() == 'WINDOWS':
        torch.multiprocessing.freeze_support()
        print(platform.system())

def get_parser(add_help=False):
    parser = argparse.ArgumentParser(
        add_help=add_help,
        description='Multi/single view classification on the elbow fracture dataset')
    parser.add_argument('--batch_size', '-b',
        default=16,
        type=int,
        help='batch_size')
    parser.add_argument('--data_dir', '-d',
        default='data',
        type=str,
        help='Data folder')
    parser.add_argument('--weights_folder', '-wf',
        default='weights',
        type=str,
        help='Trained weights folder')
    parser.add_argument('--weights_fn', '-wn',
        default='multiview_TL_CL.pth',
        type=str,
        help='Weigth filename')
    parser.add_argument('--view', '-v',
        default='multi',
        choices=['multi', 'frontal', 'lateral'],
        type=str,
        help='What view')

    parser.set_defaults(print_log=False)
    return parser

def main():
    os_checking()
    parser = get_parser()
    args = parser.parse_args()
    img_rows, img_cols = 299, 299

    ######################################## Data ########################################
    torch.cuda.empty_cache()
    if args.view == 'multi':
        train_dataset = DualViewDataset(img_rows, img_cols, args.data_dir, phase='train', use_transformer=False)
        val_dataset = DualViewDataset(img_rows, img_cols, args.data_dir, phase='val', use_transformer=False)
        test_dataset = DualViewDataset(img_rows, img_cols, args.data_dir, phase='test', use_transformer=False)
    else:
        train_dataset = SingleViewDataset(img_rows, img_cols, args.data_dir, 'train', args.view, use_transformer=False)
        val_dataset = SingleViewDataset(img_rows, img_cols, args.data_dir, 'val', args.view, use_transformer=False)
        test_dataset = SingleViewDataset(img_rows, img_cols, args.data_dir, 'test', args.view, use_transformer=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    class_names = list(test_dataset.get_stats().keys())
    print('Class names:', class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######################################## Model ########################################
    if args.view == 'multi':
        model = DualViewVGG16(len(class_names), pretrained=False)
    else:
        model = models.vgg16()
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_names))
    
    model.load_state_dict(torch.load(os.path.join(args.weights_folder, args.weights_fn))['net'])
    print('\n==> Model %s loaded' % os.path.join(args.weights_folder, args.weights_fn))
    model = model.to(device)

    print('==> Start testing...\n')
    print("==> acc, balanced acc, mean AUC, binary task acc, binary task auc")
    for phase, dataloader in {"train": train_loader, "val": val_loader, "test": test_loader}.items():
        test(model, phase, args.view, dataloader, device)

def test(model, phase, view, test_loader, device):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if view == 'multi':
                x1, x2, targets = data
                outputs, _1, _2 = model(x1.to(device), x2.to(device))
            else:
                inputs, targets = data
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

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        performance = evaluate(y_true, y_score)

        # output
        output  = '%.5f,' % performance['dataset_acc']
        output += '%.5f,' % np.mean(performance['cls_recalls'])
        output += '%.5f,' % np.mean(performance['cls_aucs'])
        output += '%.5f,' % performance['cls_accs'][0]
        output += '%.5f' % performance['cls_aucs'][0]
        print("%6s==>" % (phase), output)

def evaluate(y_true, y_score):
    performance = {'cls_aucs': [], 'cls_recalls': [], 'cls_accs': [], 'dataset_acc': 0}
    zero = np.zeros_like(y_true)
    one = np.ones_like(y_true)

    y_pre = np.zeros_like(y_true)
    for i in range(y_score.shape[0]):
        y_pre[i] = np.argmax(y_score[i])
    performance['dataset_acc'] = accuracy_score(y_true, y_pre)

    for i in range(y_score.shape[1]):
        y_true_binary = np.where(y_true == i, one, zero)
        y_score_binary = y_score[:, i]
        y_pred_binary = np.where(y_pre == i, one, zero)
        performance['cls_aucs'].append(roc_auc_score(y_true_binary, y_score_binary))
        performance['cls_recalls'].append(recall_score(y_true_binary, y_pred_binary))

    for i in range(y_score.shape[1]):
        y_true_binary = np.where(y_true == i, one, zero)
        y_pre_binary = np.where(y_pre == i, one, zero)
        performance['cls_accs'].append(accuracy_score(y_true_binary, y_pre_binary))

    return performance

if __name__ == '__main__':
    main()