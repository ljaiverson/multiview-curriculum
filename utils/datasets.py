# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2021-01-17 21:55:00
# @Last Modified by:   Jun Luo
# @Last Modified time: 2021-03-03 13:08:39

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import os


class DualViewDataset(Dataset):
    def __init__(self,
                 img_h,
                 img_w,
                 path,
                 phase='train',
                 view1_dir = 'frontal',
                 view2_dir = 'lateral',
                 use_transformer=True,
                 random_state=0,
                 curriculum=None,
                 epochs_to_equal_prob=16):
        super(DualViewDataset, self).__init__()

        ##################################### NOTE #####################################
        # this class assumes that the filenames in view1 and view 2 are exactly the same

        view1_folder = os.path.join(path, phase, view1_dir)
        view2_folder = os.path.join(path, phase, view2_dir)
        labels = os.listdir(view1_folder)
        self.view1_image_list = []
        self.view2_image_list = []
        self.all_labels = []
        self.curriculum = curriculum
        self.epochs_to_equal_prob = epochs_to_equal_prob

        for l in labels:
            imgs = os.listdir(os.path.join(view1_folder, l))
            self.view1_image_list += [os.path.join(view1_folder, l, fn) for fn in imgs]
            self.view2_image_list += [os.path.join(view2_folder, l, fn) for fn in imgs]
            self.all_labels += [int(l) for _ in range(len(imgs))]

        if not use_transformer:
            self.transformer = transforms.Compose([
                                    transforms.Resize((img_h, img_w)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
        else:
            self.transformer = transforms.Compose([
                                    transforms.RandomRotation(15),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Resize((img_h, img_w)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])

        stats = {str(l): 0 for l in set(self.all_labels)}
        for l in self.all_labels:
            stats[str(l)] += 1
        self.stats = {k: v for k, v in sorted(stats.items(), key=lambda item: item[0])}
        # print('Number of samples in all classes:', stats)

        if self.curriculum == 'knowledge':
            self.frontal_hard_score = {'0': 70, '1': 70, '2': 70, '3': 30, '4': 60, '5': 10, '6': 90}
            self.lateral_hard_score = {'0': 65, '1': 40, '2': 55, '3': 40, '4': 20, '5': 10, '6': 75}
            self.both_hard_score =    {'0': 55, '1': 35, '2': 45, '3': 25, '4': 15, '5': 5, '6': 70}

            # # choose one of the three options below
            # ##### option 1 # both views' score ######
            # next 2 lines uses both views' score
            max_score = 100
            self.easy_score = {k: max_score - v for k, v in self.both_hard_score.items() if k in self.stats}

            ###### option 2 # sum of lateral and frontal scores ######
            # # next 2 lines uses the sum of lateral and frontal scores
            # max_score = 200
            # self.easy_score = {k: max_score-(self.frontal_hard_score[k] + self.lateral_hard_score[k]) \
            #                     for k in self.frontal_hard_score if k in self.stats}

            ###### option 3 # ranking ######
            # self.easy_score = {'0': 1, '1': 3, '2': 2}
            
            self.qs = np.array([self.easy_score[str(lab)] for lab in self.all_labels])
            self.ps = self.qs / np.sum(self.qs)
            self.scalars = self._get_scalars_from_ps()

    def get_stats(self):
        return self.stats

    def _get_scalars_from_ps(self):
        l = len(self.ps)
        final_ps = np.array([1/l for _ in self.ps])
        times = final_ps / self.ps
        return np.array([pow(s, 1/self.epochs_to_equal_prob) for s in times])

    def update_ps(self):
        if self.epochs_to_equal_prob != 0:
            self.ps = self.ps * self.scalars
            self.epochs_to_equal_prob -= 1

    def get_ps(self):
        return self.ps

    def __getitem__(self, index):
        view1_img = self.transformer(Image.open(self.view1_image_list[index]).convert('RGB'))
        view2_img = self.transformer(Image.open(self.view2_image_list[index]).convert('RGB'))
        label = self.all_labels[index]
        return view1_img, view2_img, label

    def __len__(self):
        return len(self.all_labels)

class SingleViewDataset(Dataset):
    def __init__(self,
                 img_h,
                 img_w,
                 path,
                 phase,
                 view,
                 use_transformer=True,
                 random_state=0,):
        super(SingleViewDataset, self).__init__()
        self.view = view
        data_folder = os.path.join(path, phase, view)
        labels = os.listdir(data_folder)
        self.image_list = []
        self.all_labels = []

        for l in labels:
            class_c_imgs = [os.path.join(data_folder, l, fn) for fn in os.listdir(os.path.join(data_folder, l))]
            self.image_list += class_c_imgs
            self.all_labels += [int(l) for _ in range(len(class_c_imgs))]

        if not use_transformer:
            self.transformer = transforms.Compose([
                                    transforms.Resize((img_h, img_w)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
        else:
            self.transformer = transforms.Compose([
                                    transforms.RandomRotation(15),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Resize((img_h, img_w)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])

        stats = {str(l):0 for l in set(self.all_labels)}
        for l in self.all_labels:
            stats[str(l)] += 1
        self.stats = {k: v for k, v in sorted(stats.items(), key=lambda item: item[0])}
        # print('Number of samples in all classes:', stats)

    def get_stats(self):
        return self.stats

    def __getitem__(self, index):
        img = self.transformer(Image.open(self.image_list[index]).convert('RGB'))
        label = self.all_labels[index]
        return img, label

    def __len__(self):
        return len(self.image_list)
