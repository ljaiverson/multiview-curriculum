# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2021-01-18 00:18:21
# @Last Modified by:   Jun Luo
# @Last Modified time: 2021-03-03 00:26:28

import torch
import torch.nn as nn
from torchvision import models


class DualViewVGG16(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DualViewVGG16, self).__init__()
        self.feature_extractor_1 = models.vgg16(pretrained=pretrained).features # pretrained on ImageNet
        self.feature_extractor_2 = models.vgg16(pretrained=pretrained).features # pretrained on ImageNet
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flat = nn.Flatten()

        self.merge_feature = nn.Sequential(*[
                                    nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                                    ])

        self.merge_classifier = models.vgg16(pretrained=pretrained).classifier
        self.merge_classifier[6] = nn.Linear(self.merge_classifier[6].in_features, num_classes)


        # branch for view1 and branch for view 2
        self.classifier_v1 = models.vgg16(pretrained=pretrained).classifier
        self.classifier_v2 = models.vgg16(pretrained=pretrained).classifier
        self.classifier_v1[6] = nn.Linear(self.classifier_v1[6].in_features, num_classes)
        self.classifier_v2[6] = nn.Linear(self.classifier_v2[6].in_features, num_classes)


    def forward(self, x1, x2):
        x1 = self.feature_extractor_1(x1)
        x2 = self.feature_extractor_2(x2)
        x3 = self.merge_feature(torch.cat((x1, x2), 1))

        x1 = self.avgpool(x1)
        x1 = self.flat(x1)
        x2 = self.avgpool(x2)
        x2 = self.flat(x2)
        x3 = self.avgpool(x3)
        x3 = self.flat(x3)

        x1 = self.classifier_v1(x1)
        x2 = self.classifier_v2(x2)

        x3 = self.merge_classifier(x3)

        return x3, x1, x2




