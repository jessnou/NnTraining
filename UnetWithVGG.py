import torch
import torch.nn as nn
from torchvision.models import vgg16

import torch
import torch.nn as nn
from torchvision import models

class UNetVGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        features = list(vgg.features.children())

        # Энкодер: блоки свёрток
        self.enc1 = nn.Sequential(*features[0:4])    # Conv1_1, Conv1_2 + ReLU
        self.pool1 = features[4]                      # MaxPool

        self.enc2 = nn.Sequential(*features[5:9])    # Conv2_1, Conv2_2 + ReLU
        self.pool2 = features[9]                      # MaxPool

        self.enc3 = nn.Sequential(*features[10:16])  # Conv3_1, Conv3_2, Conv3_3 + ReLU
        self.pool3 = features[16]                     # MaxPool

        self.enc4 = nn.Sequential(*features[17:23])  # Conv4_1, Conv4_2, Conv4_3 + ReLU
        self.pool4 = features[23]                     # MaxPool

        self.enc5 = nn.Sequential(*features[24:30])  # Conv5_1, Conv5_2, Conv5_3 + ReLU
        self.pool5 = nn.MaxPool2d(2, 2)               # В оригинальном VGG MaxPool после Conv5 тоже есть

        # Боттлнек (с твоим DoubleConv)
        self.bottleneck = DoubleConv(512, 1024)

        # Декодер
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(1024, 512)

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # Заморозка всех слоев VGG
        for param in vgg.parameters():
            param.requires_grad = False

        # Разморозка слоев enc3 и enc4 для дообучения
        for param in self.enc3.parameters():
            param.requires_grad = True
        for param in self.enc4.parameters():
            param.requires_grad = True

    def forward(self, x):
        d1 = self.enc1(x)
        p1 = self.pool1(d1)

        d2 = self.enc2(p1)
        p2 = self.pool2(d2)

        d3 = self.enc3(p2)
        p3 = self.pool3(d3)

        d4 = self.enc4(p3)
        p4 = self.pool4(d4)

        d5 = self.enc5(p4)
        p5 = self.pool5(d5)

        bn = self.bottleneck(p5)

        up5 = self.up5(bn)
        merge5 = torch.cat([up5, d5], dim=1)
        c5 = self.conv5(merge5)

        up4 = self.up4(c5)
        merge4 = torch.cat([up4, d4], dim=1)
        c4 = self.conv4(merge4)

        up3 = self.up3(c4)
        merge3 = torch.cat([up3, d3], dim=1)
        c3 = self.conv3(merge3)

        up2 = self.up2(c3)
        merge2 = torch.cat([up2, d2], dim=1)
        c2 = self.conv2(merge2)

        up1 = self.up1(c2)
        merge1 = torch.cat([up1, d1], dim=1)
        c1 = self.conv1(merge1)

        return self.out_conv(c1)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)