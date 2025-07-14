import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from PIL import Image
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 8
num_epochs = 50
lr = 0.005
num_classes = 26

masks_dir = 'Masks/Train'
all_colors = set()

for mask_name in os.listdir(masks_dir):
    mask_path = os.path.join(masks_dir, mask_name)
    mask = Image.open(mask_path).convert("RGB")
    colors = np.array(mask).reshape(-1,3)
    unique_colors = set(tuple(c) for c in colors)
    all_colors.update(unique_colors)

class_colors = sorted(list(all_colors))
print(class_colors)

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, class_colors, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.class_colors = class_colors
        self.transform = transform

        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']

        # Преобразование маски RGB → индексы классов
        mask = np.array(mask)
        mask_index = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        for idx, color in enumerate(self.class_colors):
            matches = np.all(mask == color, axis=-1)
            mask_index[matches] = idx

        mask_index = torch.from_numpy(mask_index).long()

        return image, mask_index
    
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(p=0.3),
    A.Normalize(),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2(),
])


import torch.nn as nn
import torch

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

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        up4 = self.up4(bn)
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

train_images = "Images/Train"
train_masks = "Masks/Train"

test_images = "Images/Test"
test_masks = "Masks/Test"

# Датасеты
train_dataset = SegmentationDataset(train_images, train_masks,class_colors, transform=train_transform)
test_dataset = SegmentationDataset(test_images, test_masks,class_colors, transform=val_transform)

# Даталоадеры
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(num_classes=35).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir='runs/unet_custom_v2')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_val_loss = 0.0

    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for images, masks in loop:
        # images: [B, 3, H, W]
        # masks: [B, H, W] — индексы классов!
        
        images = images.to(device)
        masks = masks.to(device).long()  # важно!

        outputs = model(images)  # [B, num_classes, H, W]

        loss = criterion(outputs, masks)  # CrossEntropyLoss ждёт так!

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)

    # ===================
    # Валидация
    # ===================
    model.eval()
    with torch.no_grad():
        val_loop = tqdm(test_loader, total=len(test_loader), desc="Validation")
        for images, masks in val_loop:
            images = images.to(device)
            masks = masks.to(device).long()  # важно!

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_val_loss += loss.item()
            val_loop.set_postfix(val_loss=loss.item())

    avg_val_loss = running_val_loss / len(test_loader)
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss
        }, f"best_model_v2_{epoch + 1}.pth")
        print(f'✅ Новый лучший чекпоинт сохранён: Val Loss = {avg_val_loss:.4f}')

    writer.add_scalar('Loss/val', avg_val_loss, epoch)
