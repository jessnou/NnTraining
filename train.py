import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
torch.cuda.empty_cache()
from data import SegmentationDataset
from BaseUnet import UNet
from transforms import train_transform, val_transform
from config import *

# Определение цветов классов
masks_dir = train_masks
all_colors = set()
for mask_name in os.listdir(masks_dir):
    mask = Image.open(os.path.join(masks_dir, mask_name)).convert("RGB")
    colors = np.array(mask).reshape(-1, 3)
    unique_colors = set(tuple(c) for c in colors)
    all_colors.update(unique_colors)

class_colors = sorted(list(all_colors))
print(f"Классы: {class_colors}")

train_dataset = SegmentationDataset(train_images, train_masks, class_colors, transform=train_transform)
test_dataset = SegmentationDataset(test_images, test_masks, class_colors, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(num_classes=len(class_colors)).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir='runs/unet_custom_v3')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

checkpoint_path = 'best_model_v2_43.pth'

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']
    print(f"Загружен чекпоинт. Продолжаем с эпохи {start_epoch}")
except Exception as e:
    print(f"Не удалось загрузить чекпоинт: {e}")
    start_epoch = 0
    best_val_loss = float('inf')

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device).long()

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        val_loop = tqdm(test_loader, total=len(test_loader), desc="Validation")
        for images, masks in val_loop:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_val_loss += loss.item()
            val_loop.set_postfix(val_loss=loss.item())

    avg_val_loss = running_val_loss / len(test_loader)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)

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
