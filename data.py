import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

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

        mask = np.array(mask)
        mask_index = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        for idx, color in enumerate(self.class_colors):
            matches = np.all(mask == color, axis=-1)
            mask_index[matches] = idx

        mask_index = torch.from_numpy(mask_index).long()

        return image, mask_index
