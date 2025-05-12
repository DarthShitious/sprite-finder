import os
import random
import math
import numpy as np
from PIL import Image, ImageChops, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def generate_alpha_mask(size, threshold=0.5, blur_radius=8):
    noise = np.random.rand(*size).astype(np.float32)
    mask = (noise > threshold).astype(np.float32)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return mask_img

def create_composite(background_img, sprite_img, x, y, angle_rad, do_occlusion=True):
    composite = background_img.copy().convert("RGBA")
    rotated_sprite = sprite_img.rotate(-math.degrees(angle_rad), expand=True)

    # 50% chance to apply destruction
    if random.random() < 0.5:
        mask = generate_alpha_mask(rotated_sprite.size, threshold=0.7, blur_radius=6)
        alpha = rotated_sprite.getchannel('A')
        new_alpha = ImageChops.multiply(alpha, mask)
        rotated_sprite.putalpha(new_alpha)

    composite.paste(rotated_sprite, (x - rotated_sprite.width // 2, y - rotated_sprite.height // 2), rotated_sprite)

    return composite.convert("RGB")

def make_label(x, y, angle_rad, img_size=512):
    return torch.tensor([
        x / img_size,
        y / img_size,
        math.cos(angle_rad),
        math.sin(angle_rad)
    ], dtype=torch.float32)

class SpriteDataset(Dataset):
    def __init__(self, background_paths, sprite_path, samples=None, num_samples=1000, random_mode=True):
        self.background_paths = background_paths
        self.sprite_path = sprite_path
        self.samples = samples
        self.num_samples = num_samples
        self.random_mode = random_mode
        self._setup()

    def _setup(self):
        self.sprite_img = Image.open(self.sprite_path).convert("RGBA")

        self.crop_size = 256
        self.final_size = 512

        self.bg_transform = T.Compose([
            T.RandomCrop(self.crop_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Resize((self.final_size, self.final_size), interpolation=Image.BICUBIC)
        ])

        self.to_tensor = T.ToTensor()

    def regenerate(self):
        """
        Re-initialize any randomized components (sprite augmentation, etc.)
        """
        self._setup()

    def __len__(self):
        return len(self.samples) if self.samples else self.num_samples

    def __getitem__(self, idx):
        bg_path = random.choice(self.background_paths)
        bg = Image.open(bg_path).convert("RGB")

        if bg.size[0] < self.crop_size or bg.size[1] < self.crop_size:
            scale_factor = max(self.crop_size / bg.size[0], self.crop_size / bg.size[1])
            new_size = (int(bg.size[0] * scale_factor) + 1, int(bg.size[1] * scale_factor) + 1)
            bg = bg.resize(new_size, Image.BICUBIC)

        bg = self.bg_transform(bg)

        if self.random_mode:
            x = random.randint(0, self.final_size)
            y = random.randint(0, self.final_size)
            angle_rad = random.uniform(0, 2 * math.pi)
        else:
            x, y, angle_rad = self.samples[idx]

        img = create_composite(bg, self.sprite_img, x, y, angle_rad)
        img_tensor = self.to_tensor(img)
        label = make_label(x, y, angle_rad, img_size=self.final_size)

        return img_tensor, label

def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
