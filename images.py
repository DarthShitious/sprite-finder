import os
import random
import math
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class SpriteOverlayDataset(Dataset):
    def __init__(self, background_dir, sprite_path, num_samples=1000):
        self.background_dir = background_dir
        self.sprite = Image.open(sprite_path).convert("RGBA")
        self.num_samples = num_samples
        self.bg_paths = [os.path.join(background_dir, f) for f in os.listdir(background_dir)]
        self.bg_paths = [p for p in self.bg_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.bg_paths:
            raise ValueError("No valid background images found in the specified directory.")
        if not os.path.exists(background_dir):
            raise ValueError(f"Background directory {background_dir} does not exist.")
        if not os.path.exists(sprite_path):
            raise ValueError(f"Sprite image {sprite_path} does not exist.")
        if not self.sprite:
            raise ValueError(f"Sprite image {sprite_path} is empty or invalid.")
        self.crop_size = 256

        self.bg_transform = T.Compose([
            T.RandomCrop(self.crop_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Resize((512,512), interpolation=Image.BICUBIC)])
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load and transform background
        repeat = True
        while repeat:
            bg_path = random.choice(self.bg_paths)
            bg = Image.open(bg_path).convert("RGB")
            if bg.size[0] >= self.crop_size and bg.size[1] >= self.crop_size:
                repeat = False
                bg = self.bg_transform(bg)

        # Generate random rotation
        theta = random.uniform(0, 2 * math.pi)
        rotated_sprite = self.sprite.rotate(-math.degrees(theta), expand=True)

        # Calculate position for sprite
        max_x = self.crop_size - rotated_sprite.width//2
        max_y = self.crop_size - rotated_sprite.height//2
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Create composite
        composite = bg.copy().convert("RGBA")
        composite.paste(rotated_sprite, (x - rotated_sprite.width//2, y - rotated_sprite.height//2), rotated_sprite)

        # Occlusion augmentation: paste a chunk of background over the image
        if random.random() < 1.0:
            patch_size = random.choice([32, 64, 128, 256])
            px = random.randint(0, 512 - patch_size)
            py = random.randint(0, 512 - patch_size)

            # Grab patch from background and paste onto composite
            bg_patch = bg.crop((px, py, px + patch_size, py + patch_size)).convert("RGBA")
            composite.paste(bg_patch, (px, py))

        # Convert to tensor (drop alpha)
        composite = composite.convert("RGB")
        image_tensor = self.to_tensor(composite)

        # Normalize label (x, y in [0, 1])
        label = torch.tensor([
            x / 512,
            y / 512,
            math.cos(theta),
            math.sin(theta)
        ], dtype=torch.float32)


        return image_tensor, label


import matplotlib.pyplot as plt

def show_sample(dataset, idx=None, save_path=None):
    if idx is None:
        idx = random.randint(0, len(dataset)-1)
        
    img_tensor, label = dataset[idx]
    img_np = img_tensor.permute(1, 2, 0).numpy()
    x, y, cos_theta, sin_theta = label.tolist()

    # Convert normalized x/y back to pixel coords
    x_px = int(x * 512)
    y_px = int(y * 512)
    
    # Get angle from cos/sin
    angle_deg = math.degrees(math.atan2(sin_theta, cos_theta))

    plt.figure(figsize=(5,5))
    plt.imshow(img_np)
    plt.scatter([x_px], [y_px], c='red', s=50, label=f"({x_px}, {y_px}) @ {angle_deg:.1f}°")
    plt.title("Sprite Location and Rotation")
    plt.legend()
    plt.axis('off')
    plt.savefig(save_path) if save_path else plt.show()
    plt.close()
