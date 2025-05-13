import os
import torch
import time
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from models import SplitHeadSpritePredictor
from trainer import Trainer, make_gif
from sprite_composite import SpriteDataset
from scheduler import CustomDecay

def save_checkpoint(model, optimizer, epoch, save_path):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved to {save_path}")

def load_checkpoint(model, optimizer=None, checkpoint_path="checkpoint.pth", new_lr=None):
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded model from {checkpoint_path}")

    start_epoch = 0

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Loaded optimizer state")

        if new_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"⚙️ Learning rate updated to {new_lr}")

        start_epoch = checkpoint['epoch'] + 1

    return model, optimizer, start_epoch

def create_dataloader(dataset, batch_size):
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )

def main():
    # Paths
    background_dir = "/home/tabbott/flickr30k_images/flickr30k_images"
    sprite_path = "/home/tabbott/Sprites/Cave/PNG/Objects_separately/128/white_crystal_light_shadow2.png"

    # Save directory
    save_dir = os.path.join(os.getcwd(), time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)

    model_path = None

    # Hyperparameters
    num_samples = int(3200*1.25)
    batch_size = 64
    accum_steps = 32
    num_epochs = 20
    recycle_steps = 5
    kl_weight = 0
    ece_weight = 0.05
    cal_weight = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare datasets
    valid_exts = ('.jpg', '.jpeg', '.png')

    background_paths = [
        os.path.join(background_dir, fname)
        for fname in os.listdir(background_dir)
        if fname.lower().endswith(valid_exts)
    ]

    full_dataset = SpriteDataset(
        background_paths=background_paths,
        sprite_path=sprite_path,
        num_samples=num_samples,
        random_mode=True
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = create_dataloader(train_dataset, batch_size=batch_size)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size)

    # Create model
    model = SplitHeadSpritePredictor(backbone_name="resnet18", pretrained=True)
    model.to(device)

    # Load model checkpoint
    if model_path:
        model, _, _ = load_checkpoint(model=model, checkpoint_path=model_path)

    # Optimizer
    backbone_lr = 1e-6
    head_lr = 5e-4
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': backbone_lr},
        {'params': model.trunk.parameters(), 'lr': head_lr},
        {'params': model.pred_head.parameters(), 'lr': head_lr},
        {'params': model.logvar_head.parameters(), 'lr': head_lr}
    ], weight_decay=1e-4)

    scheduler = CustomDecay(optimizer=optimizer, total_epochs=10, min_factor=0.2)

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        kl_weight=kl_weight,
        ece_weight=ece_weight,
        cal_weight=cal_weight,
    )

    # Print model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.")

    # --- Training Loop ---
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        trainer.current_epoch = epoch

        # kl_weight = kl_annealing(epoch, total_anneal_epochs=50, max_kl_weight=0.1)
        # trainer.kl_weight = kl_weight

        trainer.train_epoch(accum_steps=accum_steps)
        trainer.validate_epoch()

        if (epoch+1) % recycle_steps == 0:
            print(f"[INFO] Recycling dataset at epoch {epoch+1}...")
            # Regenerate a fresh full dataset
            full_dataset = SpriteDataset(
                background_paths=background_paths,
                sprite_path=sprite_path,
                num_samples=num_samples,
                random_mode=True
            )
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            train_loader = create_dataloader(train_dataset, batch_size=batch_size)
            val_loader = create_dataloader(val_dataset, batch_size=batch_size)
            trainer.train_loader = train_loader
            trainer.val_loader = val_loader

        if (epoch+1) % 1 == 0:
            trainer.save_all()

        if (epoch+1) % 5 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                save_path=os.path.join(save_dir, f"model_state_{epoch:05d}.pth")
            )

    make_gif(save_dir=save_dir)

if __name__ == "__main__":
    main()
