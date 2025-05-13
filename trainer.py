import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import imageio
import glob
from tqdm import tqdm
from loss import combined_loss



class Trainer:
    def __init__(
            self, 
            model, 
            optimizer,
            scheduler, 
            train_loader, 
            val_loader, 
            device, 
            save_dir="results/exp001", 
            kl_weight=0.0, 
            ece_weight=0.0, 
            cal_weight=0.0
        ):

        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "val_grids"), exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.train_gnlls = []
        self.val_gnlls = []
        self.train_eces = []
        self.val_eces = []
        self.train_cal_losses = []
        self.val_cal_losses = []

        self.kl_weight = kl_weight
        self.ece_weight = ece_weight
        self.cal_weight = cal_weight

        self.current_epoch = 0

    def train_epoch(self, accum_steps=1):
        for param_group in self.optimizer.param_groups:
            print(f"Current LR: {param_group['lr']}")

        self.model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_gnll = 0.0
        running_ece = 0.0
        running_cal_loss = 0.0

        self.optimizer.zero_grad()

        for step, (imgs, labels) in enumerate(self.train_loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            outputs = self.model(imgs)
            loss, gnll_loss, ece_loss, cal_loss = combined_loss(
                outputs[:, :4],
                labels,
                outputs[:, 4:],
                ece_weight=self.ece_weight,
                cal_weight=self.cal_weight,
            )

            # Normalize loss for accumulation
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)  # Optional but recommended
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += loss.item() * accum_steps  # Undo loss normalization for tracking
            running_mae += self.compute_mae(outputs[:, :4], labels)
            running_gnll += gnll_loss.item()
            running_ece += ece_loss.item()
            running_cal_loss += cal_loss.item()

        self.scheduler.step()

        self.train_losses.append(running_loss / len(self.train_loader))
        self.train_maes.append(running_mae / len(self.train_loader))
        self.train_gnlls.append(running_gnll / len(self.train_loader))
        self.train_eces.append(running_ece / len(self.train_loader))
        self.train_cal_losses.append(running_cal_loss / len(self.train_loader))


    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        running_mae = 0.0
        running_gnll = 0.0
        running_ece = 0.0
        running_cal_loss = 0.0

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss, gnll_loss, ece_loss, cal_loss = combined_loss(outputs[:, :4],
                    labels,
                    outputs[:, 4:],
                    ece_weight=self.ece_weight,
                    cal_weight=self.cal_weight,
                )

                running_loss += loss.item()
                running_mae += self.compute_mae(outputs[:, :4], labels)
                running_gnll += gnll_loss.item()
                running_ece += ece_loss.item()
                running_cal_loss += cal_loss.item()

        self.val_losses.append(running_loss / len(self.val_loader))
        self.val_maes.append(running_mae / len(self.val_loader))
        self.val_gnlls.append(running_gnll / len(self.val_loader))
        self.val_eces.append(running_ece / len(self.val_loader))
        self.val_cal_losses.append(running_cal_loss / len(self.val_loader))

    def compute_mae(self, preds, targets):
        return torch.abs(preds - targets).mean().item()

    def save_all(self):
        imgs, labels, preds, uncertainties = self.collect_validation_outputs()

        # New: Create per-epoch subdirectory
        epoch_dir = os.path.join(self.save_dir, f"epoch{self.current_epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        self.save_metrics_plots(epoch_dir)
        self.save_val_predictions_grid(imgs, labels, preds, uncertainties, epoch_dir)
        self.save_label_vs_pred_scatter(labels, preds, uncertainties, epoch_dir)
        self.save_histograms(labels, preds, epoch_dir)
        self.save_uncertainty_2d_histograms(labels, preds, uncertainties, epoch_dir)
        self.save_calibration_curve(labels, preds, uncertainties, epoch_dir)
        print(f"[INFO] Saved all results for epoch {self.current_epoch} to {epoch_dir}")

    def collect_validation_outputs(self):
        self.model.eval()
        all_imgs = []
        all_labels = []
        all_preds = []
        all_uncertainties = []

        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="Collecting validation outputs"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                preds = outputs[:, :4]
                logvars = outputs[:, 4:]
                unc = torch.exp(0.5 * logvars)

                all_imgs.append(imgs.cpu())
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())
                all_uncertainties.append(unc.cpu())

        imgs = torch.cat(all_imgs, dim=0)
        labels = torch.cat(all_labels, dim=0)
        preds = torch.cat(all_preds, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0)

        return imgs, labels, preds, uncertainties

    def save_metrics_plots(self, save_dir):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure()
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss vs Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss_vs_epoch.png"))
        plt.close()

        plt.figure()
        plt.plot(epochs, self.train_maes, label='Train MAE')
        plt.plot(epochs, self.val_maes, label='Val MAE')
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.title("MAE vs Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mae_vs_epoch.png"))
        plt.close()

        plt.figure()
        plt.plot(epochs, self.train_gnlls, label='Train GNLL')
        plt.plot(epochs, self.val_gnlls, label='Val GNLL')
        plt.xlabel("Epoch")
        plt.ylabel("GNLL")
        plt.legend()
        plt.title("GNLL vs Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "gnll_vs_epoch.png"))
        plt.close()

        plt.figure()
        plt.plot(epochs, self.train_eces, label='Train ECE')
        plt.plot(epochs, self.val_eces, label='Val ECE')
        plt.xlabel("Epoch")
        plt.ylabel("ECE")
        plt.legend()
        plt.title("ECE vs Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ece_vs_epoch.png"))
        plt.close()

        plt.figure()
        plt.plot(epochs, self.train_cal_losses, label='Train Calibration Loss')
        plt.plot(epochs, self.val_cal_losses, label='Val Calibration Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Calibration Loss")
        plt.legend()
        plt.title("Calibration Loss vs Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "calibration_loss_vs_epoch.png"))
        plt.close()

    def save_val_predictions_grid(self, imgs, labels, preds, uncertainties, save_dir, n_cols=4):
        n_samples = min(n_cols, imgs.size(0))

        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 2 * 4))

        for i in range(n_samples):
            img = imgs[i].permute(1, 2, 0).numpy()
            label = labels[i]
            pred = preds[i]
            unc = uncertainties[i]

            x_l, y_l = int(label[0]*512), int(label[1]*512)
            angle_l = math.degrees(math.atan2(label[3], label[2]))

            x_p, y_p = int(pred[0]*512), int(pred[1]*512)
            angle_p = math.degrees(math.atan2(pred[3], pred[2]))

            mean_unc = unc.mean().item()
            pred_color = self.uncertainty_to_color(mean_unc)

            # Top row: GT
            ax = axes[0, i]
            ax.imshow(img)
            ax.scatter([x_l], [y_l], c='green', s=50)
            self.draw_arrow(ax, x_l, y_l, angle_l, color='green')
            ax.axis('off')
            ax.set_title(f"GT ({x_l},{y_l}) {angle_l:.1f}°")

            # Bottom row: Prediction
            ax = axes[1, i]
            ax.imshow(img)
            ax.scatter([x_p], [y_p], color=pred_color, s=50)
            self.draw_arrow(ax, x_p, y_p, angle_p, color='red')
            ax.axis('off')
            ax.set_title(f"Pred ({x_p},{y_p}) {angle_p:.1f}°\nUnc: {mean_unc:.3f}")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "val_grid.png"))
        plt.close()

    def save_label_vs_pred_scatter(self, labels, preds, uncertainties, save_dir):
        labels_names = ["x", "y", "cos", "sin"]

        for i in range(4):
            plt.figure(figsize=(6,6))
            unc_values = uncertainties[:,i]
            scatter = plt.scatter(labels[:,i], preds[:,i], c=unc_values, cmap='rainbow', s=10, vmin=0.0, vmax=0.6)
            plt.colorbar(scatter, label="Uncertainty")
            plt.xlabel(f"True {labels_names[i]}")
            plt.ylabel(f"Predicted {labels_names[i]}")
            plt.title(f"{labels_names[i]}: Labels vs Predictions (colored by Uncertainty)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"scatter_{labels_names[i]}.png"))
            plt.close()

    def save_histograms(self, labels, preds, save_dir):
        labels_names = ["x", "y", "cos", "sin"]

        for i in range(4):
            plt.figure(figsize=(6,4))
            plt.hist(labels[:,i], bins=30, alpha=0.5, label="True", color="g", density=True)
            plt.hist(preds[:,i], bins=30, alpha=0.5, label="Predicted", color="r", density=True)
            plt.xlabel(labels_names[i])
            plt.ylabel("Density")
            plt.title(f"{labels_names[i]} Distribution")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"histogram_{labels_names[i]}.png"))
            plt.close()

    def save_uncertainty_2d_histograms(self, labels, preds, uncertainties, save_dir):
        labels_names = ["x", "y", "cos", "sin"]

        for i in range(4):
            # Label vs Uncertainty
            plt.figure(figsize=(6,5))
            plt.hist2d(labels[:,i], uncertainties[:,i], bins=50, cmap='rainbow', norm=mcolors.LogNorm())
            plt.colorbar(label='Log Count')
            plt.xlabel(f"True {labels_names[i]}")
            plt.ylabel("Uncertainty")
            plt.title(f"Label vs Uncertainty ({labels_names[i]})")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"uncertainty_hist2d_label_{labels_names[i]}.png"))
            plt.close()

            # Prediction vs Uncertainty
            plt.figure(figsize=(6,5))
            plt.hist2d(preds[:,i], uncertainties[:,i], bins=50, cmap='rainbow', norm=mcolors.LogNorm())
            plt.colorbar(label='Log Count')
            plt.xlabel(f"Predicted {labels_names[i]}")
            plt.ylabel("Uncertainty")
            plt.title(f"Prediction vs Uncertainty ({labels_names[i]})")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"uncertainty_hist2d_pred_{labels_names[i]}.png"))
            plt.close()

    def save_calibration_curve(self, labels, preds, uncertainties, save_dir, n_bins=10):
        """
        Create and save a calibration curve plot: 
        predicted uncertainty vs actual prediction error.
        """
        labels_names = ["x", "y", "cos", "sin"]

        for i in range(4):
            pred = preds[:, i]
            label = labels[:, i]
            unc = uncertainties[:, i]

            # Calculate absolute error
            error = torch.abs(pred - label)

            # Bin by predicted uncertainty
            unc_np = unc.numpy()
            error_np = error.numpy()

            bins = np.linspace(0.0, unc_np.max(), n_bins+1)
            bin_indices = np.digitize(unc_np, bins) - 1  # Bin 0..n_bins-1

            avg_unc = []
            avg_err = []

            for b in range(n_bins):
                mask = bin_indices == b
                if mask.any():
                    avg_unc.append(unc_np[mask].mean())
                    avg_err.append(error_np[mask].mean())

            # Plot
            plt.figure(figsize=(6,6))
            plt.plot(avg_unc, avg_err, marker='o', label='Model Calibration')
            plt.plot([0, max(avg_unc)], [0, max(avg_unc)], '--k', label='Perfect Calibration')
            plt.xlabel("Predicted Uncertainty")
            plt.ylabel("Actual Error")
            plt.title(f"Calibration Curve: {labels_names[i]}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"calibration_curve_{labels_names[i]}.png"))
            plt.close()

    def draw_arrow(self, ax, x, y, angle_deg, length=40, color='blue'):
        angle_rad = math.radians(angle_deg)
        dx = length * math.cos(angle_rad)
        dy = length * math.sin(angle_rad)
        ax.arrow(x, y, dx, dy, head_width=10, head_length=10, fc=color, ec=color)

    def uncertainty_to_color(self, uncertainty, vmin=0.0, vmax=0.6):
        norm_unc = (uncertainty - vmin) / (vmax - vmin)
        norm_unc = max(0.0, min(1.0, norm_unc))
        hue_deg = 120 + norm_unc * (270 - 120)
        hue = hue_deg / 360.0
        saturation = 1.0
        value = 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return (r, g, b)

def make_gif(save_dir, output_name="val_grid.gif", duration=0.5):
    images = []
    files = sorted(glob.glob(os.path.join(save_dir, "val_grids/epoch_*.png")))

    for filename in files:
        images.append(imageio.imread(filename))

    gif_path = os.path.join(save_dir, output_name)
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"[INFO] Saved GIF: {gif_path}")
