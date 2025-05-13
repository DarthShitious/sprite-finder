import torch


def kl_annealing(epoch, total_anneal_epochs=50, max_kl_weight=0.1):
    if epoch >= total_anneal_epochs:
        return max_kl_weight
    else:
        return max_kl_weight * (epoch / total_anneal_epochs)

def calibration_loss(pred, target, logvar):
    error = (pred - target).abs()
    unc = logvar.exp().sqrt()  # predicted stddev
    cal_loss = (error - unc).abs()
    return cal_loss.mean()

def gnll_loss(pred, target, logvar, beta=0.5):
    gnll_raw = 0.5 * (logvar + ((pred - target) ** 2) / torch.exp(logvar))
    weight = torch.exp(logvar * beta).detach()
    gnll = weight * gnll_raw
    return [gnll.mean(), gnll_raw.mean()]

def ece_loss(pred, target, logvar, n_bins=10):
    abs_error = (pred - target).abs()
    uncertainty = logvar.exp().sqrt()  # predicted stddev

    # Flatten to 1D
    abs_error = abs_error.flatten()
    uncertainty = uncertainty.flatten()

    # Create bins
    bin_edges = torch.linspace(uncertainty.min(), uncertainty.max(), n_bins+1, device=uncertainty.device)
    bin_indices = torch.bucketize(uncertainty, bin_edges) - 1  # bins are 0-indexed

    losses = []
    for b in range(n_bins):
        mask = (bin_indices == b)
        if mask.any():
            avg_uncertainty = uncertainty[mask].mean()
            avg_error = abs_error[mask].mean()
            losses.append((avg_uncertainty - avg_error).abs())
    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=pred.device)

def combined_loss(pred, target, logvar, ece_weight=0.0, cal_weight=0.0, scaling_func=None):
    mae = torch.abs(pred - target).mean()
    gnll, gnll_raw = gnll_loss(pred, target, logvar)
    ece = ece_loss(pred, target, logvar)
    cal_loss = calibration_loss(pred, target, logvar)
    total_loss = gnll + ece_weight * ece + cal_weight * cal_loss
    print(f"Total Loss: {total_loss.item():.4f}, MAE: {mae.item():.4f}, GNLL: {gnll_raw.item():.4f}, ECE: {ece.item():.4f}, Calibration Loss: {cal_loss.item():.4f}")
    return total_loss, gnll_raw, ece, cal_loss
