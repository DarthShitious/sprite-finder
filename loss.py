import torch



class LogitAB:
    def __init__(self, a, b, device, eps=0.2):
        """
        a, b: either scalars or 1D tensors of shape [D], where D is # of target dimensions
        """
        super().__init__()
        self.a = torch.as_tensor(a).float().to(device)
        self.b = torch.as_tensor(b).float().to(device)
        self.a_ = self.a - eps
        self.b_ = self.b + eps
        self.eps = eps

    def __call__(self, x):
        """
        x: Tensor of shape [B, D]
        """
        x = torch.clip(x, self.a, self.b)
        a = self._match_shape(x, self.a_)
        b = self._match_shape(x, self.b_)
        return torch.log((x - a) / (b - x))

    def inv(self, x):
        """
        x: Tensor of shape [B, D] (logit-space)
        """
        x = torch.clip(x, self.a, self.b)
        a = self._match_shape(x, self.a_)
        b = self._match_shape(x, self.b_)
        exp_x = torch.exp(x)
        return (a + b * exp_x) / (1 + exp_x)

    def jacobian_logdet(self, x):
        a = self._match_shape(x, self.a_)
        b = self._match_shape(x, self.b_)
        return -torch.log(torch.clamp((x - a) * (b - x), min=self.eps))

    def _match_shape(self, x, param):
        """
        Broadcast scalar or vector param to match shape of x
        """
        if param.ndim == 0:
            return param
        elif param.ndim == 1 and x.ndim >= 2:
            # Broadcast [D] to [B, D]
            return param.view(1, -1).expand_as(x)
        else:
            raise ValueError("LogitAB: 'a' and 'b' must be scalar or 1D tensor matching x.shape[1:]")

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



def gnll_loss(pred, target, logvar, scaling_func=None, apply_jacobian=True):
    if scaling_func is not None:
        pred = scaling_func(pred)
        target = scaling_func(target)
        if apply_jacobian and hasattr(scaling_func, "jacobian_logdet"):
            jacobian = scaling_func.jacobian_logdet(pred)
            logvar = logvar + 2 * jacobian

    gnll_raw = 0.5 * (logvar + ((pred - target) ** 2) / torch.exp(logvar))

    beta = 0.5
    weight = torch.exp(logvar * beta).detach()
    gnll = weight * gnll_raw

    gnll = gnll.mean()
    gnll_raw = gnll_raw.sum(dim=-1).mean()
    return [gnll, gnll_raw]

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
    gnll, gnll_raw = gnll_loss(pred, target, logvar, scaling_func, apply_jacobian=True)
    ece = ece_loss(pred, target, logvar)
    cal_loss = calibration_loss(pred, target, logvar)
    total_loss = gnll + ece_weight * ece + cal_weight * cal_loss
    print(f"Total Loss: {total_loss.item():.4f}, MAE: {mae.item():.4f}, GNLL: {gnll_raw.item():.4f}, ECE: {ece.item():.4f}, Calibration Loss: {cal_loss.item():.4f}")
    return total_loss, gnll_raw, ece, cal_loss
