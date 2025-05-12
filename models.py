import torch
import torch.nn as nn
import torchvision.models as models
# from torchvision.models import efficientnet_b0
import warnings

class GhostBatchNorm(nn.Module):
    """
    Ghost Batch Normalization: Simulate larger batch norm behavior
    by splitting into smaller "ghost" mini-batches.
    Automatically falls back to standard BN if batch is too small or incompatible.
    """
    def __init__(self, num_features, ghost_batch_size=2, momentum=0.1, eps=1e-5, affine=True):
        super().__init__()
        self.ghost_batch_size = ghost_batch_size
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        self.bn = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps, affine=affine)

    def forward(self, x):
        orig_shape = x.shape

        if x.dim() == 2:
            B, C = x.shape
            if B < self.ghost_batch_size or B % self.ghost_batch_size != 0:
                warnings.warn(f"GhostBN fallback to standard BN: batch size {B} not divisible by ghost size {self.ghost_batch_size}")
                return self.bn(x)

            G = B // self.ghost_batch_size
            x = x.view(G, self.ghost_batch_size, C)
            x = x.contiguous().view(-1, C)
            x = self.bn(x)
            return x.view(B, C)

        elif x.dim() >= 3:
            B = x.shape[0]
            C = x.shape[1]
            rest = x.shape[2:]
            if B < self.ghost_batch_size or B % self.ghost_batch_size != 0:
                warnings.warn(f"GhostBN fallback to standard BN: batch size {B} not divisible by ghost size {self.ghost_batch_size}")
                return self.bn(x)

            G = B // self.ghost_batch_size
            x = x.view(G, self.ghost_batch_size, C, *rest)
            x = x.contiguous().view(-1, C, *rest)
            x = self.bn(x)
            return x.view(B, C, *rest)

        else:
            raise ValueError("GhostBatchNorm only supports 2D or higher tensors")


def convert_batchnorm_to_ghostnorm(model, ghost_batch_size=16):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            ghost_bn = GhostBatchNorm(
                num_features=module.num_features,
                ghost_batch_size=ghost_batch_size,
                momentum=module.momentum,
                eps=module.eps,
                affine=module.affine,
            )
            setattr(model, name, ghost_bn)
        else:
            convert_batchnorm_to_ghostnorm(module, ghost_batch_size=ghost_batch_size)


class ResNetSplitHeadSpritePredictor(nn.Module):
    def __init__(self,
                 backbone_name="resnet18",
                 pretrained=True,
                 ghost_batch_size=2,
                 trunk_layers=[1024, 1024],
                 pred_head_layers=[512, 128],
                 logvar_head_layers=[512, 128]):
        """
        Args:
            backbone_name (str): Which torchvision resnet backbone to use.
            pretrained (bool): Whether to use pretrained weights.
            trunk_layers (list): Hidden sizes for trunk MLP.
            pred_head_layers (list): Hidden sizes for prediction branch.
            logvar_head_layers (list): Hidden sizes for uncertainty branch.
        """
        super().__init__()



        # --- ZScore normalization ---
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # --- Backbone ---
        backbone_fn = getattr(models, backbone_name)
        backbone = backbone_fn(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove last fc

        # Replace batch norms with ghost norms
        self.ghost_batch_size = ghost_batch_size
        convert_batchnorm_to_ghostnorm(self.backbone, ghost_batch_size=self.ghost_batch_size)

        # # --- Freeze batchnorm in backbone
        # self.freeze_bn(self.backbone)

        backbone_out_features = list(backbone.children())[-1].in_features

        # --- LayerNorm ---
        self.ln = torch.nn.LayerNorm(backbone_out_features)

        # --- Ghost Batchnorm ---
        self.gn = GhostBatchNorm(backbone_out_features, ghost_batch_size=self.ghost_batch_size)

        # --- Dropout ---
        self.dropout = torch.nn.Dropout(0.05)

        # --- Trunk (shared MLP) ---
        self.trunk = self._make_mlp(backbone_out_features, trunk_layers)

        # --- Prediction head (for (x,y,cos,sin)) ---
        self.pred_head = self._make_mlp(trunk_layers[-1], pred_head_layers, final_out=4)

        # --- Uncertainty head (for logvars) ---
        self.logvar_head = self._make_mlp(trunk_layers[-1], logvar_head_layers, final_out=4)

    @staticmethod
    def freeze_bn(model):
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()
                module.requires_grad_(False)

    def _make_mlp(self, in_features, hidden_layers, final_out=None):
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(GhostBatchNorm(hidden_dim, ghost_batch_size=self.ghost_batch_size))
            layers.append(nn.LeakyReLU(0.01))
            in_features = hidden_dim
        if final_out is not None:
            layers.append(nn.Linear(in_features, final_out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.mean) / self.std # Normalize input
        x = self.backbone(x)           # (B, backbone_features, 1, 1)
        x = x.flatten(1)               # (B, backbone_features)
        x = self.gn(x)
        x = self.dropout(x)
        x = self.trunk(x)              # (B, trunk_output)

        preds = self.pred_head(x)      # (B, 4)
        logvars = self.logvar_head(x)  # (B, 4)

        return torch.cat([preds, logvars], dim=-1)  # (B, 8)


class SpritePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(256 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 8)  # (x, y, cos, sin) + 4 logvars
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)
