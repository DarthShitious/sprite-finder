import torch
import torch.nn as nn
import torchvision.models as models



class SplitHeadSpritePredictor(nn.Module):
    def __init__(self,
                 backbone_name="resnet18",
                 pretrained=True,
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

        # # --- Freeze batchnorm in backbone
        # self.freeze_bn(self.backbone)

        backbone_out_features = list(backbone.children())[-1].in_features

        # --- BatchNorm ---
        self.bn = torch.nn.BatchNorm1d(backbone_out_features, affine=True)

        # --- LayerNorm ---
        self.ln = torch.nn.LayerNorm(backbone_out_features)

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
            layers.append(torch.nn.BatchNorm1d(hidden_dim, affine=True))
            layers.append(nn.LeakyReLU(0.01))
            in_features = hidden_dim
        if final_out is not None:
            layers.append(nn.Linear(in_features, final_out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.mean) / self.std # Normalize input
        x = self.backbone(x)           
        x = x.flatten(1)               
        x = self.bn(x)
        x = self.dropout(x)
        x = self.trunk(x)              

        preds = self.pred_head(x)      
        logvars = self.logvar_head(x)  

        return torch.cat([preds, logvars], dim=-1)


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
