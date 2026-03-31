import torch
import torch.nn as nn
import torchvision.models as models

# ------------------------------------------------------------------------------
# 1. ResNet50 Baseline
# ------------------------------------------------------------------------------
class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        # Load torchvision resnet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove final fc
        
        self.dx_head = nn.Linear(in_features, num_classes)
        self.progression_head = nn.Linear(in_features, 1)

    def forward(self, x):
        # x shape (B, T, C, H, W). Use only the last time step.
        x_last = x[:, -1, :, :, :]
        features = self.backbone(x_last) # (B, 2048)
        
        dx_logits = self.dx_head(features)
        prog_logits = self.progression_head(features)
        return dx_logits, prog_logits

# ------------------------------------------------------------------------------
# 2. ResNet50 + LSTM
# ------------------------------------------------------------------------------
class ResNet50LSTM(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=256, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dx_head = nn.Linear(hidden_dim, num_classes)
        self.progression_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        features = self.backbone(x_flat) # (B*T, 2048)
        features = features.view(B, T, -1) # (B, T, 2048)
        
        lstm_out, _ = self.lstm(features) # (B, T, hidden_dim)
        last_out = lstm_out[:, -1, :] # (B, hidden_dim)
        
        dx_logits = self.dx_head(last_out)
        prog_logits = self.progression_head(last_out)
        return dx_logits, prog_logits

# ------------------------------------------------------------------------------
# 3. EfficientNet-B3 + BiLSTM
# ------------------------------------------------------------------------------
class EfficientNetBiLSTM(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=128, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dx_head = nn.Linear(hidden_dim * 2, num_classes)
        self.progression_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        features = self.backbone(x_flat) # (B*T, 1536)
        features = features.view(B, T, -1)
        
        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]
        
        dx_logits = self.dx_head(last_out)
        prog_logits = self.progression_head(last_out)
        return dx_logits, prog_logits

# ------------------------------------------------------------------------------
# 4. ViT + Temporal Transformer
# ------------------------------------------------------------------------------
class ViTTemporal(nn.Module):
    def __init__(self, num_classes=5, num_layers=2, nhead=4, pretrained=True):
        super().__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dx_head = nn.Linear(in_features, num_classes)
        self.progression_head = nn.Linear(in_features, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        features = self.backbone(x_flat)
        features = features.view(B, T, -1)
        
        # Transformer expects (B, T, E)
        out = self.transformer_encoder(features)
        # using the last timestep representation
        last_out = out[:, -1, :]
        
        dx_logits = self.dx_head(last_out)
        prog_logits = self.progression_head(last_out)
        return dx_logits, prog_logits

# ------------------------------------------------------------------------------
# 5. TimeSformer (Simplified implementation for torchcam compatibility)
# ------------------------------------------------------------------------------
class TimeSformerSimplified(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        """
        Uses a ViT backbone. Instead of complex divided space-time attention from scratch,
        we patch the images together or use a 3D architecture.
        Since video transformers are complex, we proxy this with a video resnet 
        from torchvision or a flattened ViT approach.
        Here we use mvit_v1_b (Multiscale Vision Transformer for video) from torchvision.
        """
        super().__init__()
        self.backbone = models.video.mvit_v1_b(weights=models.video.MViT_V1_B_Weights.KINETICS400_V1 if pretrained else None)
        in_features = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        
        self.dx_head = nn.Linear(in_features, num_classes)
        self.progression_head = nn.Linear(in_features, 1)

    def forward(self, x):
        # x shape (B, T, C, H, W)
        # MViT expects (B, C, T, H, W)
        x_vid = x.permute(0, 2, 1, 3, 4)
        
        # Padding T dimension if it's too small for MViT which expects typically 16 frames.
        # We'll temporally interpolate to 16 frames via interpolate
        x_vid = nn.functional.interpolate(x_vid, size=(16, x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=False)
        
        features = self.backbone(x_vid) # (B, 768)
        
        dx_logits = self.dx_head(features)
        prog_logits = self.progression_head(features)
        return dx_logits, prog_logits

# ------------------------------------------------------------------------------
# 6. ConvLSTM
# ------------------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    """Basic ConvLSTM cell."""
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=4 * hidden_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMNetwork(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=64):
        """
        We use a lightweight CNN base to reduce spatial dims before ConvLSTM 
        to avoid taking massive memory.
        """
        super().__init__()
        # Initial dimension reduction
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ) # output approx (64, H/8, W/8)
        
        # ConvLSTM
        self.conv_lstm = ConvLSTMCell(input_dim=64, hidden_dim=hidden_dim, kernel_size=3)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dx_head = nn.Linear(hidden_dim, num_classes)
        self.progression_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # We need spatial feature maps for ConvLSTM
        h, c = None, None
        
        for t in range(T):
            xt = x[:, t, :, :, :]
            feat = self.spatial_encoder(xt) # (B, 64, H', W')
            
            if h is None:
                _, _, h_s, w_s = feat.shape
                h = torch.zeros(B, self.conv_lstm.hidden_dim, h_s, w_s, device=x.device)
                c = torch.zeros(B, self.conv_lstm.hidden_dim, h_s, w_s, device=x.device)
                
            h, c = self.conv_lstm(feat, (h, c))
            
        # h has shape (B, hidden_dim, H', W') -> the last hidden state
        pooled = self.pool(h).view(B, -1)
        
        dx_logits = self.dx_head(pooled)
        prog_logits = self.progression_head(pooled)
        return dx_logits, prog_logits

# ------------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------------
def get_model(model_name, num_classes=5):
    models_dict = {
        "resnet_baseline": ResNet50Baseline,
        "resnet50_lstm": ResNet50LSTM,
        "efficientnet_bilstm": EfficientNetBiLSTM,
        "vit_temporal": ViTTemporal,
        "timesformer": TimeSformerSimplified,
        "convlstm": ConvLSTMNetwork
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model_name: {model_name}. Supported: {list(models_dict.keys())}")
        
    return models_dict[model_name](num_classes=num_classes)
