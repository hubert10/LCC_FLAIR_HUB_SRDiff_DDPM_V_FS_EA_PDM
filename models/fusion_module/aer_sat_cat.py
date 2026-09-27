import torch
import torch.nn as nn
import torch.nn.functional as F

class FFCAT(nn.Module):
    """
    Fuses features from SITS and AER branches by concatenation at each resolution,
    projecting to match the channel dimension and spatial size of the AER features.
    """
    def __init__(self, sits_dims=[64, 128, 256], aer_dims=[128, 256, 512], use_norm=True):
        super(FFCAT, self).__init__()

        # 1x1 conv to project concatenated features back to AER channels
        self.fuse_layers = nn.ModuleList([
            nn.Conv2d(s + a, a, kernel_size=1, bias=not use_norm)
            for s, a in zip(sits_dims, aer_dims)
        ])

        # Optional normalization
        self.norm_layers = nn.ModuleList([
            nn.BatchNorm2d(a) if use_norm else nn.Identity()
            for a in aer_dims
        ])

        self.activation = nn.ReLU(inplace=True)

    def forward(self, aer_feats, sits_feats):
        assert len(sits_feats) == len(aer_feats), "Mismatched number of SITS and AER feature maps."

        fused_feats = []
        for i, (s_feat, a_feat) in enumerate(zip(sits_feats, aer_feats)):
            # Resize SITS feature to match AER spatial dimensions
            if s_feat.shape[2:] != a_feat.shape[2:]:
                s_feat = F.interpolate(s_feat, size=a_feat.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate along channel dimension
            fused = torch.cat([s_feat, a_feat], dim=1)
            # Project back to AER channels
            fused = self.fuse_layers[i](fused)
            fused = self.norm_layers[i](fused)
            fused = self.activation(fused)
            fused_feats.append(fused)
        return fused_feats
