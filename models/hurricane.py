from segment_anything import sam_model_registry
import torch
import torch.nn as nn


class HurricaneModel(nn.Module):
    def __init__(self, output_size=5, freeze_encoder=True, checkpoint_path="checkpoints/sam_vit_b_01ec64.pth"):
        super().__init__()

        # ViT Encoder
        sam = sam_model_registry["vit_b"](checkpoint_path)
        self.encoder = sam.image_encoder    # out: [B, 256, 64, 64]

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Decoder up-sample blocks (dim 1 is size 512 -> concat pre/post image feature maps from SAM)
        self.decoder_up = nn.Sequential(
            self._decoder_up_block(512, 256),    # 256x128x128
            self._decoder_up_block(256, 128),    # 128x256x256
            self._decoder_up_block(128, 64),     # 64x512x512
            self._decoder_up_block(64, 32),      # 32x1024x1024
        )

        # self.skip = nn.Conv2d(512, 16, kernel_size=1)   # skip connection
        self.fl = nn.Conv2d(32, output_size, kernel_size=1)    # 5x1024x1024

    def _decoder_up_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),   # upsample (by 2x)
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=1),  # add padding (in_dim = out_dim)
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pre_image, post_image):
        """
            pre_image (Tensor): [B, 3, 1024, 1024]
            post_image (Tensor): [B, 3, 1024, 1024]
            output (Tensor): [B, 5, 1024, 1024]
        """

        # Encoder
        pre_features = self.encoder(pre_image)    # [B, 256, 64, 64]
        post_features = self.encoder(post_image)  # [B, 256, 64, 64]

        bottleneck = torch.cat((pre_features, post_features), dim=1)    # [B, 512, 64, 64]

        # Decoder
        decoder_up = self.decoder_up(bottleneck)    # [B, 32, 1024, 1024]
        output = self.fl(decoder_up)    # [B, 5, 1024, 1024]

        return output
    