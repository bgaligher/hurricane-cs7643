"""
Additional References:
    1. Upsample decoder motivated by U-Net architecture: https://github.com/milesial/Pytorch-UNet

"""

from segment_anything import sam_model_registry
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Hurricane Model (Base Model):

'''
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
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),   # upsample H,W by 2x
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


'''
Hurricane Dropout Model (Base Model + decoder dropout layers)
'''
class HurricaneDOModel(nn.Module):
    def __init__(self, output_size=5, freeze_encoder=True, dropout_rate=0, checkpoint_path="checkpoints/sam_vit_b_01ec64.pth"):
        super().__init__()

        self.dropout_rate = dropout_rate

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
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),   # upsample H,W by 2x
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=1),  # add padding (in_dim = out_dim)
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate),
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


'''
Hurricane Model (Concat Skip Connections):

'''
class HurricaneCatModel(nn.Module):
    def __init__(self, output_size=5, freeze_encoder=True, dropout_rate=0, checkpoint_path="checkpoints/sam_vit_b_01ec64.pth"):
        super().__init__()

        self.dropout_rate = dropout_rate

        # ViT Encoder
        sam = sam_model_registry["vit_b"](checkpoint_path)
        self.encoder = sam.image_encoder    # out: [B, 256, 64, 64]

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # skip connection layers
        self.skip1 = nn.Conv2d(512, 256, kernel_size=1)
        self.skip2 = nn.Conv2d(512, 128, kernel_size=1)
        self.skip3 = nn.Conv2d(512, 64, kernel_size=1)

        # Decoder up-sample blocks (dim 1 is size 512 -> concat pre/post image feature maps from SAM)
        self.dec_up1 = self._decoder_up_block(512, 256)     # 256x128x128
        self.dec_up2 = self._decoder_up_block(512, 128)     # 128x256x256
        self.dec_up3 = self._decoder_up_block(256, 64)      # 64x512x512
        self.dec_up4 = self._decoder_up_block(128, 32)      # 32x1024x1024
        
        self.fl = nn.Conv2d(32, output_size, kernel_size=1)    # 5x1024x1024

    def _decoder_up_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),   # upsample H,W by 2x
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=1),  # add padding (in_dim = out_dim)
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate),
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

        # Decoder Upsample 1 + Skip 1
        x = self.dec_up1(bottleneck)        # [B, 256, 128, 128]
        skip1 = self.skip1(F.interpolate(bottleneck, scale_factor=2, mode='bilinear'))  # [B, 256, 128, 128]
        x = torch.cat((x, skip1), dim=1)    # [B, 512, 128, 128]

        # Decoder Upsample 2 + Skip 2
        x = self.dec_up2(x)        # [B, 128, 256, 256]
        skip2 = self.skip2(F.interpolate(bottleneck, scale_factor=4, mode='bilinear'))  # [B, 128, 256, 256]
        x = torch.cat((x, skip2), dim=1)    # [B, 256, 256, 256]

        # Decoder Upsample 3 + Skip 3
        x = self.dec_up3(x)        # [B, 64, 512, 512]
        skip3 = self.skip3(F.interpolate(bottleneck, scale_factor=8, mode='bilinear'))  # [B, 64, 512, 512]
        x = torch.cat((x, skip3), dim=1)    # [B, 128, 512, 512]
        

        x = self.dec_up4(x)     # [B, 32, 1024, 1024]

        output = self.fl(x)    # [B, 5, 1024, 1024]

        return output


'''
Hurricane Model (Addition Skip Connections)

'''
class HurricaneAddModel(nn.Module):
    def __init__(self, output_size=5, freeze_encoder=True, dropout_rate=0, checkpoint_path="checkpoints/sam_vit_b_01ec64.pth"):
        super().__init__()

        self.dropout_rate = dropout_rate

        # ViT Encoder
        sam = sam_model_registry["vit_b"](checkpoint_path)
        self.encoder = sam.image_encoder    # out: [B, 256, 64, 64]

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # skip connection layers
        self.skip1 = nn.Conv2d(512, 256, kernel_size=1)
        self.skip2 = nn.Conv2d(512, 128, kernel_size=1)
        self.skip3 = nn.Conv2d(512, 64, kernel_size=1)

        # Decoder up-sample blocks (dim 1 is size 512 -> concat pre/post image feature maps from SAM)
        self.dec_up1 = self._decoder_up_block(512, 256)     # 256x128x128
        self.dec_up2 = self._decoder_up_block(256, 128)     # 128x256x256
        self.dec_up3 = self._decoder_up_block(128, 64)      # 64x512x512
        self.dec_up4 = self._decoder_up_block(64, 32)       # 32x1024x1024
        
        self.fl = nn.Conv2d(32, output_size, kernel_size=1)    # 5x1024x1024

    def _decoder_up_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),   # upsample H,W by 2x
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=1),  # add padding (in_dim = out_dim)
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate),
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

        # Decoder Upsample 1 + Skip 1
        x = self.dec_up1(bottleneck)        # [B, 256, 128, 128]
        skip1 = self.skip1(F.interpolate(bottleneck, scale_factor=2, mode='bilinear'))  # [B, 256, 128, 128]
        x = x + skip1   # [B, 256, 128, 128]

        # Decoder Upsample 2 + Skip 2
        x = self.dec_up2(x)        # [B, 128, 256, 256]
        skip2 = self.skip2(F.interpolate(bottleneck, scale_factor=4, mode='bilinear'))  # [B, 128, 256, 256]
        x = x + skip2    # [B, 128, 256, 256]

        # Decoder Upsample 3 + Skip 3
        x = self.dec_up3(x)        # [B, 64, 512, 512]
        skip3 = self.skip3(F.interpolate(bottleneck, scale_factor=8, mode='bilinear'))  # [B, 64, 512, 512]
        x = x + skip3   # [B, 64, 512, 512]
        

        x = self.dec_up4(x)     # [B, 32, 1024, 1024]

        output = self.fl(x)    # [B, 5, 1024, 1024]

        return output


'''
Hurricane Model w/ LoRA

 Additional References:
    1. https://www.datacamp.com/tutorial/mastering-low-rank-adaptation-lora-enhancing-large-language-models-for-efficient-adaptation
    2. https://discuss.pytorch.org/t/iterating-all-layers-in-network/180341/2
    3. https://discuss.pytorch.org/t/how-to-replace-a-layer-with-own-custom-variant/43586/12
'''

class LoRALayer(nn.Module):
    '''
    Class obtained from this article: https://medium.com/@aseer-ansari/parameter-efficient-fine-tuning-lora-in-pytorch-3749f45c64af
    '''
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(nn.Module):
    '''
    Class obtained from this article: https://medium.com/@aseer-ansari/parameter-efficient-fine-tuning-lora-in-pytorch-3749f45c64af
    '''
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
    
class HurricaneLoRAModel(nn.Module):
    def __init__(self, output_size=5, rank=4, alpha=8, checkpoint_path="checkpoints/sam_vit_b_01ec64.pth"):
        super().__init__()

        self.rank = rank
        self.alpha = alpha

        # ViT Encoder
        sam = sam_model_registry["vit_b"](checkpoint_path)
        self.encoder = sam.image_encoder    # out: [B, 256, 64, 64]
        
        # freeze all parametrs in base encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # integrate LoRA into encoder
        self.integrate_LoRA()

        # Decoder up-sample blocks (dim 1 is size 512 -> concat pre/post image feature maps from SAM)
        self.decoder_up = nn.Sequential(
            self._decoder_up_block(512, 256),    # 256x128x128
            self._decoder_up_block(256, 128),    # 128x256x256
            self._decoder_up_block(128, 64),     # 64x512x512
            self._decoder_up_block(64, 32),      # 32x1024x1024
        )

        self.fl = nn.Conv2d(32, output_size, kernel_size=1)    # 5x1024x1024

    def integrate_LoRA(self):
        for name, module in self.encoder.named_modules():
            if 'attn.qkv' in name or 'attn.proj' in name:
                if isinstance(module, nn.Linear):
                    pieces = name.split('.')    # list of module pieces
                    current_piece = self.encoder
                    for piece in pieces[:-1]:
                        # iterate through pieces to obtain 'attn'
                        current_piece = getattr(current_piece, piece)
                    # set the new attribute to linear w/ LoRA layer
                    setattr(current_piece, pieces[-1],LinearWithLoRA(module, rank=self.rank, alpha=self.alpha))
        
        # enable backprop of LoRA params
        for name, param in self.encoder.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    def _decoder_up_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),   # upsample H,W by 2x
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
