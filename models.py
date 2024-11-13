import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

#import models
#copy paste modules into here for colab run

class SelfAttention(nn.Module):
    def __init__(self, channels, size, dropout = 0.0, noise_steps = 1000):
        super(SelfAttention, self).__init__()
        self.channels = channels # number of channels from CNN
        self.size = size  # number of pixels a side
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, 4* channels),
            nn.GELU(),
            nn.Linear(4* channels, channels),
            nn.Dropout(dropout),
        )
        self.t_emb = nn.Embedding(noise_steps + 1, channels)
        self.p_emb = nn.Embedding(size * size, channels)

    def forward(self, x, t):
        time_emb = self.t_emb(t).unsqueeze(1) # B, 1, C
        time_emb = time_emb.repeat(1,self.size*self.size,1) # B, size*size, C
        pos = torch.arange(0, self.size * self.size, dtype=torch.long, device= x.device) #size*size
        pos_emb = self.p_emb(pos)# size*size, C
        emb = time_emb + pos_emb #B, C, size*size 

        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2) # B, size * size, C
        x = x + emb

        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        attention_value = attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
        return attention_value


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, dropout = 0.0):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
        self.dropout = nn.Dropout(dropout) #Use dropout regularization

    def forward(self, x):
        if self.residual:
            x = F.gelu(x + self.double_conv(x))
        else:
            x = F.gelu(self.double_conv(x))
        return self.dropout(x)


#Down pooling
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x 


#Up pooling
class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x 


class UNet(nn.Module):

    def __init__(self, c_in=3, c_out=3):
        super().__init__()

        # ci = num channels, di = pixels, both of layer i
        c1,d1 = 8, 64
        c2,d2 = 16, 32
        c3,d3 = 32, 16
        c4,d4 = 64, 8

        self.inc = DoubleConv(c_in, c1)     #64x64x3 -> 64x64x8
        self.down1 = Down(c1, c2)           #64x64x8 -> 32x32x16
        self.sa1 = SelfAttention(c2, d2)
        self.down2 = Down(c2, c3)           #32x32x16 -> 16x16x32
        self.sa2 = SelfAttention(c3, d3)
        self.down3 = Down(c3, c4)          #16x16x32 -> 8x8x64
        self.sa3 = SelfAttention(c4, d4)

        self.bot = nn.Sequential(
            DoubleConv(c4, c4),
            #SelfAttention(c4,d4),
            DoubleConv(c4, c4),
            #SelfAttention(c4,d4),
        ) #Any chance self attention here helps?

        self.up1 = Up(c4 + c3, c3)              #8x8x64 -> 16x16x64 + 16x16x32(skip) -> 16x16x32
        self.sa4 = SelfAttention(c3, d3)
        self.up2 = Up(c3 + c2, c2)               #16x16x32 -> 32x32x32 + 32x32x16(skip) -> 32x32x16
        self.sa5 = SelfAttention(c2, d2)
        self.up3 = Up(c2 + c1, c1)              #32x32x16 -> 64x64x16 + 64x64x8(skip) -> 64x64x8
        self.sa6 = SelfAttention(c1, d1)
        self.outc = nn.Conv2d(c1, c_out, kernel_size=1)  #64x64x16 -> 64x64x3


    def forward(self, x, t):

        x_1 = self.inc(x)
        x_2 = self.down1(x_1)
        x_2 = self.sa1(x_2,t)
        x_3 = self.down2(x_2)
        x_3 = self.sa2(x_3,t)
        x_4 = self.down3(x_3)
        x_4 = self.sa3(x_4,t)

        x = self.bot(x_4)

        x = self.up1(x,x_3)
        x = self.sa4(x,t)
        x = self.up2(x, x_2)
        x = self.sa5(x,t)
        x = self.up3(x, x_1)
        x = self.sa6(x,t)

        x = self.outc(x)

        return x


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, img_size=64, device="cpu"):
        self.noise_steps = noise_steps
        # We use the sinusoidal time schedule, so that alpha_hat = cos(half_pi * (t+s)/s)^2
        # The first gap will be approx ((pi/2)*s/(1 + s))^2, so
        # beta_start = ((pi/2)*s/(1 + s))^2 <-> s/(1 + s) = sqrt(beta_start) / half_pi
        self.beta_start = beta_start
        self.img_size = img_size
        self.device = device

        self.beta, self.alpha, self.alpha_hat = self.prepare_noise_schedule()

    def prepare_noise_schedule(self):
        half_pi = torch.acos(torch.zeros(1, device = self.device)).item() #approx pi/2
        s = torch.sqrt(torch.tensor(self.beta_start, device = self.device)) / half_pi

        alpha_hat = torch.cos(torch.linspace(s, half_pi - s, self.noise_steps, device = self.device)) ** 2

        alpha_shift = torch.roll(alpha_hat, 1, 0)
        alpha_shift[0] = 1
        alpha = alpha_hat / alpha_shift
        beta = 1 - alpha

        return beta, alpha, alpha_hat

    def noise_images(self, x, t):
        alpha_hat = self.alpha_hat

        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]

        eps = torch.randn_like(x).to(self.device)
        noised_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        #noised_image = noised_image.clamp(-1,1) #Might not be necessary? But also keeps images in distribution...
        return noised_image, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        #logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                  predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1,1) + 1)/2
        x = (x * 255).type(torch.uint8)
        return x