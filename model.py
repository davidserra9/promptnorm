## PromptNorm: Image Geometry Guides Ambient Light Normalization
## David Serrano-Lozano et al., 2025.
## dserrano@cvc.uab.cat
## Computer Vision Center

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange

################ Layer Norm ################

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

################ Gated Feed-Forward Network (GFFN) ################

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

################ Transposed Self-Attention ################

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

################ Geometry-Guided Self-Attention ################

class DepthAwareCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DepthAwareCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        
        self.q_depth = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_depth_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, depth):
        b,c,h,w = x.shape
        _, _, h_d, w_d = depth.shape

        kv_rgb = self.kv_dwconv(self.kv(x))
        q_depth = self.q_depth_dwconv(self.q_depth(depth))
        k_rgb, v_rgb = kv_rgb.chunk(2, dim=1)
                
        q_depth = rearrange(q_depth, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k_rgb = rearrange(k_rgb, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_rgb = rearrange(v_rgb, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_depth = torch.nn.functional.normalize(q_depth, dim=-1)
        k_rgb = torch.nn.functional.normalize(k_rgb, dim=-1)
        attn = (q_depth @ k_rgb.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v_rgb)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out

################ Geometry-Aware Transformer Block ################
class GeometryAwareTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(GeometryAwareTransformerBlock, self).__init__()

        self.norm1_combined = LayerNorm(dim*2, LayerNorm_type)
        self.attn = DepthAwareCrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.norm1_combined(x)
        rgb, depth = torch.chunk(x, 2, dim=1)  # Split the input into RGB and depth
        rgb = rgb + self.attn(rgb, depth)
        rgb = rgb + self.ffn(self.norm2(rgb))
        return torch.cat([rgb, depth], dim=1)

################ Transformer Block ################

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    
################ Downsample and Upsample ################

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

################ Overlap Patch Embedding ################

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

################ Prompt Generation Block ################

class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

################ Depth Encoder ################

def depth_encoder(input_dim, output_dim, num_layers):
    layers = []
    for i in range(num_layers):
        in_channels = input_dim if i == 0 else output_dim
        layers.extend([
            nn.Conv2d(in_channels, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        ])
    return nn.Sequential(*layers)

################ PromptNorm Model ################

class PromptNorm(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 32,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = False,
    ):

        super(PromptNorm, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed_depth = OverlapPatchEmbed(3, dim)


        self.decoder = decoder
        
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = dim*2)
            self.prompt2 = PromptGenBlock(prompt_dim=128,prompt_len=5,prompt_size = 32,lin_dim = dim*4)
            self.prompt3 = PromptGenBlock(prompt_dim=320,prompt_len=5,prompt_size = 16,lin_dim = dim*8)

        self.encoder_level1 = nn.Sequential(*[GeometryAwareTransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level1_depth = depth_encoder(input_dim=dim, output_dim=dim, num_layers=3)
        self.down1_2_depth = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[GeometryAwareTransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level2_depth = depth_encoder(input_dim=int(dim*2**1), output_dim=int(dim*2**1), num_layers=3)
        self.down2_3_depth = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[GeometryAwareTransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        
        self.encoder_level3_depth = depth_encoder(input_dim=int(dim*2**2), output_dim=int(dim*2**2), num_layers=3)
        self.down3_4_depth = Downsample(int(dim*2**2)) ## From Level 2 to Level 3

        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+(dim*4), int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 448, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+448,int(dim*2**2),kernel_size=1,bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 192, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+192,int(dim*2**2),kernel_size=1,bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64,int(dim*2**1),kernel_size=1,bias=bias)


        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, depth_map, noise_emb = None):

        inp_enc_level1 = self.patch_embed(inp_img)
        depth_enc_level1 = self.patch_embed_depth(depth_map)

        out_enc_level1 = self.encoder_level1(torch.cat([inp_enc_level1, depth_enc_level1], dim=1))
        out_enc_level1, _ = torch.chunk(out_enc_level1, 2, dim=1)
        out_depth_level1 = self.encoder_level1_depth(depth_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_depth_level2 = self.down1_2_depth(out_depth_level1)
        
        out_enc_level2 = self.encoder_level2(torch.cat([inp_enc_level2, inp_depth_level2], dim=1))
        out_enc_level2, _ = torch.chunk(out_enc_level2, 2, dim=1)
        out_depth_level2 = self.encoder_level2_depth(inp_depth_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        inp_depth_level3 = self.down2_3_depth(out_depth_level2)

        out_enc_level3 = self.encoder_level3(torch.cat([inp_enc_level3, inp_depth_level3], dim=1))
        out_enc_level3, _ = torch.chunk(out_enc_level3, 2, dim=1)
        out_depth_level3 = self.encoder_level3_depth(inp_depth_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent_depth = self.down3_4_depth(out_depth_level3)

        latent = self.latent(inp_enc_level4)
        if self.decoder:
            dec3_param = self.prompt3(latent)

            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
                        
        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
           
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
