## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
import clip
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

##########################################################################
## Generate Blur Matrix
class GenerateBM:
    # 加载模型和标记器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device)
    patch_size = 16
    prompts = ["not", "Slightly", "Noticeably", "Heavily"]
    weights = [0, 85, 170, 255]
    text_list = [f"the image is {p} blurred" for p in prompts]

    @staticmethod
    def preprocess(patches):
        # 输入张量形状为 patches = [8, patch_num, 3, 16, 16]
        # 获取输入小块张量的维度信息
        batch_size, num_blocks, channels, block_size, _ = patches.size()
        # 展平小块张量为二维张量，以便进行预处理
        flattened_blocks = patches.view(batch_size * num_blocks, channels, block_size, block_size)
        _preprocess = transforms.Compose([
            transforms.Resize((224, 224), antialias=None),  # 调整大小为 224x224 像素
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))  # 标准化
        ])
        # 对展平后的小块进行预处理
        preprocessed_blocks = torch.stack([_preprocess(block) for block in flattened_blocks])
        # 将预处理后的小块重新形状为原始形状
        preprocessed_blocks = preprocessed_blocks.view(batch_size, num_blocks, channels, 224, 224)
        return preprocessed_blocks

    @staticmethod
    def softmax(x):
        x = x.cpu().numpy()
        e_x = np.exp(x - np.max(x))  # 避免指数爆炸
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def split_image(image):
        # image = [8, 3, 128, 128]
        # 获取输入图像的维度信息
        batch_size, channels, height, width = image.size()
        # 初始化一个列表来存储分割后的小块
        patches = []
        # 遍历每张图像
        for i in range(batch_size):
            # 初始化一个列表来存储当前图像的小块
            image_patches = []
            # 遍历图像的高度和宽度
            for y in range(0, height, GenerateBM.patch_size):
                for x in range(0, width, GenerateBM.patch_size):
                    # 确定当前小块的边界框
                    box = (x, y, min(x + GenerateBM.patch_size, width), min(y + GenerateBM.patch_size, height))
                    # 切割当前小块
                    patch = image[i, :, box[1]:box[3], box[0]:box[2]]
                    # 将小块添加到列表中
                    image_patches.append(patch)
            # 将当前图像的小块列表添加到整体列表中
            patches.append(torch.stack(image_patches))
        # 将分割后的小块列表转换为张量，并返回
        return torch.stack(patches)  # [8, patch_num, 3, 16, 16]

    @staticmethod
    def combine_patches(patches, image_size):
        # image_size = [H, W]
        # patches = [b, patch_num, c, h, w]
        # 获取输入张量的维度信息
        batch_size, num_blocks, channels, _, _ = patches.size()
        height, width = image_size
        # 初始化一个张量来存储合并后的图像
        combined_image = torch.zeros(batch_size, channels, height, width, dtype=patches.dtype, device=patches.device)
        # 计算每个块在原始图像中的位置并将其添加到合并后的图像中
        block_index = 0
        for y in range(0, height, GenerateBM.patch_size):
            for x in range(0, width, GenerateBM.patch_size):
                combined_image[:, :, y:y + GenerateBM.patch_size, x:x + GenerateBM.patch_size] = patches[:, block_index]
                block_index += 1
        return combined_image

    @staticmethod
    def calculate_blur_value(image_patch):
        # image_patch = [patch_num, patch_size, patch_size]
        # patch预处理
        image_input = image_patch.to(GenerateBM.device)
        # 将文本编码
        text_input = clip.tokenize(GenerateBM.text_list).to(GenerateBM.device)
        # 对图片和文本进行编码
        with torch.no_grad():
            image_features = GenerateBM.model.encode_image(image_input)
            text_features = GenerateBM.model.encode_text(text_input)
        # 计算图片与文本之间的相似度
        similarities = image_features @ text_features.T
        blur_values = []
        for i in range(similarities.size(0)):
            blur_values.append((GenerateBM.softmax(similarities[i]) * GenerateBM.weights).sum(axis=0))
        for i in range(len(blur_values)):
            val = blur_values[i]
            blur_values[i] = torch.full((GenerateBM.patch_size, GenerateBM.patch_size), val).unsqueeze(0)
        blur_values = torch.stack(blur_values)
        return blur_values  # [patch_num, 1, patch_size, patch_size]

    @staticmethod
    def generate_blur_matrix(inp_img):
        # inp_img = [8, 3, 128, 128]
        image_size = inp_img.size()[-2:]
        splited_img = GenerateBM.split_image(inp_img)
        preprocessed_img = GenerateBM.preprocess(splited_img)
        blur_matrix = []
        for batch_index in range(preprocessed_img.size(0)):
            current_patch = preprocessed_img[batch_index]
            blur_matrix.append(GenerateBM.calculate_blur_value(current_patch))
        blur_matrix = torch.stack(blur_matrix)  # [batch_num, patch_num, 1, patch_size, patch_size]
        return GenerateBM.combine_patches(blur_matrix, image_size)



##########################################################################
## Layer Norm

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



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
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
    


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class enc_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(enc_FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.bm_in = nn.Conv2d(1, hidden_features, kernel_size=1, bias=bias)

        self.bm_dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)


    def forward(self, x, blur_matrix):
        # blur_matrix.shape = img_wide*img_height*1
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        bm = self.bm_dwconv(self.bm_in(blur_matrix))
        x = F.gelu(x1 * bm) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
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



##########################################################################
class enc_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(enc_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = enc_FeedForward(dim, ffn_expansion_factor, bias)

        self.norm_bm = LayerNorm(1, LayerNorm_type)

    def forward(self, x):
        x, blur_matrix = torch.split(x, split_size_or_sections=[x.shape[1]-1, 1], dim=1)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x), self.norm_bm(blur_matrix))
        x = torch.cat([x, blur_matrix], dim=1)
        return x



##########################################################################
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



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class single_channel_Downsample(nn.Module):
    def __init__(self, n_feat):
        super(single_channel_Downsample, self).__init__()

        self.body = nn.Sequential(nn.PixelUnshuffle(2),
                                  nn.Conv2d(n_feat*4, n_feat, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[enc_TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[enc_TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[enc_TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[enc_TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.generate_bm = GenerateBM.generate_blur_matrix

        self.bm_down = single_channel_Downsample(1)

    def forward(self, inp_img):
        blur_matrix = self.generate_bm(inp_img)
        
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_bm_level1 = blur_matrix.to(inp_enc_level1.device)
        inp_enc_con_level1 = torch.cat([inp_enc_level1, inp_enc_bm_level1], dim=1)  # dim1 = [3+1]
        out_enc_level1 = self.encoder_level1(inp_enc_con_level1)[:,:-1,:,:]

        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc_bm_level2 = self.bm_down(inp_enc_bm_level1)
        inp_enc_con_level2 = torch.cat([inp_enc_level2, inp_enc_bm_level2], dim=1)
        out_enc_level2 = self.encoder_level2(inp_enc_con_level2)[:,:-1,:,:]

        inp_enc_level3 = self.down2_3(out_enc_level2)
        inp_enc_bm_level3 = self.bm_down(inp_enc_bm_level2)
        inp_enc_con_level3 = torch.cat([inp_enc_level3, inp_enc_bm_level3], dim=1)
        out_enc_level3 = self.encoder_level3(inp_enc_con_level3)[:,:-1,:,:]

        inp_enc_level4 = self.down3_4(out_enc_level3)
        inp_enc_bm_level4 = self.bm_down(inp_enc_bm_level3)
        inp_enc_con_level4 = torch.cat([inp_enc_level4, inp_enc_bm_level4], dim=1)
        latent = self.latent(inp_enc_con_level4)[:,:-1,:,:]
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

