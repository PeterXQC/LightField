import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math

# new import for HAT
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from einops import rearrange

class get_model(nn.Module):
	def __init__(self, args):
		super(get_model, self).__init__()
		n_blocks = 15
		channels = 64
		self.angRes = args.angRes_in
		self.window_size = 7
		self.upscale_factor = args.scale_factor
		self.featureFusion = FeatureFusionNet2()
		self.rpi = self.calculate_rpi_sa()

		self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False)

		self.HFEM_1 = HFEM(self.angRes, n_blocks, channels, self.window_size, first=False)
		self.HFEM_2 = HFEM(self.angRes, n_blocks, channels, self.window_size, first=False)
		self.HFEM_3 = HFEM(self.angRes, n_blocks, channels, self.window_size, first=False)
		self.HFEM_4 = HFEM(self.angRes, n_blocks, channels, self.window_size, first=False)
		self.HFEM_5 = HFEM(self.angRes, n_blocks, channels, self.window_size, first=False)

		# define tail module for upsamling
		UpSample = [
			Upsampler(self.upscale_factor, channels, kernel_size=3, stride=1, dilation=1, padding=1, act=False),
			nn.Conv2d(channels, 1, kernel_size=1, stride=1, dilation=1, padding=0, bias=False)]
		self.UpSample = nn.Sequential(*UpSample)
		

	def forward(self, x, info=None):

		# Upscaling
		x_upscale = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

		# Reshaping
		x_reshaped = SAI2MacPI(x, self.angRes)

		# extract super-shallow features
		x = self.init_conv(x_reshaped)

		HFEM_1 = self.HFEM_1(x, self.rpi)
		HFEM_2 = self.HFEM_2(HFEM_1, self.rpi)
		HFEM_3 = self.HFEM_3(HFEM_2, self.rpi)
		HFEM_4 = self.HFEM_4(HFEM_3, self.rpi)
		HFEM_5 = self.HFEM_5(HFEM_4, self.rpi)

		# Reshaping
		x_out = MacPI2SAI(HFEM_5 + x, self.angRes)
		x_out = self.UpSample(x_out)
		x_out += x_upscale
		# 将5x5视角重排，然后用3D卷积
		x_out_split = LFsplit(x_out, 5)
		out_new = self.featureFusion(x_out_split)  # sen_add: 增加3D卷积，实现多角度特征融合
		return out_new, HFEM_1, HFEM_2, HFEM_3, HFEM_4, HFEM_5
	
	# calculate rpi for Window attention
	def calculate_rpi_sa(self):
		# calculate relative position index for SA
		coords_h = torch.arange(self.window_size)
		coords_w = torch.arange(self.window_size)
		coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
		coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
		relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
		relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
		relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
		relative_coords[:, :, 1] += self.window_size - 1
		relative_coords[:, :, 0] *= 2 * self.window_size - 1
		relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
		return relative_position_index


# import matplotlib.pyplot as plt
# tmp = x_out_split[0,0,:,:].cpu().detach().numpy()
# plt.imshow(tmp)
# plt.show()

class HFEM(nn.Module):
	def __init__(self, angRes, n_blocks, channels, window_size, first=False):
		super(HFEM, self).__init__()
		self.first = first 
		self.n_blocks = n_blocks
		self.angRes = angRes

		# define head module epi feature
		head_epi = []
		if first:  
			head_epi.append(nn.Conv2d(angRes, channels, kernel_size=3, stride=1, padding=1, bias=False))
		else:
			head_epi.append(nn.Conv2d(angRes*channels, channels, kernel_size=3, stride=1, padding=1, bias=False))

		self.head_epi = nn.Sequential(*head_epi)

		self.epi2spa = nn.Sequential(
			nn.Conv2d(4*channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(angRes),
		)

		# define head module intra spatial feature
		head_spa_intra = []
		if first:  
			head_spa_intra.append(nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=int(angRes),
											padding=int(angRes), bias=False))
		else:
			head_spa_intra.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
											padding=int(angRes), bias=False))

		self.head_spa_intra = nn.Sequential(*head_spa_intra)

		# define head module inter spatial feature
		head_spa_inter = []
		if first:  
			head_spa_inter.append(nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False))
		else:
			head_spa_inter.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False))

		self.head_spa_inter = nn.Sequential(*head_spa_inter)

		# define head module intra angular feature
		head_ang_intra = []
		if first: 
			head_ang_intra.append(nn.Conv2d(1, channels, kernel_size=int(angRes), stride=int(angRes), dilation=1,
											padding=0, bias=False))

		else:
			head_ang_intra.append(nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), dilation=1,
											padding=0, bias=False))

		self.head_ang_intra = nn.Sequential(*head_ang_intra)

		self.ang2spa_intra = nn.Sequential(
			nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(angRes), 
		)

		# define head module inter angular feature
		head_ang_inter = []
		if first:  
			head_ang_inter.append(nn.Conv2d(1, channels, kernel_size=int(angRes*2), stride=int(angRes*2), dilation=1,
											padding=0, bias=False))

		else:
			head_ang_inter.append(nn.Conv2d(channels, channels, kernel_size=int(angRes*2), stride=int(angRes*2),
											dilation=1, padding=0, bias=False))

		self.head_ang_inter = nn.Sequential(*head_ang_inter)

		self.ang2spa_inter = nn.Sequential(
			nn.Conv2d(channels, int(4*angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(2*angRes),
		)

		# define  module attention fusion feature
		self.attention_fusion = AttentionFusion(channels)
											
		# define  module spatial residual group
		self.conv_RG = nn.Conv2d(5*channels, channels, kernel_size=1, stride =1, dilation=1, padding=0, bias=False)
		self.RG = ResidualGroup(self.n_blocks, channels, window_size, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), )

	def forward(self, x, rpi):
		# MO-EPI feature extractor
		data_0, data_90, data_45, data_135 = MacPI2EPI(x, self.angRes)

		data_0 = self.head_epi(data_0)
		data_90 = self.head_epi(data_90)
		data_45 = self.head_epi(data_45)
		data_135 = self.head_epi(data_135)
	
		mid_merged = torch.cat((data_0, data_90, data_45, data_135), 1)
		x_epi = self.epi2spa(mid_merged)

		# intra/inter spatial feature extractor
		x_s_intra = self.head_spa_intra(x)
		x_s_inter = self.head_spa_inter(x)
	
		# intra/inter angular feature extractor
		x_a_intra = self.head_ang_intra(x)
		x_a_intra = self.ang2spa_intra(x_a_intra)

		x_a_inter = self.head_ang_inter(x)
		x_a_inter = self.ang2spa_inter(x_a_inter)

		# fusion feature and refinement
		out = x_s_intra.unsqueeze(1)
		out = torch.cat([x_s_inter.unsqueeze(1), out], 1)
		out = torch.cat([x_a_intra.unsqueeze(1), out], 1)
		out = torch.cat([x_a_inter.unsqueeze(1), out], 1)
		out = torch.cat([x_epi.unsqueeze(1), out], 1)

		[out, att_weight] = self.attention_fusion(out)
		out = self.conv_RG(out)
		out = self.RG(out, rpi)

		# allow skip entire HFEM.
		return out + x


class AttentionFusion(nn.Module):
	def __init__(self, channels, eps=1e-5):
		super(AttentionFusion, self).__init__()
		self.epsilon = eps
		self.alpha = nn.Parameter(torch.ones(1))
		self.gamma = nn.Parameter(torch.zeros(1))
		self.beta = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		m_batchsize, N, C, height, width = x.size()
		x_reshape = x.view(m_batchsize, N, -1)
		M = C * height * width

		# compute covariance feature
		mean = torch.mean(x_reshape, dim=-1).unsqueeze(-1)
		x_reshape = x_reshape - mean
		cov = (1 / (M - 1) * x_reshape @ x_reshape.transpose(-1, -2)) * self.alpha
		# print(cov)
		norm = cov / ((cov.pow(2).mean((1, 2), keepdim=True) + self.epsilon).pow(0.5))  # l-2 norm

		attention = torch.tanh(self.gamma * norm + self.beta)
		x_reshape = x.view(m_batchsize, N, -1)

		out = torch.bmm(attention, x_reshape)
		out = out.view(m_batchsize, N, C, height, width)

		out += x
		out = out.view(m_batchsize, -1, height, width)
		return out, attention


## Residual Channel Attention Block (RCAB)
class ResidualBlock(nn.Module):
	def __init__(self, n_feat, kernel_size, stride, dilation, padding, window_size):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True)
		self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True)
		self.relu = nn.ReLU(inplace=True)
		# # initialization
		# initialize_weights([self.conv1, self.conv2], 0.1)
		self.CALayer = CALayer(n_feat, reduction=int(n_feat//4))

		# add in window attention
		self.WALayer = WindowAttention(n_feat, to_2tuple(window_size), 6)

	def forward(self, x, rpi):
		out = self.relu(self.conv1(x))
		out = self.conv2(out)
		CAout = self.CALayer(out)
		print(out.shape)
		WAout = self.WALayer(out, rpi)
		return x + WAout + CAout


## Residual Group
class ResidualGroup(nn.Module):
	def __init__(self, n_blocks, n_feat, window_size, kernel_size, stride, dilation, padding):
		super(ResidualGroup, self).__init__()
		self.fea_resblock = RGSequential(n_feat, n_blocks, kernel_size, stride, dilation, padding, window_size)
		self.conv = nn.Conv2d(n_feat, n_feat,  kernel_size=kernel_size, stride=stride, dilation=dilation,
							  padding=padding, bias=True)

	def forward(self, x, rpi):
		res = self.fea_resblock(x, rpi)
		res = self.conv(res)
		res += x
		return res

def make_layer(block, nf, n_layers,kernel_size, stride, dilation, padding ):
	layers = []
	for _ in range(n_layers):
		layers.append(block(nf, kernel_size, stride, dilation, padding))
	return nn.Sequential(*layers)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(CALayer, self).__init__()
		# global average pooling: feature --> point
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		# feature channel downscale and upscale --> channel weight
		self.conv_du = nn.Sequential(
			nn.Conv2d(channel, channel // reduction, 1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // reduction, channel, 1, padding=0),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.conv_du(y)
		return x * y


class Upsampler(nn.Sequential):
	def __init__(self, scale, n_feat,kernel_size, stride, dilation, padding,  bn=False, act=False, bias=True):

		m = []
		if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
			for _ in range(int(math.log(scale, 2))):
				m.append(nn.Conv2d(n_feat, 4 * n_feat, kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding, bias=True))
				m.append(nn.PixelShuffle(2))
				if bn: m.append(nn.BatchNorm2d(n_feat))
				if act: m.append(act())
		elif scale == 3:
			m.append(nn.Conv2d(n_feat, 9 * n_feat, kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding, bias=True))
			m.append(nn.PixelShuffle(3))
			if bn: m.append(nn.BatchNorm2d(n_feat))
			if act: m.append(act())
		else:
			raise NotImplementedError

		super(Upsampler, self).__init__(*m)


def SAI2MacPI(x, angRes):
	b, c, hu, wv = x.shape
	h, w = hu // angRes, wv // angRes
	tempU = []
	for i in range(h):
		tempV = []
		for j in range(w):
			tempV.append(x[:, :, i::h, j::w])
		tempU.append(torch.cat(tempV, dim=3))
	out = torch.cat(tempU, dim=2)
	return out


def SAI24DLF(x, angRes):
	uh, vw = x.shape
	h0, w0 = int(uh // angRes), int(vw // angRes)

	LFout = torch.zeros(angRes, angRes, h0, w0)
	for u in range(angRes):
		start_u = u * h0
		end_u = (u + 1) * h0
		for v in range(angRes):
			start_v = v * w0
			end_v = (v + 1) * w0
			img_tmp = x[start_u:end_u, start_v:end_v]
			LFout[u, v, :, :] = img_tmp

	return LFout


def MacPI2SAI(x, angRes):
	out = []
	for i in range(angRes):
		out_h = []
		for j in range(angRes):
			out_h.append(x[:, :, i::angRes, j::angRes])
		out.append(torch.cat(out_h, 3))
	out = torch.cat(out, 2)
	return out


def MacPI2EPI(x, angRes):
	data_0 = []
	data_90 = []
	data_45 = []
	data_135 = []

	index_center = int(angRes // 2)
	for i in range(0, angRes, 1):
		img_tmp = x[:, :, index_center::angRes, i::angRes]
		data_0.append(img_tmp)
	data_0 = torch.cat(data_0, 1)

	for i in range(0, angRes, 1):
		img_tmp = x[:, :, i::angRes, index_center::angRes]
		data_90.append(img_tmp)
	data_90 = torch.cat(data_90, 1)

	for i in range(0, angRes, 1):
		img_tmp = x[:, :, i::angRes, i::angRes]
		data_45.append(img_tmp)
	data_45 = torch.cat(data_45, 1)

	for i in range(0, angRes, 1):
		img_tmp = x[:, :, i::angRes, angRes - i - 1::angRes]
		data_135.append(img_tmp)
	data_135 = torch.cat(data_135, 1)

	return data_0, data_90, data_45, data_135


def weights_init(m):
    pass

def ssim(img1, img2, window_size=11, size_average=True):
    C1 = 0.01**2
    C2 = 0.03**2

    window = torch.ones(window_size, window_size).float().cuda()
    window = window.unsqueeze(0).unsqueeze(0)  # Add two singleton dimensions: [1, 1, window_size, window_size]
    window /= window.sum()

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Convert grayscale images to "RGB"
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.size(1) == 1:
            y = y.repeat(1, 3, 1, 1)
        
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = nn.functional.l1_loss(x_vgg, y_vgg)
        return loss
    
# Total loss
class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        
        self.criterion_MSE = nn.MSELoss()
        self.criterion_Perceptual = PerceptualLoss()
        
        self.alpha = 0.5  # Weight for SSIM
        self.beta = 0.5   # Weight for Perceptual loss
        self.gamma = 1.0  # Weight for MSE

    def forward(self, SR, HR, criterion_data=[]):
        loss_mse = self.criterion_MSE(SR, HR)
        loss_ssim = 1 - ssim(SR, HR)
        loss_perceptual = self.criterion_Perceptual(SR, HR)
        
        total_loss = self.gamma * loss_mse + self.alpha * loss_ssim + self.beta * loss_perceptual
        return total_loss

def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_out = []
    for u in range(angRes):
        for v in range(angRes):
            data_out.append(data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w])
    data_out = torch.cat(data_out, dim=1)
    return data_out


class FeatureFusionNet(nn.Module):
    def __init__(self):
        super(FeatureFusionNet, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=25, out_channels=1, kernel_size=(25, 1, 1), stride=1, padding=(12, 0, 0)) # 25个角度图融合
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0) # 2D卷积

    def forward(self, x):
        # Assuming input shape is (B, 25, C, H, W),
        # where B is batch size, C is number of channels, H and W are height and width
        x = self.conv3d(x)
        # Now the shape is (B, 1, C, H, W), we need to squeeze the second dimension
        x = x.squeeze(1)
        # Now the shape is (B, C, H, W), we can apply 2D convolution
        x = self.conv2d(x)
        return x




# class FeatureFusionNet2(nn.Module):
#     def __init__(self):
#         super(FeatureFusionNet2, self).__init__()
#         self.conv1 = nn.Conv2d(25, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
#         self.fc = nn.Linear(256*256, 256*256)
#         self.pool = nn.AdaptiveAvgPool2d((256, 256))
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = self.conv2(x)
#         x = nn.functional.relu(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = x.view(x.size(0), 1, 256, 256)
#         x = self.pool(x)
#         return x



class FeatureFusionNet2(nn.Module):
    def __init__(self):
        super(FeatureFusionNet2, self).__init__()
        self.conv1 = nn.Conv2d(25, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((256, 256))

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        return x

# new element that needs work
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
# in Residule group, we ran res block n_bock many times. 
# this does same thing without nn.sequential to allow more parameters in forward.
class RGSequential(nn.Module):
    def __init__(self, n_feat, n_blocks, kernel_size, stride, dilation, padding, window_size):
        super(RGSequential, self).__init__()
        
        # Define the layer you want to repeat
        block = ResidualBlock(n_feat, kernel_size, stride, dilation, padding, window_size)
        
        # Use a ModuleList to store repeated layers
        self.layers = nn.ModuleList([block for _ in range(n_blocks)])
        
    def forward(self, x, rpi):
        for layer in self.layers:
            x = layer(x, rpi)
        return x
    
class OCAB(nn.Module):
    # overlapping cross-attention block

    def __init__(self, dim,
                input_resolution,
                window_size,
                overlap_ratio,
                num_heads,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm
                ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3,  bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim,dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2) # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1) # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1) # b, 2*c, h, w

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv) # b, c*w*w, nw
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1] # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut

        x = x + self.mlp(self.norm2(x))
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    

    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

