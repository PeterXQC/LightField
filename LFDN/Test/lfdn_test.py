from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import utils
import warnings
from LFDataset import LFDataset
from DeviceParameters import to_device
from MainNet import MainNet
from Functions import CropLF, MergeLF, ComptPSNR
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import scipy.io as scio 
import scipy.misc as scim
import os,time
import logging,argparse
from datetime import datetime

warnings.filterwarnings("ignore")
plt.ion()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler('Testing_9sas_%s.log' % datetime.now().strftime("%Y%m%d%H%M"))
log.addHandler(fh)

# Testing settings
parser = argparse.ArgumentParser(description="Light Field Denoising")
parser.add_argument("--sigma", type=int, default=10, help="The number of stages")
parser.add_argument("--stageNum", type=int, default=6, help="The number of stages")
parser.add_argument("--angResolution", type=int, default=5., help="The angular resolution of original LF")
parser.add_argument("--batchSize", type=int, default=1, help="Batch size")
parser.add_argument("--cropPatchSize", type=int, default=16, help="The size of cropped LF patch")
parser.add_argument("--overlap", type=int, default=0, help="The size of cropped LF patch")
parser.add_argument("--channelNum", type=int, default=1, help="The number of input channels")
parser.add_argument("--modelPath", type=str, default='D:\XuQichen\LightField\LFDN\Test\model\lfdn_SASOnly_Res10.pth', help="Path for loading trained model ")
parser.add_argument("--dataPath", type=str, default='D:\XuQichen\LightField\LFDN\LFData\\train_data.mat', help="Path for loading testing data ")
parser.add_argument("--savePath", type=str, default='D:\XuQichen\LightField\LFDN\Denoised\\', help="Path for saving results ")

opt = parser.parse_args()
logging.info(opt)

if __name__ == '__main__':

    lf_dataset = LFDataset(opt)
    dataloader = DataLoader(lf_dataset, batch_size=opt.batchSize, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MainNet(opt)
    model.load_state_dict(torch.load(opt.modelPath))
    model.eval()
    to_device(model, device)

    with torch.no_grad():
        num = 0
        avg_psnr_y = 0
        avg_ssim_y = 0
        for _, sample in enumerate(dataloader):
            num += 1
            LF = sample['LF']  # test lf
            noiLF = sample['noiLF']  # test lf
            
            cropStride = opt.cropPatchSize - opt.overlap
            noiLFStack, coordinate = CropLF(noiLF, opt.cropPatchSize, cropStride)  # [b,n,u,v,x,y,c]
            b, n, u, v, x, y, c = noiLFStack.shape
            
            print(noiLFStack.shape)
            print(coordinate)
            
            denoilfStack = torch.zeros(b, n, u, v, x, y, c)  # [b,n,u,v,x,y,c]

            start = time.time()   
            # reconstruction
            for i in range(noiLFStack.shape[1]):
                denoiLFPatch = model(noiLFStack[:, i, :, :, :, :].permute(0, 1, 2, 5, 3, 4).cuda())  # [b,u,v,c,x,y]
                denoilfStack[:, i, :, :, :, :, :] = denoiLFPatch.permute(0, 1, 2, 4, 5, 3)  # [b,n,u,v,x,y,c]
            end = time.time() - start
            print(end)
            
            denoiLF = MergeLF(denoilfStack, coordinate, opt.overlap)  # [b,u,v,x,y,c]
            b, u, v, x, y, c = denoiLF.shape
            LF = LF[:, :, :, opt.overlap // 2:opt.overlap // 2 + x, opt.overlap // 2:opt.overlap // 2 + y, :]

            lf_psnr_y = 0
            lf_ssim_y = 0
   
            for ind_uv in range(u * v):
                lf_psnr_y += ComptPSNR(
                    np.squeeze(denoiLF.reshape(b, u * v, x, y, c)[0, ind_uv].cpu().numpy()),
                    np.squeeze(LF.reshape(b, u * v, x, y, c)[0, ind_uv].cpu().numpy())) / (u * v)
                                       
                lf_ssim_y += compare_ssim(
                    np.squeeze((denoiLF.reshape(b, u * v, x, y, c)[0, ind_uv].cpu().numpy() * 255.0).astype(np.uint8)),
                    np.squeeze((LF.reshape(b, u * v, x, y, c)[0, ind_uv].cpu().numpy() * 255.0).astype(np.uint8)),
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False) / (u * v)

            avg_psnr_y += lf_psnr_y / len(dataloader)           
            avg_ssim_y += lf_ssim_y / len(dataloader) 
            
            
            #save reconstructed LF
            scio.savemat(os.path.join(opt.savePath, str(num) + '.mat'),
                         {'lf_recons': torch.squeeze(denoiLF).numpy()})  # [u,v,x,y,c]
            
                         
            log.info('Index: %d  PSNR: %.2f  SSIM: %.3f' % (num, lf_psnr_y, lf_ssim_y))
        log.info('Average PSNR: %.2f  SSIM: %.3f' % (avg_psnr_y, avg_ssim_y))
