import subprocess
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.aln_dataset import ALNDatasetGeom
from model import PromptNorm
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import time
from PIL import Image
from lpips import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class PromptNormModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptNorm(decoder=True)
        self.lpips_loss = LPIPS(net="vgg").requires_grad_(False)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x, depth):
        return self.net(x, depth)

def main():
    print("Options")
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)

    # Careful: the input and target folders are the same just to avoid creating a new DataLoader class, but there is no target images.
    valset = ALNDatasetGeom(input_folder=opt.test_input_dir, geom_folder=opt.test_normals_dir, target_folder=opt.test_target_dir)
    
    valloader = DataLoader(valset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    model = PromptNormModel()
    model.load_state_dict(torch.load(opt.pretrained_ckpt_path, map_location='cuda:0')['state_dict'])
    model.eval()
    model.cuda()

    times = []

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    for [name, _], input_img, depth, gt_img in tqdm(valloader):
        with torch.no_grad():
            input_img = input_img.cuda()
            depth = depth.cuda()

            start = time.time()

            output = model(input_img, depth)

            times.append(time.time()-start)

            output = output.squeeze().permute(1,2,0).cpu().numpy()
            output = np.clip(output, 0, 1)

            # Save image with PIL
            output = Image.fromarray((output*255).astype(np.uint8)).convert('RGB')
            output.save(f'{opt.output_path}/{name[0]}.png')
    
    print(f"Average inference time: {np.mean(times)}")

if __name__ == '__main__':
    main()



