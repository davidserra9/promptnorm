import subprocess
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.aln_dataset import ALNDatasetGeom
from model import PromptNorm
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.val_utils import compute_psnr_ssim
from lpips import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class PromptNormModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #self.net = PromptIR(decoder=True)
        self.net = PromptNorm(decoder=True)
        self.l1_loss  = nn.L1Loss()
        self.lpips_loss = LPIPS(net="vgg").requires_grad_(False)
        self.ssim_loss = SSIM()
        self.lpips_lambda = 0.1
        self.ssim_lambda = 0.2
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, depth_patch, clean_patch) = batch
        restored = self.net(degrad_patch, depth_patch)

        # Compute losses
        l1_loss = self.l1_loss(restored,clean_patch)
        lpips_loss = self.lpips_loss(restored,clean_patch)
        ssim_loss = 1 - self.ssim_loss(restored,clean_patch)
        total_loss = l1_loss + self.lpips_lambda * lpips_loss + self.ssim_lambda * ssim_loss

        # Logging to TensorBoard (if installed) by default
        self.log("l1_loss", l1_loss, sync_dist=True)
        self.log("lpips_loss", lpips_loss, sync_dist=True)
        self.log("ssim_loss", ssim_loss, sync_dist=True)
        self.log("total_loss", total_loss, sync_dist=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        ([clean_name, de_id], degrad_patch, depth_patch, target_patch) = batch

        with torch.no_grad():
            restored = self.net(degrad_patch, depth_patch)

        psnr_i, ssim_i, _ = compute_psnr_ssim(restored, target_patch)

        self.log("val_psnr", psnr_i, sync_dist=True)
        self.log("val_ssim", ssim_i, sync_dist=True)


    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        return [optimizer],[scheduler]

def main():
    print("Options")
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)

    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)

    trainset = ALNDatasetGeom(input_folder=opt.train_input_dir,
                               depth_folder=opt.train_depth_dir,
                               target_folder=opt.train_target_dir,
                               resize_width_to=opt.resize_width,
                               patch_size=opt.patch_size)

    testset = ALNDatasetGeom(input_folder=opt.test_input_dir, depth_folder=opt.test_depth_dir, target_folder=opt.test_target_dir)
    
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=opt.num_workers)
    model = PromptNormModel()
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)


if __name__ == '__main__':
    main()



