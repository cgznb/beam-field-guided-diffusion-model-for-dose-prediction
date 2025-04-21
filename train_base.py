import os
from dataset import Dataset_PSDM_train, Dataset_PSDM_val
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
from Data_generalization.Nii_utils import NiiDataRead
import torch
import re
import pandas as pd
import argparse
import torch.utils.data as Data
import shutil
from guided_diffusion.unet import UNetModel_hu, UNetModel_MS_Former
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
import torch.distributed as dist
from guided_diffusion.resample import create_named_schedule_sampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="1", help='which gpu is used')
parser.add_argument('--bs', type=int, default=6, help='batch size')
parser.add_argument('--T', type=int, default=1000, help='T')
parser.add_argument('--epoch', type=int, default=2000, help='all_epochs')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda")

Time = "cmp_dosediffbase4"
train_bs = args.bs
val_bs = args.bs
lr_max = 0.0001
img_size = (128, 128)
all_epochs = args.epoch
data_root_train = '/home/cxx/dose_prediction/myself/Fourth/data/diff_data_128_first/NPY/train'
data_root_val = '/home/cxx/dose_prediction/myself/Fourth/data/diff_data_128_first/NPY/value'
original_dir = r'/home/cxx/dose_prediction/myself/Fourth/data/diff_data_128_first'

L2 = 0.0001

save_name = 'T{}_bs{}_lrmax{}_L2{}_epoch{}_Time{}'.format(args.T, train_bs,lr_max,L2, args.epoch,Time)
save_path = os.path.join('trained_models', save_name)
if os.path.exists(save_path):
    old_save_path = save_path + '_old'
    os.makedirs(old_save_path, exist_ok=True)
    for item in os.listdir(save_path):
        item_path = os.path.join(save_path, item)
        # 构建新路径
        new_item_path = os.path.join(old_save_path, item)
        # 移动文件或目录
        shutil.move(item_path, new_item_path)
os.makedirs(os.path.join('trained_models', save_name), exist_ok=True)

train_writer = SummaryWriter(os.path.join('trained_models', save_name, 'log/train'), flush_secs=2)

train_data = Dataset_PSDM_train(data_root=data_root_train)
val_data = Dataset_PSDM_val(data_root=data_root_val)
train_dataloader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)


print('train_lenth: %i   val_lenth: %i' % (train_data.len, val_data.len))

dis_channels = 18

model = UNetModel_MS_Former(image_size=img_size, in_channels=1, ct_channels=1, dis_channels=dis_channels,
                       model_channels=128, out_channels=1, num_res_blocks=2, attention_resolutions=(8, 16),
                       dropout=0,
                       channel_mult=(1, 1, 2, 4), conv_resample=True, dims=2, num_classes=None,
                       use_checkpoint=False,
                       use_fp16=False, num_heads=4, num_head_channels=-1, num_heads_upsample=-1,
                       use_scale_shift_norm=True,
                       resblock_updown=False, use_new_attention_order=False)

diffusion = SpacedDiffusion(use_timesteps=space_timesteps(args.T, [args.T]),
                            betas=gd.get_named_beta_schedule("linear", args.T),
                            model_mean_type=(gd.ModelMeanType.EPSILON),
                            model_var_type=(gd.ModelVarType.FIXED_LARGE),
                            loss_type=gd.LossType.MSE, rescale_timesteps=False)

diffusion_test = SpacedDiffusion(use_timesteps=space_timesteps(args.T, 'ddim8'),
                                betas=gd.get_named_beta_schedule("linear", args.T),
                                model_mean_type=(gd.ModelMeanType.EPSILON),
                                model_var_type=(gd.ModelVarType.FIXED_LARGE),
                                loss_type=gd.LossType.MSE, rescale_timesteps=False)

model = model.to(device)

schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=L2)
lr_scheduler = MultiStepLR(optimizer, milestones=[int((3 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)
best_MAE = 1000

def get_TargetPrescriptionDose(filename):
    df = pd.read_excel('data/' + 'dose128/' + 'xy2_statistic.xlsx')
    patient_id = filename
    Target = df.loc[df['patient_ID'] == int(patient_id), 'TargetPrescriptionDose'].to_numpy().tolist()[0]
    number_str = re.findall(r'\d+', Target)[0]
    Target = pd.to_numeric(number_str, errors='coerce')
    return Target


for epoch in range(all_epochs):
    lr = optimizer.param_groups[0]['lr']
    model.train()
    train_epoch_loss = []
    for i, (ct, dis, rtdose) in enumerate(train_dataloader):
        ct, dis, rtdose = ct.cuda().float(), dis.cuda().float(), rtdose.cuda().float()

        optimizer.zero_grad()
        t, weights = schedule_sampler.sample(rtdose.shape[0], rtdose.device)
        losses = diffusion.training_losses(model=model, x_start=rtdose, t=t, model_kwargs={'ct': ct, 'dis': dis}, noise=None)
        loss = (losses["loss"] * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_epoch_loss.append(loss.item())
        print('[%d/%d, %d/%d] train_loss: %.3f' %
              (epoch + 1, all_epochs, i + 1, len(train_dataloader), loss.item()))
    lr_scheduler.step()


    train_epoch_loss = np.mean(train_epoch_loss)
    train_writer.add_scalar('lr', lr, epoch + 1)
    train_writer.add_scalar('train_loss', train_epoch_loss, epoch + 1)

    if (epoch == 0) or (((epoch + 1) % 10) == 0):
        model.eval()
        val_epoch_MAE = []
        image_CT = []
        ture_rtdose = []
        pred_rtdose = []
        with torch.no_grad():
            for i, (names, ct, dis, rtdose) in enumerate(val_dataloader):
                names, ct, dis, rtdose = names, ct.cuda().float(), dis.cuda().float(), rtdose.cuda().float()
                pred = diffusion_test.ddim_sample_loop(
                    model=model, shape=(ct.size(0), 1, img_size[0], img_size[1]), noise=None, clip_denoised=True,
                    denoised_fn=None, cond_fn=None, model_kwargs={'ct': ct, 'dis': dis}, device=None, progress=False,
                    eta=0.0)
                for m in range(rtdose.size(0)):
                    name = names[m].split('_')[0]
                    img, _, _, _ = NiiDataRead(
                        os.path.join(original_dir, 'value_preprocess', name, 'dose.nii.gz'))
                    max = img.max()
                    min = img.min()
                    rtdose[m, :, :, :] = (rtdose[m, :, :, :] * (max - min) + min)
                    pred[m, :, :, :] = (pred[m, :, :, :] * (max - min) + min)
                body_mask = dis[:, 1:2]
                MAE = (torch.abs(rtdose - pred) * body_mask).sum() / body_mask.sum()

                val_epoch_MAE.append(MAE.item())
                if i in [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
                    image_CT.append(ct[0:1, :, :, :].cpu())
                    ture_rtdose.append(rtdose[0:1, :, :, :].cpu())
                    pred_rtdose.append(pred[0:1, :, :, :].cpu())

        val_epoch_MAE = np.mean(val_epoch_MAE)
        train_writer.add_scalar('val_MAE', val_epoch_MAE, epoch + 1)

        image_CT = torch.cat(image_CT, dim=0)
        image_CT = make_grid(image_CT, 2, normalize=True)
        train_writer.add_image('image_CT', image_CT, epoch + 1)
        ture_rtdose = torch.cat(ture_rtdose, dim=0)
        ture_rtdose = make_grid(ture_rtdose, 2, normalize=True)
        train_writer.add_image('ture_rtdose', ture_rtdose, epoch + 1)
        pred_rtdose = torch.cat(pred_rtdose, dim=0)
        pred_rtdose = make_grid(pred_rtdose, 2, normalize=True)
        train_writer.add_image('pred_rtdose', pred_rtdose, epoch + 1)

        torch.save(model.state_dict(),
                   os.path.join('trained_models', save_name, 'model_epoch' + str(epoch + 1) + '.pth'))
        if val_epoch_MAE < best_MAE:
            best_MAE = val_epoch_MAE
            torch.save(model.state_dict(),
                       os.path.join('trained_models', save_name, 'model_best_mae.pth'))


train_writer.close()
print('saved_model_name:', save_name)