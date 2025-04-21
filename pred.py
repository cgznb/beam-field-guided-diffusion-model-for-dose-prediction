import os
import numpy as np
import torch
import ast
import argparse
import pandas as pd
import argparse
import shutil
from Nii_utils import NiiDataRead, NiiDataWrite
from guided_diffusion.unet import UNetModel_MS_Former,UNetModel_hu
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from evaluate_openKBP import get_Dose_score_and_DVH_score, get_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
parser.add_argument('--model_path', type=str, default=r'trained_models\T1000_bs5_lrmax0.0001_L20.0001_epoch1600_Timeaddcrtdiff4\model_best_mae.pth' ,help='trained model path')
parser.add_argument('--TTA', type=int, default=1, help='0/1')
parser.add_argument('--bs', type=int, default=32, help='batchsize')
parser.add_argument('--T', type=int, default=1000, help='T')
parser.add_argument('--ddim', type=str, default='8', help='ddim')
parser.add_argument('--Time', type=str, default='test' ,help='label')
args = parser.parse_args()
img_size = (128, 128)
dis_channels = 19
set = 'p' #p/n
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.Time = 'alltest'
args.ddim = 8

excel_Path = r'D:\medical_image\code\dose_prediction\myself\First\data\hutest\xy2_statistic.xlsx'
data_dir = r'D:\medical_image\code\dose_prediction\myself\First\data\hutest/{}_preprocess'.format(args.Time)
gt_dir = r'D:\medical_image\code\dose_prediction\myself\First\data\hutest/{}'.format(args.Time)
new_dir = 'Resultsdosediff4/ddim{}_TTA{}_Time{}_set{}'.format(args.ddim, args.TTA, args.Time,set)
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
print(new_dir)
os.makedirs(os.path.join(new_dir, 'predictions'), exist_ok=True)

if args.TTA:
    TTA_num = 4
else:
    TTA_num = 1

diffusion = SpacedDiffusion(use_timesteps=space_timesteps(args.T, 'ddim{}'.format(args.ddim)),
                            betas=gd.get_named_beta_schedule("linear", args.T),
                            model_mean_type=(gd.ModelMeanType.EPSILON),
                            model_var_type=(gd.ModelVarType.FIXED_LARGE),
                            loss_type=gd.LossType.MSE, rescale_timesteps=False)

net = UNetModel_MS_Former(image_size=img_size, in_channels=1, ct_channels=1, dis_channels=dis_channels,
                       model_channels=128, out_channels=1, num_res_blocks=2, attention_resolutions=(8,16),
                       dropout=0,
                       channel_mult=(1, 1, 2, 4), conv_resample=True, dims=2, num_classes=None,
                       use_checkpoint=False,
                       use_fp16=False, num_heads=4, num_head_channels=-1, num_heads_upsample=-1,
                       use_scale_shift_norm=True,
                       resblock_updown=False, use_new_attention_order=False)
net.cuda()
checkpoint = torch.load(args.model_path)
net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
net.eval()

with torch.no_grad():
    for i, ID in enumerate(os.listdir(os.path.join(data_dir))):
        print('{} {}'.format(i, ID))
        CT, spacing, origin, direction = NiiDataRead(os.path.join(data_dir, ID, 'CT.nii.gz'))
        dose, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'dose.nii.gz'))
        dose_crt, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'dose_crt.nii.gz'))
        Mask_beam, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_beam.nii.gz'))
        Mask_body, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_body.nii.gz'))
        Mask_gtv, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_gtv.nii.gz'))
        Mask_Kidney_L, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_Kidney_L.nii.gz'))
        Mask_Kidney_R, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_Kidney_R.nii.gz'))
        Mask_Liver, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_Liver.nii.gz'))
        Mask_SpinalCord, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_SpinalCord.nii.gz'))
        Mask_Stomach, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_Stomach.nii.gz'))
        Mask_ptv, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_ptv.nii.gz'))
        Mask_Heart, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_Heart.nii.gz'))

        PSDM_beam, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_beam.nii.gz'))
        PSDM_body, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_body.nii.gz'))
        PSDM_gtv, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_gtv.nii.gz'))
        PSDM_Kidney_L, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_Kidney_L.nii.gz'))
        PSDM_Kidney_R, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_Kidney_R.nii.gz'))
        PSDM_Liver, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_Liver.nii.gz'))
        PSDM_SpinalCord, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_SpinalCord.nii.gz'))
        PSDM_Stomach, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_Stomach.nii.gz'))
        PSDM_ptv, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_ptv.nii.gz'))
        PSDM_Heart, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_Heart.nii.gz'))

        PTVs_mask = Mask_ptv
        max = CT.max()
        min = CT.min()
        if max != min:
            CT = (CT - min) / (max - min)
        else:
            CT = 1
        dmax = dose_crt.max()
        dmin = dose_crt.min()
        if dmax != dmin:
            dose_crt = (dose_crt - dmin) / (dmax - dmin)
        else:
            dose_crt = 1

        original_shape = CT.shape
        pred_rtdose = np.zeros(original_shape)

        n_num = original_shape[0] // args.bs
        n_num = n_num + 0 if original_shape[0] % args.bs == 0 else n_num + 1
        for n in range(n_num):
            if n == n_num - 1:
                CT_one = CT[n * args.bs:, :, :]
                dis_one = np.concatenate((
                                            dose_crt[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PTVs_mask[n * args.bs:, :, :][np.newaxis, :, :, :],

                                          Mask_body[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_gtv[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_Kidney_L[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_Kidney_R[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_Liver[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_SpinalCord[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_Heart[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_Stomach[n * args.bs:, :, :][np.newaxis, :, :, :],


                                          PSDM_body[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_gtv[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_Heart[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_Kidney_L[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_Kidney_R[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_Liver[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_Stomach[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_ptv[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_SpinalCord[n * args.bs:, :, :][np.newaxis, :, :, :],

                                          ), axis=0)

            else:
                CT_one = CT[n * args.bs: (n + 1) * args.bs, :, :]
                dis_one = np.concatenate(
                                        (dose_crt[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          PTVs_mask[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],

                                          Mask_body[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          Mask_gtv[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          Mask_Kidney_L[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          Mask_Kidney_R[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          Mask_Liver[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          Mask_SpinalCord[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          Mask_Heart[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          Mask_Stomach[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],

                                          PSDM_body[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          PSDM_gtv[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          PSDM_Heart[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          PSDM_Kidney_L[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          PSDM_Kidney_R[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          PSDM_Liver[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          PSDM_Stomach[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          PSDM_ptv[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          PSDM_SpinalCord[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                                          ), axis=0)
            CT_one_tensor = torch.from_numpy(CT_one).unsqueeze(1).float()
            dis_one_tensor = torch.from_numpy(dis_one).float().permute(1, 0, 2, 3)
            for TTA_i in range(TTA_num):
                if TTA_i == 0:
                    CT_one_tensor_TTA = CT_one_tensor.cuda()
                    dis_one_tensor_TTA = dis_one_tensor.cuda()
                elif TTA_i == 1:
                    CT_one_tensor_TTA = torch.flip(CT_one_tensor, dims=[2]).cuda()
                    dis_one_tensor_TTA = torch.flip(dis_one_tensor, dims=[2]).cuda()
                elif TTA_i == 2:
                    CT_one_tensor_TTA = torch.flip(CT_one_tensor, dims=[3]).cuda()
                    dis_one_tensor_TTA = torch.flip(dis_one_tensor, dims=[3]).cuda()
                elif TTA_i == 3:
                    CT_one_tensor_TTA = torch.flip(CT_one_tensor, dims=[2, 3]).cuda()
                    dis_one_tensor_TTA = torch.flip(dis_one_tensor, dims=[2, 3]).cuda()
                noise = None
                pred_rtdose_one = diffusion.ddim_sample_loop(net, (
                CT_one_tensor_TTA.size(0), 1, img_size[0], img_size[1]),
                                                             model_kwargs={'ct': CT_one_tensor_TTA,
                                                                           'dis': dis_one_tensor_TTA},
                                                             noise=noise, clip_denoised=True, eta=0.0,
                                                             progress=True)
                if TTA_i == 1:
                    pred_rtdose_one = torch.flip(pred_rtdose_one, dims=[2])
                elif TTA_i == 2:
                    pred_rtdose_one = torch.flip(pred_rtdose_one, dims=[3])
                elif TTA_i == 3:
                    pred_rtdose_one = torch.flip(pred_rtdose_one, dims=[2, 3])
                if n == n_num - 1:
                    pred_rtdose[n * args.bs:, :, :] += pred_rtdose_one[:, 0, :, :].cpu().numpy()
                else:
                    pred_rtdose[n * args.bs: (n + 1) * args.bs, :, :] += pred_rtdose_one[:, 0, :, :].cpu().numpy()

        pred_rtdose = pred_rtdose / TTA_num
        df = pd.read_excel(excel_Path)
        pres_dose = df.loc[df['patient_name'] == int(ID), 'TargetPrescriptionDose'].to_numpy().tolist()[0]
        pres_dose = ast.literal_eval(pres_dose)
        pres_dose = float(pres_dose[0])
        pred_rtdose = (pred_rtdose * (pres_dose * 1.15))
        #
        # max = dose.max()
        # min = dose.min()
        # if max != min:
        #     pred_rtdose = (pred_rtdose * (max - min) + min)
        # else:
        #     pred_rtdose = max

        # pred_rtdose = pred_rtdose / TTA_num
        # pred_rtdose = (pred_rtdose + 1) * 31
        pred_rtdose = pred_rtdose * Mask_body
        pred_rtdose[pred_rtdose < 0] = 0
        os.makedirs(os.path.join(new_dir, 'predictions', ID))
        NiiDataWrite(os.path.join(new_dir, 'predictions', ID, 'dose.nii.gz'),
                     pred_rtdose, spacing, origin, direction)

Dose_score, Dose_std, DVH_score, DVH_std = get_Dose_score_and_DVH_score(prediction_dir=os.path.join(new_dir, 'predictions'),
                                                     gt_dir=gt_dir)
print('Dose_score: {}'.format(Dose_score))
print('DVH_score: {}'.format(DVH_score))
get_metrics(prediction_dir=os.path.join(new_dir, 'predictions'),
            gt_dir=gt_dir,
            save_dir=new_dir,
            excel_Path = r'D:\medical_image\code\dose_prediction\myself\First\data\hutest\xy2_statistic.xlsx')



with open(os.path.join(new_dir, 'score.txt'), 'w') as file:
    file.write('Dose_score: {} {}\nDVH_score: {} {}'.format(Dose_score, Dose_std, DVH_score, DVH_std))
