import numpy as np
from multiprocessing import Pool
import pandas as pd
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p,subfiles
import cv2
import os
def get_3D_Dose_dif(pred, gt, possible_dose_mask=None):
    if possible_dose_mask is not None:
        pred = pred[possible_dose_mask > 0]
        gt = gt[possible_dose_mask > 0]

    dif = np.mean(np.abs(pred - gt))
    return dif

def get_hi(_dose,_mask):
    output={}
    _roi_dose=_dose[_mask>0]
    output['d2'] = np.percentile(_roi_dose,98)
    output['d98'] = np.percentile(_roi_dose, 2)
    output['d50'] = np.percentile(_roi_dose, 50)
    output['HI'] = (output['d2']-output['d98'])/output['d50']
    output['mean'] = np.mean(_roi_dose)
    output['max'] = np.max(_roi_dose)
    return output

def get_dice(_dose_pred,_dose_gt,_mask,pres_dose=None):
    # _dose_pred[_mask==0]=0
    # _dose_gt[_mask==0]=0

    output = {}
    inter = np.linspace(pres_dose * 0.0, pres_dose * 1., 100)
    dices=[]
    for i in range(len(inter)):
        temp =  inter[i]
        dose1=np.array((_dose_pred>temp),dtype='uint8')
        dose2=np.array((_dose_gt>temp),dtype='uint8')
        dice =2*(dose1*dose2).sum()/(dose1.sum()+dose2.sum())
        dices.append(dice)
    output['x_p']=[i/pres_dose*100 for i in inter]
    output['y_dice']=dices
    return output

def get_oar_metric(_dose,_mask):
    output={}
    _roi_dose = _dose[_mask > 0]
    _roi_size = len(_roi_dose)
    output['d2'] = np.percentile(_roi_dose, 98)
    output['V20'] = len(_roi_dose[_roi_dose>20])/_roi_size
    output['V30'] =  len(_roi_dose[_roi_dose>30])/_roi_size
    return output

def get_d95(_dose,_mask):
    _roi_dose = _dose[_mask > 0]
    _roi_size = len(_roi_dose)
    out = np.percentile(_roi_dose, 5) # d95
    return out

def get_ci(_dose,_ptv_mask,_pres_dose):
    _ptv_mask[_ptv_mask>0]=1
    iso_dose=(_dose>=_pres_dose)
    iso_dose = np.array(iso_dose,dtype='uint8')
    P=iso_dose.sum()
    T=_ptv_mask.sum()
    T_P=(_ptv_mask*iso_dose).sum()
    ci=(T_P)**2/(P*T)
    return ci

def cal_one_patient(args):
    pred_file=args
    gt_file = pred_file.replace('hupred','hutest')
    dataset_path = os.path.join('data', 'dose_128', 'reprocess_data/')
    patient_id=pred_file.split('\\')[-1].split('.')[0]
    # if patient_id=='596865':
    #     print(patient_id)

    # 获取患者的处方剂量
    prescription_dose = {'300780': 54., '700810': 48., '622076': 54., '710585': 40.}
    pres_dose=prescription_dose[patient_id]

    # 获取预测剂量数据
    pred=sitk.ReadImage(pred_file)
    pred=sitk.GetArrayFromImage(pred)
    # 获取临床剂量数据
    gt=sitk.ReadImage(gt_file)
    gt=sitk.GetArrayFromImage(gt)
    # print(patient_id,gt.max())
    #
    # 获取PTV mask
    ptv_mask = sitk.ReadImage(dataset_path+'ptv/'+patient_id+'.nii.gz')
    ptv_mask = sitk.GetArrayFromImage(ptv_mask)
    #获取危及器官mask
    dataset_path1='data/dose128/reprocess_data/' # 注意仅路径用于获取多个危及器官标签
    # "1": "liver", "2": "right kidney", "3": "left kidney", "4": "stomach", "5": "spinal cord",
    oar_mask = sitk.ReadImage(dataset_path1+'oar/'+patient_id+'.nii.gz')
    oar_mask = sitk.GetArrayFromImage(oar_mask)
    print(oar_mask.max())

    # 计算MAE误差
    Dose_score = get_3D_Dose_dif(pred,gt)
    ptv_score=get_3D_Dose_dif(pred,gt,ptv_mask)

    # 将临床剂量和预测剂量按PTV的95%体积以上为处方剂量进行归一化
    pred_d95=get_d95(pred,ptv_mask)
    gt_d95 = get_d95(gt, ptv_mask)
    gt = gt * pres_dose / gt_d95
    pred = pred*pres_dose/pred_d95
    # 计算 HI，D2,D98,D95,Dmax，Dmean
    ptv_hi_pred=get_hi(pred,ptv_mask)
    ptv_hi_gt=get_hi(gt,ptv_mask)
    # 计算 CI
    ptv_ci_pred=get_ci(pred,ptv_mask,pres_dose)
    ptv_ci_gt=get_ci(gt,ptv_mask,pres_dose)
    # 计算DSCs
    ptv_dice = get_dice(pred, gt,ptv_mask,pres_dose)

    # 计算危及器官相关指标
    # "1": "liver", "2": "right kidney", "3": "left kidney", "4": "stomach", "5": "spinal cord"
    liver_mask = (oar_mask==1)
    liver_mask[ptv_mask>0] = 0 # 将PTV排除
    liver_score=get_3D_Dose_dif(pred,gt,liver_mask)
    rk_score=get_3D_Dose_dif(pred,gt,oar_mask==2)
    lk_score=get_3D_Dose_dif(pred,gt,oar_mask==3)
    stomach_score=get_3D_Dose_dif(pred,gt,oar_mask==4)
    spinal_cord_score = get_3D_Dose_dif(pred, gt, oar_mask == 5)


    liver_pred_metric=get_oar_metric(pred,liver_mask)
    liver_gt_metric = get_oar_metric(gt, liver_mask)

    metric = {'mse': Dose_score,'patient_id':patient_id,
              'ptv_hi_pred':ptv_hi_pred,'ptv_hi_gt':ptv_hi_gt,'ptv_dice':ptv_dice,
              'liver_pred_metric':liver_pred_metric,'liver_gt_metric':liver_gt_metric,
              'ptv_ci_pred':ptv_ci_pred,'ptv_ci_gt':ptv_ci_gt,
              'mse_ptv':ptv_score,'mse_liver':liver_score,'mse_lk':lk_score,'mse_rk':rk_score,'mse_st':stomach_score,'mse_sc':spinal_cord_score,
              }
    return metric

def metric_cal():
    excel_output_path = output_path+ '\\' + dataset + '_' + plan + '.xlsx'
    # 多线程并行处理
    p = Pool(5)
    metrics = p.map(cal_one_patient, filenames)
    p.close()
    p.join()
    # print(metrics)

    me_list = []
    filename_list = []
    ptv_hi_pred,ptv_hi_gt=[],[]
    ptv_dice=[]
    liver_pred_metric,liver_gt_metric=[],[]
    pred_hi,gt_hi=[],[]
    pred_ci, gt_ci = [], []
    pred_d2, pred_d98, pred_d50,pred_dmax,pred_dmean = [], [], [],[],[]
    gt_d2, gt_d98, gt_d50,gt_dmax,gt_dmean = [], [], [],[],[]

    for metric in metrics:
        filename_list.append(metric['patient_id'])
        me_list.append(metric['mse'])
        ptv_hi_pred.append(metric['ptv_hi_pred'])
        ptv_hi_gt.append(metric['ptv_hi_gt'])
        # ptv_hi_pred.append(metric['ptv_ci_pred'])
        # ptv_hi_gt.append(metric['ptv_ci_gt'])
        ptv_dice.append(metric['ptv_dice'])
        liver_pred_metric.append(metric['liver_pred_metric'])
        liver_gt_metric.append(metric['liver_gt_metric'])
        pred_hi.append(metric['ptv_hi_pred']['HI'])
        pred_d2.append(metric['ptv_hi_pred']['d2'])
        pred_d98.append(metric['ptv_hi_pred']['d98'])
        pred_d50.append(metric['ptv_hi_pred']['d50'])
        pred_dmax.append(metric['ptv_hi_pred']['max'])
        pred_dmean.append(metric['ptv_hi_pred']['mean'])
        gt_hi.append(metric['ptv_hi_gt']['HI'])
        gt_d2.append(metric['ptv_hi_gt']['d2'])
        gt_d98.append(metric['ptv_hi_gt']['d98'])
        gt_d50.append(metric['ptv_hi_gt']['d50'])
        gt_dmax.append(metric['ptv_hi_gt']['max'])
        gt_dmean.append(metric['ptv_hi_gt']['mean'])
        gt_ci.append(metric['ptv_ci_gt'])
        pred_ci.append(metric['ptv_ci_pred'])

    df = pd.DataFrame()
    df['filename'] = filename_list
    df['mse'] = me_list
    df['pred_hi'] = pred_hi
    df['gt_hi'] = gt_hi
    df['pred_ci'] = pred_ci
    df['gt_ci'] = gt_ci
    df['pred_d2'] = pred_d2
    df['gt_d2'] = gt_d2
    df['pred_d98'] = pred_d98
    df['gt_d98'] = gt_d98
    df['pred_d50'] = pred_d50
    df['gt_d50'] = gt_d50
    df['pred_dmax'] = pred_dmax
    df['gt_dmax'] = gt_dmax
    df['pred_dmean'] = pred_dmean
    df['gt_dmean'] = gt_dmean
    # df['ptv_hi_pred'] = ptv_hi_pred
    # df['ptv_hi_gt'] = ptv_hi_gt
    # df['ptv_dice'] = ptv_dice
    df['liver_pred_metric'] = liver_pred_metric
    df['liver_gt_metric'] = liver_gt_metric

    print(df.describe())
    df.to_excel(excel_output_path)

    df2 = pd.DataFrame()
    df2['x']=ptv_dice[0]['x_p']
    for i,filename in enumerate(filename_list):
        df2[filename]=ptv_dice[i]['y_dice']
    print(df2.describe())
    df2.to_excel(excel_output_path.replace('.xlsx','_dice.xlsx'))

    mse_ptv,mse_liver,mse_lk,mse_rk,mse_st,mse_sc=[],[],[],[],[],[]
    filename_list = []
    for metric in metrics:
        filename_list.append(metric['patient_id'])
        mse_ptv.append(metric['mse_ptv'])
        mse_liver.append(metric['mse_liver'])
        mse_lk.append(metric['mse_lk'])
        mse_rk.append(metric['mse_rk'])
        mse_st.append(metric['mse_st'])
        mse_sc.append(metric['mse_sc'])
    df3 = pd.DataFrame()
    df3['filename']= filename_list
    df3['mse_ptv'] = mse_ptv
    df3['mse_liver'] = mse_liver
    df3['mse_lk'] = mse_lk
    df3['mse_rk'] = mse_rk
    df3['mse_st'] = mse_st
    df3['mse_sc'] =mse_sc
    print(df3.describe())
    df3.to_excel(excel_output_path.replace('.xlsx','_mse.xlsx'))
    return

def images_show():
    for filename in filenames:
        patient_id = filename.split('\\')[-1].split('.')[0]
        # 获取患者的处方剂量
        pres_dose = prescription_dose[patient_id]
        # 获取预测剂量
        pred = sitk.ReadImage(filename)
        pred = sitk.GetArrayFromImage(pred)
        # 获取临床剂量
        gt = sitk.ReadImage(filename.replace('dose_pred','dose_gt'))
        gt = sitk.GetArrayFromImage(gt)
        # 获取PTV mask
        ptv_masks = sitk.ReadImage(dataset_path + '/ptv/' + patient_id + '.nii.gz')
        ptv_masks = sitk.GetArrayFromImage(ptv_masks)
        # 获取Body mask
        body_masks = sitk.ReadImage(dataset_path + '/body/' + patient_id + '.nii.gz')
        body_masks = sitk.GetArrayFromImage(body_masks)

        # 将临床剂量和预测剂量按PTV的95%体积以上为处方剂量进行归一化
        pred_d95 = get_d95(pred, ptv_masks)
        gt_d95 = get_d95(gt, ptv_masks)
        gt = gt * pres_dose / gt_d95
        pred = pred * pres_dose / pred_d95
        # 获取PTV中Z轴方向不为零的切片索引
        im_z_sum = np.sum(np.sum(ptv_masks, axis=1), axis=1)
        im_z_idx = np.nonzero(im_z_sum)
        for z in im_z_idx[0]:
            slice_gt_dose_o=gt[z,:,:]
            slice_pred_dose_o=pred[z,:,:]
            body_mask = body_masks[z,:,:]
            body_mask1 = np.repeat(np.expand_dims(body_mask, axis=2), 3, axis=2)
            f_max=slice_gt_dose_o.max()
            ### slice_gt_dose = slice_gt_dose_o/f_max*255
            ### slice_pred_dose = slice_pred_dose_o / f_max * 255
            slice_gt_dose = cv2.normalize(slice_gt_dose_o,None,0,255,norm_type=cv2.NORM_MINMAX)
            slice_gt_dose =slice_gt_dose.astype('uint8')
            slice_gt_dose = cv2.applyColorMap(slice_gt_dose,colormap=cv2.COLORMAP_JET)
            slice_gt_dose = slice_gt_dose * body_mask1
            slice_pred_dose = cv2.normalize(slice_pred_dose_o,None,0,255,norm_type=cv2.NORM_MINMAX)
            slice_pred_dose =slice_pred_dose.astype('uint8')
            slice_pred_dose = cv2.applyColorMap(slice_pred_dose,colormap=cv2.COLORMAP_JET)
            slice_pred_dose =slice_pred_dose * body_mask1
            cv2.imwrite(output_path+patient_id+'_'+str(z)+'_pred.png',slice_pred_dose)
            cv2.imwrite(output_path + patient_id + '_' + str(z) + '_gt.png', slice_gt_dose)
            # # 误差显示
            err_o=abs(slice_gt_dose_o-slice_pred_dose_o)
            print("最大误差：{}，最小误差：{}".format(err_o.max(), err_o.min()))
            err = np.clip(err_o,a_max=30,a_min=0)
            err = err/30*255
            # err = cv2.normalize(err_o,None,0,255,norm_type=cv2.NORM_MINMAX)
            err =err.astype('uint8')
            err = cv2.applyColorMap(err,colormap=cv2.COLORMAP_JET)
            err = err * body_mask1
            cv2.imwrite(output_path + patient_id + '_' + str(z) + '_error.png', err)

    return

if __name__ == '__main__':
    pred_path = os.path.join('data', 'hutest', 'hupred')
    gt_path = os.path.join('data', 'hutest', 'hutest')
    dataset_path = os.path.join('data', 'dose_128', 'reprocess_data')
    output_path = os.path.join('data', 'hutest', 'diff_data_128out')
    dataset = 'liver'
    plan = 'try'
    def maybe_mkdir_p(path):
        if not os.path.exists(path):
            os.makedirs(path)


    maybe_mkdir_p(output_path)
    print('output_path', output_path)
    # prescription_dose = {'624608': 45., '649046': 40., '686348': 50., '702090': 48., '710585': 40.}
    prescription_dose = {'300780': 54., '700810': 48., '622076': 54., '710585': 40.}
    # 按病人分别计算指标
    filenames = subfiles(pred_path, suffix='.nii.gz')
    filenames = [f for f in filenames]
    print('病人序列号：', filenames)
    metric_cal()
    images_show()