import numpy as np
import os
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import ast
import seaborn as sns
"""
These codes are modified from https://github.com/ababier/open-kbp
"""

def evaluate_excel_data(excel_Path):
    # df = pd.read_excel(excel_Path)
    # pres_dose = df.loc[df['patient_name'] == int(patient_id), 'TargetPrescriptionDose'].to_numpy().tolist()[0]
    # pres_dose = ast.literal_eval(pres_dose)
    # pres_dose = float(pres_dose[0])
    return

def get_3D_Dose_dif(pred, gt, possible_dose_mask=None):
    if possible_dose_mask is not None:
        pred = pred[possible_dose_mask > 0]
        gt = gt[possible_dose_mask > 0]

    dif = np.mean(np.abs(pred - gt))
    return dif


def get_DVH_metrics(_dose, _mask, mode, spacing=None):
    output = {}

    if mode == 'target':
        _roi_dose = _dose[_mask > 0]
        _roi_size = len(_roi_dose)
        if _roi_size == 0:
            output['D1'] = np.nan
            output['D95'] = np.nan
            output['D99'] = np.nan
        else:
            # D1
            output['D1'] = np.percentile(_roi_dose, 99)
            # D95
            output['D95'] = np.percentile(_roi_dose, 5)
            # D99
            output['D99'] = np.percentile(_roi_dose, 1)

    elif mode == 'OAR':
        if spacing is None:
            raise Exception('calculate OAR metrics need spacing')

        _roi_dose = _dose[_mask > 0]
        _roi_size = len(_roi_dose)

        if _roi_size == 0:
            output['D_0.1_cc'] = np.nan
            output['mean'] = np.nan
        else:
            _voxel_size = np.prod(spacing)
            voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / _voxel_size))
            # D_0.1_cc
            fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / _roi_size * 100
            output['D_0.1_cc'] = np.percentile(_roi_dose, fractional_volume_to_evaluate)
            # Dmean
            output['mean'] = np.mean(_roi_dose)
    else:
        raise Exception('Unknown mode!')

    return output

def get_pres_dose(dicom_file):
    patient_name = dicom_file.split('/RP')[0].split('/')[-2]
    prescription_dose = []
    dicom_rp = pydicom.dcmread(dicom_file, force=True)
    for dr in dicom_rp.DoseReferenceSequence:
        prescription_dose.append(dr.TargetPrescriptionDose)
    return prescription_dose

def get_Dose_score_and_DVH_score(prediction_dir, gt_dir):
    list_dose_dif = []
    list_DVH_dif = []

    list_patient_ids = tqdm(os.listdir(prediction_dir))
    for patient_id in list_patient_ids:
        pred_nii = sitk.ReadImage(prediction_dir + '/' + patient_id + '/dose.nii.gz')
        pred = sitk.GetArrayFromImage(pred_nii)

        gt_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/dose.nii.gz')
        gt = sitk.GetArrayFromImage(gt_nii)

        # Dose dif
        possible_dose_mask_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/body.nii.gz')
        possible_dose_mask = sitk.GetArrayFromImage(possible_dose_mask_nii)
        list_dose_dif.append(get_3D_Dose_dif(pred, gt, possible_dose_mask))



        # DVH dif
        for structure_name in ['gtv'
                               'SpinalCord',
                               'Heart',
                               'Kidney_L',
                               'Kidney_R',
                               'Liver',
                               'Stomach',
                               'ptv']:
            structure_file = gt_dir + '/' + patient_id + '/' + structure_name + '.nii.gz'

            # If the structure has been delineated
            if os.path.exists(structure_file):
                structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
                structure = sitk.GetArrayFromImage(structure_nii)

                spacing = structure_nii.GetSpacing()
                if structure_name.find('ptv') > -1:
                    mode = 'target'
                else:
                    mode = 'OAR'
                pred_DVH = get_DVH_metrics(pred, structure, mode=mode, spacing=spacing)
                gt_DVH = get_DVH_metrics(gt, structure, mode=mode, spacing=spacing)

                for metric in gt_DVH.keys():
                    list_DVH_dif.append(abs(gt_DVH[metric] - pred_DVH[metric]))

    return np.mean(list_dose_dif), np.std(list_dose_dif), np.mean(list_DVH_dif), np.std(list_DVH_dif)

def calculate_all_metrics(dose, structure,spacing,pres_dose):

    _roi_dose = dose[structure > 0]
    _roi_size = len(_roi_dose)

    if _roi_size == 0:
        D2 = np.NaN
        D95 = np.NaN
        D98 = np.NaN
        D_0dot1_cc = np.NaN
        Dmean = np.NaN
        D50 = np.NaN
        Dmax = np.NaN
        HI = np.NaN
        CI = np.NaN
    else:
        # D2
        D2 = np.percentile(_roi_dose, 98)
        # D95
        D95 = np.percentile(_roi_dose, 5)
        # D99
        D98 = np.percentile(_roi_dose, 2)
        # D50
        D50 = np.percentile(_roi_dose, 50)
        # max
        Dmax = np.max(_roi_dose)
        # HI
        HI = (D2-D98)/D50
        CI = get_ci(dose,structure,pres_dose)
        # D_0.1_cc
        _voxel_size = np.prod(spacing)
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / _voxel_size))
        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / _roi_size * 100
        D_0dot1_cc = np.percentile(_roi_dose, fractional_volume_to_evaluate)
        # Dmean
        Dmean = np.mean(_roi_dose)

    return D2, D95, D98, D_0dot1_cc, Dmean, D50, Dmax, HI, CI
def show_all_metrics(dose, structure,spacing,pres_dose):

    return
def get_d95(_dose,_mask):
    _roi_dose = _dose[_mask > 0]
    _roi_size = len(_roi_dose)
    out = np.percentile(_roi_dose, 5) # d95
    return out

def calculate_show_mean_metrics_dif(save_dir,metric_dif_list,structures):
    excel_path = os.path.join(save_dir,'metrics_dif.xlsx')
    output_txt_path =  os.path.join(save_dir, 'mean_metrics_dif.txt')
    output_img_path = os.path.join(save_dir, 'mean_metrics_dif_fig.png')

    df = pd.read_excel(excel_path)
    # patient_ids = df.iloc[:, 1]  # 读取第一列
    results = {structure: {metric: {'mean': None, 'std': None} for metric in metric_dif_list} for structure in structures}
    # 遍历每个结构
    for i, structure in enumerate(structures):
        for metric in metric_dif_list:
            # 获取当前结构和指标对应的列名
            column_name = f'{metric}'
            if column_name in df.columns:
                values = df[column_name].to_numpy().tolist()
                value = []
                for data in values:
                    # data = pd.to_numeric(data, errors='coerce')
                    print(type(data))
                    if isinstance(data, str):
                        data = data.replace('nan', '999').replace('inf', '999')
                        str_met = ast.literal_eval(data)
                        if str_met[i] != 999:
                            value.append(str_met[i])
                    else:
                        value.append(data)
                mean_value = np.mean(value)
                std_value = np.std(value)
                results[structure][metric]['mean'] = mean_value
                results[structure][metric]['std'] = std_value
    with open(output_txt_path, 'w') as f:
        for structure, metrics_data in results.items():
            f.write(f"Structure: {structure}\n")
            for metric, stats in metrics_data.items():
                f.write(f"  {metric} - Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}\n")

    fig, axes = plt.subplots(len(structures), 1, figsize=(10, 15))
    fig.tight_layout(pad=5.0)
    for idx, structure in enumerate(structures):
        mean_values = []
        metric_names = []

        for metric in metric_dif_list:
            mean_values.append(results[structure][metric]['mean'])
            metric_names.append(metric)

        bars =  axes[idx].bar(metric_names, mean_values,capsize=5)
        axes[idx].set_title(f'{structure} - Mean Values')
        axes[idx].set_ylabel('Mean Value')
        # axes[idx].set_xlabel('Metrics')

        # 旋转X轴标签，避免重叠
        axes[idx].set_xticklabels(metric_names, rotation=45, ha='right')

        for bar in bars:
                yval = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')
    # 再次自动调整布局，确保旋转后的标签不与其他子图重叠
    # plt.tight_layout()
    # 保存图像
    plt.savefig(output_img_path)
    plt.show()
    return
def save_show_mean_metric(metric_list,structures,patient_structure_metrics,save_dir):
    average_metrics = {}
    variance_metrics = {}
    for structure_name in structures:
        average_metrics[structure_name] = {
            'gt': {},
            'pred': {}
        }
        variance_metrics[structure_name] = {
            'gt': {},
            'pred': {}
        }
        for metric_name in metric_list:
            gt_values = []
            pred_values = []

            for patient_id in patient_structure_metrics:
                gt_values.extend(patient_structure_metrics[patient_id][structure_name]['gt'][metric_name])
                pred_values.extend(patient_structure_metrics[patient_id][structure_name]['pred'][metric_name])
            average_metrics[structure_name]['gt'][metric_name] = np.nanmean(gt_values)
            variance_metrics[structure_name]['gt'][metric_name] = np.nanstd(gt_values)
            average_metrics[structure_name]['pred'][metric_name] = np.nanmean(pred_values)
            variance_metrics[structure_name]['pred'][metric_name] = np.nanstd(pred_values)

    txt_path = os.path.join(save_dir, 'mean_metrics_summary.txt')

    # 保存结果到txt文件
    with open(txt_path, 'w') as f:
        for structure, data in average_metrics.items():
            f.write(f"Structure: {structure}\n")
            for metric_name in metric_list:
                gt_mean = data['gt'][metric_name]
                gt_var = variance_metrics[structure]['gt'][metric_name]
                pred_mean = data['pred'][metric_name]
                pred_var = variance_metrics[structure]['pred'][metric_name]
                f.write(f"  {metric_name}:\n")
                f.write(f"    GT  - Mean: {gt_mean:.2f}, Variance: {gt_var:.2f}\n")
                f.write(f"    Pred - Mean: {pred_mean:.2f}, Variance: {pred_var:.2f}\n")

    # 使用 Seaborn 设置配色方案
    sns.set(style="whitegrid")
    dopamine_colors = sns.color_palette("hsv", 10)  # 多巴胺配色风格

    # 可视化每个结构的 gt 和 pred 的平均值
    # 可视化并保存图像
    for structure in structures:
        gt_means = [average_metrics[structure]['gt'][metric] for metric in metric_list]
        pred_means = [average_metrics[structure]['pred'][metric] for metric in metric_list]
        gt_stds = [variance_metrics[structure]['gt'][metric] for metric in metric_list]
        pred_stds = [variance_metrics[structure]['pred'][metric] for metric in metric_list]

        x = np.arange(len(metric_list))  # X轴的位置
        width = 0.35  # 柱状图的宽度

        fig, ax = plt.subplots(figsize=(10, 6))
        bars_gt = ax.bar(x - width / 2, gt_means, width, yerr=gt_stds, label='GT', color=dopamine_colors[0], capsize=5)
        bars_pred = ax.bar(x + width / 2, pred_means, width, yerr=pred_stds, label='Pred', color=dopamine_colors[1],
                           capsize=5)

        # 添加标题和标签
        ax.set_xlabel('Metrics', fontsize=14)
        ax.set_ylabel('Mean Values', fontsize=14)
        ax.set_title(f'{structure} - GT vs Pred Mean Values with Std', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_list, fontsize=12, rotation=45)
        ax.legend()

        # 在柱子上显示数值
        # for bars in [bars_gt, bars_pred]:
        #     for bar in bars:
        #         yval = bar.get_height()
        #         ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
        for bar in bars_gt:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 4, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

        for bar in bars_pred:
            yval = bar.get_height()
            ax.text(bar.get_x() + 3 * bar.get_width() / 4, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
        # 自动调整布局
        # plt.tight_layout()
        # 保存图像
        fig_dir = os.path.join(save_dir, 'mean_figs')

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_path = os.path.join(fig_dir, f'{structure}_gt_vs_pred.png')
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close()
    return

def clean_dose(dose):
    try:
        # Try to extract the first valid float value from the string
        return float(dose.strip("[]'"))
    except (ValueError, TypeError):
        # If there's an error, return None
        return None

def get_ci(_dose,_ptv_mask,_pres_dose):
    _ptv_mask[_ptv_mask>0]=1
    iso_dose=(_dose>=_pres_dose)
    iso_dose = np.array(iso_dose,dtype='uint8')
    P=iso_dose.sum()
    T=_ptv_mask.sum()
    T_P=(_ptv_mask*iso_dose).sum()
    ci=(T_P)**2/(P*T)
    return ci

def get_metrics(prediction_dir, gt_dir, save_dir,excel_Path):

    list_patients_id = []
    list_dose_difs = []
    list_structures = []
    list_D2_difs = []
    list_D95_difs = []
    list_D98_difs = []
    list_D_0dot1_cc_difs = []
    list_Dmean_difs = []
    list_MAEs = []
    list_HI_difs = []
    list_CI_difs = []
    list_D50_difs = []
    list_Dmax_difs = []
    # 创建字典，将所有差异列表与对应的总列表关联
    all_dif_lists = {
        'D2_difs': list_D2_difs,
        'D95_difs': list_D95_difs,
        'D98_difs': list_D98_difs,
        'D_0dot1_cc_difs': list_D_0dot1_cc_difs,
        'Dmean_difs': list_Dmean_difs,
        'MAEs': list_MAEs,
        'D50_difs': list_D50_difs,
        'Dmax_difs': list_Dmax_difs,
        'HI_difs': list_HI_difs,
        'CI_difs': list_CI_difs
    }
    df = pd.read_excel(excel_Path)
    # structures = ['gtv','SpinalCord','Heart','Kidney_L','Kidney_R','Liver','Stomach','ptv','body']
    structures = ['gtv', 'SpinalCord','Heart', 'Kidney_L', 'Kidney_R', 'Liver', 'Stomach', 'ptv', 'body']
    metric_list = ['D2', 'D95', 'D98', 'D_0dot1_cc', 'mean', 'D50', 'Dmax', 'HI', 'CI', 'MAE']
    metric_dif_list = ['D2_dif', 'D95_dif', 'D98_dif', 'D_0dot1_cc_dif', 'mean_dif', 'dose_dif', 'MAE', 'HI_dif', 'CI_dif',
                       'D50_dif', 'Dmax_dif']
    list_patient_ids = tqdm(os.listdir(prediction_dir))
    patient_structure_metrics = {patient_id: {structure_name: {
        'gt': {metric_name: [] for metric_name in metric_list},
        'pred': {metric_name: [] for metric_name in metric_list}
    } for structure_name in structures} for patient_id in list_patient_ids}

    for patient_id in list_patient_ids:
        list_D2_dif = []
        list_D95_dif = []
        list_D98_dif = []
        list_D_0dot1_cc_dif = []
        list_Dmean_dif = []
        list_MAE = []
        list_HI_dif= []
        list_CI_dif = []
        list_D50_dif = []
        list_Dmax_dif = []

        list_patients_id.append(patient_id)
        list_structures.append(structures)

        pred_nii = sitk.ReadImage(prediction_dir + '/' + patient_id + '/dose.nii.gz')
        pred = sitk.GetArrayFromImage(pred_nii)

        gt_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/dose.nii.gz')
        gt = sitk.GetArrayFromImage(gt_nii)
        gt[gt < 0] = 0

        # Dose dif
        possible_dose_mask_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/body.nii.gz')
        possible_dose_mask = sitk.GetArrayFromImage(possible_dose_mask_nii)
        list_dose_difs.append(get_3D_Dose_dif(pred, gt, possible_dose_mask))


        structure_file = gt_dir + '/' + patient_id + '/' + 'ptv' + '.nii.gz'
        # If the structure has been delineated
        if os.path.exists(structure_file):
            structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
            structure = sitk.GetArrayFromImage(structure_nii)
        # pred_d95 = get_d95(pred, structure)
        # gt_d95 = get_d95(gt, structure)
        pres_dose = df.loc[df['patient_name'] == int(patient_id), 'TargetPrescriptionDose'].to_numpy().tolist()[0]
        pres_dose = ast.literal_eval(pres_dose)
        pres_dose = float(pres_dose[0])
        # gt = gt * pres_dose / gt_d95
        # pred = pred * pres_dose/ pred_d95
        metric_lists = {
            'D2_dif': list_D2_dif,
            'D95_dif': list_D95_dif,
            'D98_dif': list_D98_dif,
            'D_0dot1_cc_dif': list_D_0dot1_cc_dif,
            'Dmean_dif': list_Dmean_dif,
            'D50_dif': list_D50_dif,
            'Dmax_dif': list_Dmax_dif,
            'HI_dif': list_HI_dif,
            'CI_dif': list_CI_dif
        }
        # DVH dif
        for structure_name in structures:
            structure_file = gt_dir + '/' + patient_id + '/' + structure_name + '.nii.gz'
            structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
            structure = sitk.GetArrayFromImage(structure_nii)
            spacing = structure_nii.GetSpacing()

            if spacing is None:
                raise Exception('calculate OAR metrics need spacing')
            # pred_D2, pred_D95, pred_D98, pred_D_0dot1_cc, pred_mean,pred_D50,pred_Max,pred_HI,pred_CI = get_all_metrics(pred, structure, spacing,pres_dose)
            # gt_D2, gt_D95, gt_D98, gt_D_0dot1_cc, gt_mean,gt_D50,gt_Max,gt_HI,gt_CI = get_all_metrics(gt, structure, spacing,pres_dose)
            gt_metrics= calculate_all_metrics(gt, structure, spacing, pres_dose)
            pred_metrics = calculate_all_metrics(pred, structure, spacing, pres_dose)
            #metric_list = ['D2', 'D95', 'D98', 'D_0dot1_cc', 'mean', 'D50', 'Dmax', 'HI', 'CI', 'MAE']
            for i, metric_name in enumerate(metric_list):
                if metric_name == 'MAE':
                    patient_structure_metrics[patient_id][structure_name]['gt'][metric_name].append(get_3D_Dose_dif(pred, gt, structure))
                    patient_structure_metrics[patient_id][structure_name]['pred'][metric_name].append(get_3D_Dose_dif(pred, gt, structure))
                else:
                    patient_structure_metrics[patient_id][structure_name]['gt'][metric_name].append(gt_metrics[i])
                    patient_structure_metrics[patient_id][structure_name]['pred'][metric_name].append(pred_metrics[i])



            diffs = [abs(p - g) for p, g in zip(pred_metrics, gt_metrics)]
            for key, diff_value in zip(metric_lists.keys(), diffs):
                metric_lists[key].append(diff_value)
            list_MAE.append(get_3D_Dose_dif(pred, gt, structure))
            current_dif_lists = [
                list_D2_dif, list_D95_dif, list_D98_dif, list_D_0dot1_cc_dif,
                list_Dmean_dif, list_MAE, list_D50_dif, list_Dmax_dif,
                list_HI_dif, list_CI_dif
            ]
        for key, current_dif in zip(all_dif_lists.keys(), current_dif_lists):
            all_dif_lists[key].append(current_dif)
        txt_path = os.path.join(save_dir, 'metrics_summary.txt')
        with open(txt_path, 'w') as f:
            for patient_id, structures_data in patient_structure_metrics.items():
                f.write(f'Patient ID: {patient_id}\n')
                for structure_name, metrics_data in structures_data.items():
                    f.write(f'  Structure: {structure_name}\n')
                    # 写入 GT 数据
                    f.write('    GT Metrics:\n')
                    for metric_name, values in metrics_data['gt'].items():
                        if values:  # 检查 values 是否非空
                            value = values[0]  # 如果非空，提取第一个元素
                            f.write(f'      {metric_name}: {value:.2f}\n')
                        else:
                            f.write(f'      {metric_name}: \n')  # 如果为空，保持空

                    # 写入 Pred 数据
                    f.write('    Pred Metrics:\n')
                    for metric_name, values in metrics_data['pred'].items():
                        if values:  # 检查 values 是否非空
                            value = values[0]  # 如果非空，提取第一个元素
                            f.write(f'      {metric_name}: {value:.2f}\n')
                        else:
                            f.write(f'      {metric_name}: \n')  # 如果为空，保持空

    save_show_mean_metric(metric_list, structures, patient_structure_metrics, save_dir)
    df = pd.DataFrame()
    df['patient_name'] = list_patients_id
    df['structure'] = list_structures
    df['D2_dif'] = list_D2_difs
    df['D95_dif'] = list_D95_difs
    df['D98_dif'] = list_D98_difs
    df['D_0dot1_cc_dif'] = list_D_0dot1_cc_difs
    df['mean_dif'] = list_Dmean_difs
    df['dose_dif'] = list_dose_difs
    df['MAE'] = list_MAEs
    df['HI_dif'] = list_HI_difs
    df['CI_dif'] = list_CI_difs
    df['D50_dif'] = list_D50_difs
    df['Dmax_dif'] = list_Dmax_difs
    df.to_excel(save_dir + '/metrics_dif.xlsx')
    calculate_show_mean_metrics_dif(save_dir, metric_dif_list, structures)


    return