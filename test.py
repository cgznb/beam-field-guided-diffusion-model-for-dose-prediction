import os
import numpy as np
import torch
import argparse
import shutil
import pandas as pd
import ast
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from Nii_utils import NiiDataRead, NiiDataWrite
from guided_diffusion.unet import UNetModel_MS_Former,UNetModel_hu
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from evaluate_openKBP import get_Dose_score_and_DVH_score,get_metrics
#

get_metrics(prediction_dir=os.path.join('Resultshu/ddim8_TTA1_Timealltest_setp', 'predictions'),
            gt_dir=r'data/hutest/alltest',
            save_dir=r'Resultshu/ddim8_TTA1_Timealltest_setp',
            excel_Path = r'data\hutest\xy2_statistic.xlsx')
# # 示例列表，包含字符串形式的数字
# str_list = "['1.1']"
#
# # 将每个字符串元素转换为浮点数
# float_list = float(str_list[0])
#
# print(float_list)  # 输出: [1.1, 2.2, 3.3]
# pres_dose = df.loc[df['patient_name'] == int(patient_id), 'MAE'].to_numpy().tolist()[0]
# pres_dose = ast.literal_eval(pres_dose)
# pres_dose = float(pres_dose[0])
#
# def calculate_mean_metrics_dif(save_dir,metric_dif_list,structures):
#     excel_path = os.path.join(save_dir,'metrics_dif.xlsx')
#     output_txt_path =  os.path.join(save_dir, 'mean_metrics_dif.txt')
#     output_img_path = os.path.join(save_dir, 'mean_metrics_dif_fig.png')
#
#     df = pd.read_excel(excel_path)
#     # patient_ids = df.iloc[:, 1]  # 读取第一列
#     results = {structure: {metric: {'mean': None, 'std': None} for metric in metric_dif_list} for structure in structures}
#     # 遍历每个结构
#     for i, structure in enumerate(structures):
#         for metric in metric_dif_list:
#             # 获取当前结构和指标对应的列名
#             column_name = f'{metric}'
#             if column_name in df.columns:
#                 values = df[column_name].to_numpy().tolist()
#                 value = []
#                 for data in values:
#                     # data = pd.to_numeric(data, errors='coerce')
#                     print(type(data))
#                     if isinstance(data, str):
#                         data = data.replace('nan', '999').replace('inf', '999')
#                         str_met = ast.literal_eval(data)
#                         if str_met[i] != 999:
#                             value.append(str_met[i])
#                     else:
#                         value.append(data)
#                 mean_value = np.mean(value)
#                 std_value = np.std(value)
#                 results[structure][metric]['mean'] = mean_value
#                 results[structure][metric]['std'] = std_value
#     with open(output_txt_path, 'w') as f:
#         for structure, metrics_data in results.items():
#             f.write(f"Structure: {structure}\n")
#             for metric, stats in metrics_data.items():
#                 f.write(f"  {metric} - Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}\n")
#
#     fig, axes = plt.subplots(len(structures), 1, figsize=(10, 15))
#     fig.tight_layout(pad=5.0)
#     for idx, structure in enumerate(structures):
#         mean_values = []
#         metric_names = []
#
#         for metric in metric_dif_list:
#             mean_values.append(results[structure][metric]['mean'])
#             metric_names.append(metric)
#
#         bars =  axes[idx].bar(metric_names, mean_values,capsize=5)
#         axes[idx].set_title(f'{structure} - Mean Values')
#         axes[idx].set_ylabel('Mean Value')
#         axes[idx].set_xlabel('Metrics')
#         for bar in bars:
#                 yval = bar.get_height()
#                 axes[idx].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')
#
#     # 保存图像
#     plt.savefig(output_img_path)
#     plt.show()
#     return
# import numpy as np
#
# # 定义一个包含 NaN 值的数组
# data_with_nan = np.array([1, 2, np.nan, 0, 5, np.nan, 7])
#
# # 使用 np.nanmean 计算均值，自动忽略 NaN 值
# mean_value = np.nanmean(data_with_nan)
#
# print(f"The mean value (ignoring NaN) is: {mean_value:.2f}")