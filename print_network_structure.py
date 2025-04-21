import torch
from torchviz import make_dot
import netron

from guided_diffusion.unet import UNetModel_MS_Former,UNetModel_hu
# 假设 UNetModel_hu 是你的模型类
model = UNetModel_hu(image_size=(128, 128), in_channels=1, ct_channels=1, dis_channels=21,
                       model_channels=128, out_channels=1, num_res_blocks=2, attention_resolutions=(16, 32),
                       dropout=0,
                       channel_mult=(1, 1, 2, 3, 4), conv_resample=True, dims=2, num_classes=None,
                       use_checkpoint=False,
                       use_fp16=False, num_heads=4, num_head_channels=-1, num_heads_upsample=-1,
                       use_scale_shift_norm=True,
                       resblock_updown=False, use_new_attention_order=False)
# 创建一个虚拟输入
x = torch.randn(6,1,128, 128)
ct = torch.randn(6,1,128, 128)
dis = torch.randn(6,21,128, 128)
timesteps = torch.tensor([512])

# 获取模型输出
# output = model(x, timesteps, ct, dis)
#
# # 可视化模型
# dot = make_dot(output, params=dict(model.named_parameters()))
#
# dot.render("Results/output", format="png")
# 将模型导出为 ONNX 格式
torch.onnx.export(model, (x, timesteps, ct, dis), "unet_model.onnx",export_params=True,verbose=True, keep_initializers_as_inputs=True)

# 使用 Netron 打开模型可视化
netron.start("unet_model.onnx")