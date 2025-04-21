import torch
from guided_diffusion.unet import UNetModel_MS_Former

# 初始化模型（替换为你的参数）
model = UNetModel_MS_Former(
    image_size=(128, 128),
    in_channels=1,
    ct_channels=1,
    dis_channels=19,
    model_channels=128,
    out_channels=1,
    num_res_blocks=2,
    attention_resolutions=(8, 16),
    dropout=0,
    channel_mult=(1, 1, 2, 4),
    conv_resample=True,
    dims=2,
    num_classes=None,
    use_checkpoint=False,
    use_fp16=False,
    num_heads=4,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=False,
    use_new_attention_order=False
)

# 计算参数
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
param_MB = total_params * 4 / (1024 ** 2)
param_B = total_params / 1e9

# 打印结果
print(f"Total Parameters: {total_params:,}  ({param_MB:.2f} MB ≈ {param_B:.3f} B)")
