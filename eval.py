import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from networks.dual_stream_resnet import DualStreamResNet
from options.test_options import TestOptions
from eval_config import *

def load_model(model_path):
    """自动检测并加载模型"""
    state_dict = torch.load(model_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    # 检查键名判断模型类型
    first_key = list(state_dict.keys())[0]
    
    if first_key.startswith('rgb_branch.') or first_key.startswith('grad_branch.'):
        print("[INFO] 检测到双流模型")
        model = DualStreamResNet(fusion_method='concat', pretrained=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("[INFO] 检测到单流模型")
        model = resnet50(num_classes=1)
        model.load_state_dict(state_dict, strict=True)
    
    return model

# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')

print(f"加载模型: {model_name}")
model = load_model(model_path)
model.cpu()
model.eval()

# 设置测试模式（根据你的需要）
USE_GRADIENT = False  # 单流时:RGB为False Grad为True
fusion_mode = None  # 双流时传 'concat'

# 自动判断模式
if isinstance(model, DualStreamResNet):
    fusion_mode = 'concat'
    mode_str = "DualStream"
else:
    mode_str = "Gradient" if USE_GRADIENT else "RGB"

rows = [[f"{model_name} testing on {mode_str} mode"],
        ['testset', 'accuracy', 'avg precision']]

print(f"{model_name} model testing on {mode_str} mode...")

for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True

    acc, ap, r_acc, f_acc, _, _ = validate(model, opt, 
                                           use_gradient=USE_GRADIENT, 
                                           fusion_mode=fusion_mode)
    rows.append([val, acc, ap, r_acc, f_acc])
    print(f"({val}) acc: {acc:.4f}; ap: {ap:.4f}")

# 保存结果
csv_name = results_dir + f'/{model_name}_{mode_str}.csv'
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)