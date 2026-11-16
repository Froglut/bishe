import torch
import numpy as np
import time
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader


def validate(model, opt, use_gradient=True, fusion_mode=None):
    """
    验证函数，兼容 RGB、梯度替换、双流三种模式
    """
    # 确定模式
    if fusion_mode is None:
        if use_gradient:
            fusion_mode = 'grad'
        else:
            fusion_mode = 'rgb'
    
    print("\n" + "="*60)
    print(f"开始验证流程")
    print(f"   模式: {fusion_mode}")
    print(f"   数据集: {opt.dataroot}")
    print("="*60)
    
    # 创建数据加载器
    print("[1/3] 创建DataLoader...", flush=True)
    start_time = time.time()
    data_loader = create_dataloader(opt, fusion_mode=fusion_mode)
    print(f"   完成 (耗时: {time.time()-start_time:.2f}秒)")
    print(f"   批次数量: {len(data_loader)}")
    
    # 推理
    print("[2/3] 开始模型推理...", flush=True)
    print("-" * 40)
    
    with torch.no_grad():
        y_true = []
        y_pred = []
        total_batches = len(data_loader)
        
        for idx, data in enumerate(data_loader):
            # ========== 根据模式解包数据 ==========
            if fusion_mode == 'concat':
                # 双流模式：返回 (rgb, grad, label)
                rgb, grad, label = data
                
                # 移动到设备
                if torch.cuda.is_available():
                    rgb = rgb.cuda()
                    grad = grad.cuda()
                else:
                    rgb = rgb
                    grad = grad
                
                # ========== 双流模型需要两个输入 ==========
                output = model(rgb, grad)
                # ======================================
            else:
                # RGB 或梯度模式：返回 (img, label)
                img, label = data
                
                if torch.cuda.is_available():
                    img = img.cuda()
                else:
                    img = img
                
                # 单流模型只需要一个输入
                output = model(img)
            # ====================================
            
            pred = output.sigmoid().flatten().tolist()
            
            y_pred.extend(pred)
            y_true.extend(label.flatten().tolist())
            
            if (idx + 1) % 50 == 0 or (idx + 1) == total_batches:
                print(f"   Batch [{idx+1:4d}/{total_batches}] 已完成", flush=True)
        
        print("-" * 40)
        print(f"推理完成: 共 {len(y_true)} 个样本")
    
    # 计算指标（保持不变）
    print("[3/3] 计算评估指标...", end=" ", flush=True)
    start_time = time.time()
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)
    r_acc = accuracy_score(y_true[real_mask], y_pred_binary[real_mask]) if real_mask.sum() > 0 else 0.0
    f_acc = accuracy_score(y_true[fake_mask], y_pred_binary[fake_mask]) if fake_mask.sum() > 0 else 0.0
    acc = accuracy_score(y_true, y_pred_binary)
    ap = average_precision_score(y_true, y_pred)
    
    print(f"完成 (耗时: {time.time()-start_time:.2f}秒)")
    
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)
    print(f"   准确率 (Acc): {acc:.4f}")
    print(f"   平均精度 (AP): {ap:.4f}")
    print(f"   真实图准确率: {r_acc:.4f}")
    print(f"   伪造图准确率: {f_acc:.4f}")
    print("="*60 + "\n")
    
    return acc, ap, r_acc, f_acc, y_true, y_pred