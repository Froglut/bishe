import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions



"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    print(">>> STEP 1: 解析参数")
    opt = TrainOptions().parse()
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    val_opt = get_val_opt()

    print(">>> STEP 2: 创建数据加载器")
    data_loader = create_dataloader(opt, fusion_mode=opt.fusion_mode)
    print(">>> STEP 3: 数据加载器创建成功")

    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    print(">>> STEP 4: 创建模型")
    model = Trainer(opt)
    print(">>> STEP 5: 模型创建成功")
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    print(">>> STEP 6: 开始训练循环")
    for epoch in range(opt.niter):
        print(f">>> EPOCH {epoch} 开始")
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size
            print(f">>> BATCH {i} 开始", end=" ")
            model.set_input(data)
            print("-> set_input完成", end=" ")
            model.optimize_parameters()
            print("-> 优化完成")

            if i >= 5:  # 只看前几个batch
                print(">>> 前5个batch完成，继续观察...")
                break

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')

            # print("Iter time: %d sec" % (time.time()-iter_data_time))
            # iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation 验证部分
        print(f">>> EPOCH {epoch} 结束，开始验证")
        model.eval()
        acc, ap = validate(model.model, val_opt,use_gradient=opt.use_grad,fusion_mode=opt.fusion_mode)[:2]
        print(f">>> 验证完成: acc={acc}, ap={ap}")
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()
