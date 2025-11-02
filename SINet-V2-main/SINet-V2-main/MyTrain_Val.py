# author: Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# data: 2021-01-16
import os
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn

def fun(x):
    return 1-x

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)#hjc 加权BCE更关注于难像素，参考F3net
                                                                                #选取了周围15个像素作为周围像素,添加这个的目的是提高训练困难样本的速度
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')#hjc with_logits可以避免数据溢出，reduce的作用主要是对于NLP
                                                            #其自然语言与上下的字母有关。而图片之间除非是视频，否则关联性不大，即上一张图片对下一张图片影响不大。
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))#权重加成

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))#hjc dim是消除第二三维的数据，只对该batchsize中的图像进行权重更新
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)#hjc 损失函数可以改进，如采用GIoU进行改进，因为小目标会存在重叠区域为0的情况
    return (wiou + wbce).mean()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    train_loss_all = 0
    epoch_step = 0
    acc_sum = 0
    F1_sum = 0
    IoU_sum = 0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gtsp = gts[0][0]
            gtsp = np.asarray(gtsp, np.float32)
            gtsp /= (gtsp.max() + 1e-8)
            gts = gts.cuda()

            preds = model(images)
            loss_init = structure_loss(preds[0], gts) + structure_loss(preds[1], gts) + structure_loss(preds[2], gts) #hjc 先验损失
            loss_final = structure_loss(preds[3], gts) #hjc 候选损失

            loss = loss_init + loss_final
            #训练集单张图片损失
            train_loss_init = structure_loss(preds[0][:1], gts[:1]) + structure_loss(preds[1][:1], gts[:1]) + structure_loss(preds[2][:1], gts[:1])
            train_loss_final = loss_final = structure_loss(preds[3][:1], gts[:1])
            train_loss = train_loss_init + train_loss_final

            loss.backward()

            clip_gradient(optimizer, opt.clip) #hjc 避免权重参数更新的过于迅速设定合适权重更新区间，当sumsq_diff（当前权重平方和）>clip_gradient时
                                                # 添加缩放因子scale_factor = clip_gradient / sumsq_diff
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data
            train_loss_all += train_loss.data
            # precision，recall计算，随后计算指标IoU,F1_score 训练集
            # res = F.upsample(preds[3], size=gtsp.shape, mode='bilinear', align_corners=False)
            res = preds[3][0].sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)  #hjc 归一化
            resn = 1-res
            gtsn = 1-gtsp
            tn1 = resn * gtsn
            tn1[tn1 < 0.2] = 0
            tn = sum(map(sum, tn1))
            tp1 = res * gtsp
            tp1[tp1 < 0.2] = 0
            tp = sum(map(sum, tp1))
            precision = tp / (sum(map(sum, res)) + 1e-8)  # 查准率
            recall = tp / (sum(map(sum, gtsp)) + 1e-8)  # 查全率
            allnumber = resn*0+1
            acc_sum += (tp + tn) / (sum(map(sum, allnumber)) + 1e-8)
            F1_sum += 2 * (precision * recall) / (precision + recall + 1e-8)
            IoU_sum += precision * recall / (precision + recall - precision * recall + 1e-8)

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} '
                    'Loss2: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.data, 'Loss_final': loss_final.data,
                                    'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                # TensorboardX-Outputs
                res = preds[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_init', torch.tensor(res), step, dataformats='HW')
                res = preds[3][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')
        acc = acc_sum / epoch_step
        F1 = F1_sum / epoch_step
        IoU = IoU_sum / epoch_step
        writer.add_scalars('F1-epoch', {'train': torch.tensor(F1)}, global_step=epoch)
        writer.add_scalars('IoU-epoch', {'train': torch.tensor(IoU)}, global_step=epoch)
        writer.add_scalars('acc-epoch', {'train': torch.tensor(acc)}, global_step=epoch)
        train_loss_all /= epoch_step
        loss_all /=epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalars('Loss-epoch', {'train': train_loss_all, 'train_batch': loss_all}, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise



def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()#hjc 节点False，禁止反向传播更新参数，只进行验证
    with torch.no_grad():
        loss_all_val = 0
        mae_sum = 0
        F1_sum = 0
        acc_sum = 0
        IoU_sum = 0
        for i in range(test_loader.size):
            image, gt, name, image_for_post, valgts = test_loader.load_data()
            # gts = gts.cuda()
            gts = valgts.cuda()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res = model(image)
            loss_init = structure_loss(res[0], gts) + structure_loss(res[1], gts) + structure_loss(res[2], gts) #hjc 先验损失
            loss_final = structure_loss(res[3], gts) #hjc 候选损失

            loss = loss_init + loss_final
            loss_all_val += loss.data

            res = F.upsample(res[3], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)  #hjc 归一化
            #precision，recall计算，随后计算指标IoU,F1_score 验证集
            resn = 1-res
            gtn = 1-gt
            tn1 = resn * gtn
            tn1[tn1 < 0.2] = 0
            tn = sum(map(sum, tn1))
            tp1 = res * gt
            tp1[tp1 < 0.2] = 0
            tp = sum(map(sum, tp1))
            precision = tp / (sum(map(sum, res)) + 1e-8) #查准率
            recall = tp / (sum(map(sum, gt)) + 1e-8)  #查全率
            allnumber = resn*0+1
            acc_sum += (tp + tn) / (sum(map(sum, allnumber)) + 1e-8)
            F1_sum += 2*(precision*recall) / (precision + recall + 1e-8)
            IoU_sum += precision*recall / (precision + recall - precision*recall + 1e-8)
            #MAE计算
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        loss_all_val /= test_loader.size #hjc val_loss计算
        mae = mae_sum / test_loader.size
        acc = acc_sum / test_loader.size
        F1 = F1_sum / test_loader.size
        IoU = IoU_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        writer.add_scalars('F1-epoch', {'val': torch.tensor(F1)}, global_step=epoch)
        writer.add_scalars('acc-epoch', {'val': torch.tensor(acc)}, global_step=epoch)
        writer.add_scalars('IoU-epoch', {'val': torch.tensor(IoU)}, global_step=epoch)
        writer.add_scalars('Loss-epoch',  {'val': loss_all_val}, global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        # testparam = 0.1 * F1 + 0.1 * IoU + 0.8 * (3.5 - mae*100)
        if epoch == 1:
            # best_mae = testparam
            best_mae = mae
        else:
            # if testparam > best_mae:
            #     best_mae = testparam
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')#训练轮次120
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')#学习率
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size')#一次性训练图像数量12
    parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')#图像处理后训练的尺寸768
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')#权重更新阈值，大小不能大于0.5
    parser.add_argument('--decay_rate', type=float, default=0.01, help='decay rate of learning rate')#学习衰减率(指数级衰减)0.1
    parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')#每50轮次衰减一次
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='./dataset_3k/camo_adv/train/',  #train_val/train/
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./dataset_3k/camo_adv/val/',  #TrainDataset/train_val/val/
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./snapshot/SINet_V2/adv_camo512',#取第二折为验证集
                        help='the path to save model and log')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    cudnn.benchmark = True

    # build the model
    model = Network(channel=32).cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'tra_1-conv_camo_0.1_img/',
                              gt_root=opt.train_root + 'camo_0.1_mask/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=0)
    val_loader = test_dataset(image_root=opt.val_root + 'Image1/',
                              gt_root=opt.val_root + 'masks1/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + '/log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path +'/camo512')#hjc 可视化工具对应文件夹
    best_mae = 1
    best_epoch = 0
    # res_img = torch.randn(1, 3, 352, 352)#可视化模型随机输入
    # res_img = res_img.cuda()
    # writer.add_graph(model, res_img)
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)
