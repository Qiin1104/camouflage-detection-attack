import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio
import cv2
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import test_dataset
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)  # hjc 加权BCE更关注于难像素，参考F3net
    # 选取了周围15个像素作为周围像素,添加这个的目的是提高训练困难样本的速度
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')  # hjc with_logits可以避免数据溢出，reduce的作用主要是对于NLP
    # 其自然语言与上下的字母有关。而图片之间除非是视频，否则关联性不大，即上一张图片对下一张图片影响不大。
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))  # 权重加成

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))  # hjc dim是消除第二三维的数据，只对该batchsize中的图像进行权重更新
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)  # hjc 损失函数可以改进，如采用GIoU进行改进，因为小目标会存在重叠区域为0的情况
    return (wiou + wbce).mean()


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/race/race27/Net_epoch_best.pth')
opt = parser.parse_args()
writer = SummaryWriter('./snapshot/SINet_V2/AI_COD3KIR/graph/Doc2k7')  # hjc 可视化工具对应文件夹
for _data_name in ['dataset_3k']:  # , 'COD10K', 'CHAMELEON'
    data_path = './{}/race/test'.format(_data_name)
    save_path = './res/race/{}/'.format(opt.pth_path.split('/')[-3])
    # preimg_path = './res/race/{}/{}/'.format(opt.pth_path.split('/')[-3], 'bz')
    model = Network(imagenet_pretrained=False)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    # os.makedirs(preimg_path, exist_ok=True)
    image_root = '{}/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    loss_all_test = 0
    F1_sum = 0
    acc_sum = 0
    IoU_sum = 0

    for i in range(test_loader.size):
        image, gt, name, _, testgts = test_loader.load_data1()
        # gts = testgts.cuda()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2
        # loss_init = structure_loss(res5, gts) + structure_loss(res4, gts) + structure_loss(res3, gts)  # hjc 先验损失
        # loss_final = structure_loss(res2, gts)  # hjc 候选损失

        # loss = loss_init + loss_final
        # loss_all_test += loss.data
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # 测试集指标
        # resn = 1 - res
        # gtn = 1 - gt
        # tn = sum(map(sum, resn * gtn))
        # tp = sum(map(sum, res * gt))
        # precision = tp / (sum(map(sum, res)) + 1e-8)  # 查准率
        # recall = tp / (sum(map(sum, gt)) + 1e-8)  # 查全率
        # allnumber = resn * 0 + 1
        # acc_val = (tp + tn) / (sum(map(sum, allnumber)) + 1e-8)
        # f1_val = 2 * (precision * recall) / (precision + recall + 1e-8)
        # iou_val = precision * recall / (precision + recall - precision * recall + 1e-8)
        # writer.add_scalar('accval-step', torch.tensor(acc_val), global_step=i)
        # writer.add_scalar('F1val-step', torch.tensor(f1_val), global_step=i)
        # writer.add_scalar('IoUval-step', torch.tensor(iou_val), global_step=i)
        # # 求和
        # acc_sum += acc_val
        # F1_sum += f1_val
        # IoU_sum += iou_val

        # writer.add_image('pre_val', torch.tensor(res), i, dataformats='HW')
        # writer.add_image('gt_val', torch.tensor(gt), i, dataformats='HW')
        # grid_image = make_grid(image[0].clone().cpu().data, 1, normalize=True)
        # writer.add_image('RGB_val', grid_image, i)
        print('> {} - {}'.format(_data_name, name))
        imageio.imsave(save_path + name, res)
        # 预测图添加最小外接矩形，并对应输出中心坐标的txt文件
        # image_pre = cv2.imread(save_path + name)
        # img = cv2.cvtColor(image_pre, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(img, 20, 250, cv2.THRESH_BINARY)#范围可调
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for c in contours:
        #     # 找到边界坐标
        #     # x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
        #     # cv2.rectangle(image_pre, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        #     # # 找面积最小的矩形
        #     # rect = cv2.minAreaRect(c)
        #     # # 得到最小矩形的坐标
        #     # box = cv2.boxPoints(rect)
        #     # # 标准化坐标到整数
        #     # box = np.int0(box)
        #     # # 画出边界
        #     # cv2.drawContours(image_pre, [box], 0, (0, 0, 255), 3)
        #     # 计算最小封闭圆的中心和半径
        #     (x, y), radius = cv2.minEnclosingCircle(c)
        #     # 换成整数integer
        #     (x, y) = (int(x), int(y))
        #     radius = int(radius)
        #     # 画圆
        #     cv2.circle(image_pre, (x, y), radius, (0, 255, 0), 2)
        #     xy = "%d,%d" % (x, y)
        #     cv2.line(image_pre, (x - 2, y), (x + 2, y), (255, 0, 0))
        #     cv2.line(image_pre, (x, y - 2), (x, y + 2), (255, 0, 0))
        #     cv2.putText(image_pre, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 225), thickness=1)
        #
        # cv2.drawContours(image_pre, contours, -1, (255, 0, 0), 1)

        # cv2.imwrite(preimg_path + name, image_pre)
        # If `mics` not works in your environment, please comment it and then use CV2
        # cv2.imwrite(save_path+name,res*255)
    # loss_all_test /= test_loader.size
    # acc = acc_sum / test_loader.size
    # F1 = F1_sum / test_loader.size
    # IoU = IoU_sum / test_loader.size
    # writer.add_scalars('F1-epoch', {'test': torch.tensor(F1)}, global_step=11)
    # writer.add_scalars('acc-epoch', {'test': torch.tensor(acc)}, global_step=11)
    # writer.add_scalars('IoU-epoch', {'test': torch.tensor(IoU)}, global_step=11)
    # writer.add_scalars('Loss-epoch', {'test': loss_all_test}, global_step=11)
