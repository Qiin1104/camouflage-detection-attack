import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.pvt import IdeNet
from utils.dataloader import My_test_dataset
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size default 352')
parser.add_argument('--pth_path', type=str, default='./checkpoints/IdeNetnew/Net_epoch_best.pth')
opt = parser.parse_args()
writer = SummaryWriter('./snapshot/xr')
for _data_name in ['F_ADV']:
    data_path = './dataset2/{}/xr'.format(_data_name)
    save_path = './result/F_ADV/xr'
    model = IdeNet(train_mode=False)
    print(torch.cuda.is_available())
    model.load_state_dict(torch.load(opt.pth_path, map_location='cuda:0'))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/IFGSM/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    print('root',image_root,gt_root)
    test_loader = My_test_dataset(image_root, gt_root, opt.testsize)
    print('****',test_loader.size)
    F1_sum = 0
    acc_sum = 0
    IoU_sum = 0
    NUMBER = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        print('***name',name)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        P = model(image)
        P[-1] = (torch.tanh(P[-1]) + 1.0) / 2.0
        
        res = F.upsample(P[-1], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # 测试集指标
        resn = 1 - res
        gtn = 1 - gt
        tn = sum(map(sum, resn * gtn))
        tp = sum(map(sum, res * gt))
        precision = tp / (sum(map(sum, res)) + 1e-8)  # 查准率
        recall = tp / (sum(map(sum, gt)) + 1e-8)  # 查全率
        allnumber = resn * 0 + 1
        acc_val = (tp + tn) / (sum(map(sum, allnumber)) + 1e-8)
        f1_val = 2 * (precision * recall) / (precision + recall + 1e-8)
        iou_val = precision * recall / (precision + recall - precision * recall + 1e-8)
        writer.add_scalar('accval-step', torch.tensor(acc_val), global_step=i)
        writer.add_scalar('F1val-step', torch.tensor(f1_val), global_step=i)
        writer.add_scalar('IoUval-step', torch.tensor(iou_val), global_step=i)
        # 求和
        acc_sum += acc_val
        F1_sum += f1_val
        IoU_sum += iou_val

        writer.add_image('pre_val', torch.tensor(res), i, dataformats='HW')
        writer.add_image('gt_val', torch.tensor(gt), i, dataformats='HW')
        grid_image = make_grid(image[0].clone().cpu().data, 1, normalize=True)
        writer.add_image('RGB_val', grid_image, i)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+name,res*255)
    acc = acc_sum / test_loader.size
    F1 = F1_sum / test_loader.size
    IoU = IoU_sum / test_loader.size
    writer.add_scalars('F1-epoch', {'test': torch.tensor(F1)}, global_step=NUMBER)
    writer.add_scalars('acc-epoch', {'test': torch.tensor(acc)}, global_step=NUMBER)
    writer.add_scalars('IoU-epoch', {'test': torch.tensor(IoU)}, global_step=NUMBER)
