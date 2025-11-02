from sklearn.model_selection import KFold
from PIL import Image
import os
K = 5
Imgs_dir = "H:/hjc20220923/SInetV2/SINet-V2-main/dataset_3k/race/train/image"
GT_dir = "H:/hjc20220923/SInetV2/SINet-V2-main/dataset_3k/race/train/mask"
Imgs_val = "H:/hjc20220923/SInetV2/SINet-V2-main/dataset_3k/race/5/val/Imgs"
Imgs_train = "H:/hjc20220923/SInetV2/SINet-V2-main/dataset_3k/race/5/train/Imgs"
GT_val = "H:/hjc20220923/SInetV2/SINet-V2-main/dataset_3k/race/5/val/GT"
GT_train = "H:/hjc20220923/SInetV2/SINet-V2-main/dataset_3k/race/5/train/GT"
filelist1 = [f for f in os.listdir(Imgs_dir) if (f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".gif") or f.endswith(".png") or f.endswith(".jfif"))]
filelist2 = [f for f in os.listdir(GT_dir) if (f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".gif") or f.endswith(".png") or f.endswith(".jfif"))]
kf = KFold(n_splits=K, shuffle=True, random_state=8)  # 初始化KFold
j = -1
for train_index, test_index in kf.split(filelist1):  # 调用split方法切分数据
    print('train_index:%s , test_index: %s ' %(train_index,test_index))
    j += 1
    if j == 2 and j < K:
        for i in train_index:
            f1 = filelist1[i]
            foo1 = Image.open(Imgs_dir + "/" + f1)
            imgsize1 = foo1.size
            f2 = filelist2[i]
            foo2 = Image.open(GT_dir + "/" + f2)
            imgsize2 = foo2.size
            print("imgs:")
            print(f1, imgsize1,i)  # 打印的是当前的图片分辨率
            print("GT:")
            print(f2, imgsize2,i)  # 打印的是当前的图片分辨率
            foo1.save(Imgs_train + "/" + f1.replace(".jpg", ".png"), quality=95)
            foo2.save(GT_train + "/" + f1.replace("_pseudo.jpg", ".png"), quality=95)
        for i in test_index:
            f1 = filelist1[i]
            foo1 = Image.open(Imgs_dir + "/" + f1)
            imgsize1 = foo1.size
            f2 = filelist2[i]
            foo2 = Image.open(GT_dir + "/" + f2)
            imgsize2 = foo2.size
            print("imgs:")
            print(f1, imgsize1,i)  # 打印的是当前的图片分辨率
            print("GT:")
            print(f2, imgsize2,i)  # 打印的是当前的图片分辨率
            foo1.save(Imgs_val + "/" + f1.replace(".jpg", ".png"), quality=95)
            foo2.save(GT_val + "/" + f1.replace("_pseudo.jpg", ".png"), quality=95)
            # if len(f1) + 7 != len(f2) and len(f1) != len(f2) and len(f1) + 6 != len(f2):
            #     print(f2)
            #     exit()
        break
    else:
        continue


