from PIL import Image
import os
import random

Imgs_dir = "C:/Users/Administrator/Desktop/AI对抗资料/datagraph/datagraph(1)/Demo/SInet_train/EISegfile/gt/GT1000"
Imgsave_dir = "C:/Users/Administrator/Desktop/AI对抗资料/datagraph/datagraph(1)/Demo/SInet_train/EISegfile/gt"
i = 0
filelist = [f for f in os.listdir(Imgs_dir) if (f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".gif") or f.endswith(".png") or f.endswith(".jfif"))]
for f1 in filelist:
    i += 1
    foo = Image.open(Imgs_dir + "/" + f1)
    imgsize = foo.size
    # print(f1, imgsize, i)  # 打印的是当前的图片分辨率
    # if imgsize[0] < 1024 or imgsize[1] < 768:
    #     foo = foo.resize((1024, 768), Image.ANTIALIAS)  # 40 ，20 是要调整的大小
    # foo.save(Imgsave_dir + "/" + f1.replace(".jpg" or ".jpeg" or ".png"), quality=95)
    j = random.randint(0, 3)
    if i % 4 == j:
        print("imgs:")
        print(f1, imgsize, j)  # 打印的是当前的图片分辨率
        # if imgsize[0] < 1024 or imgsize[1] < 768:
        #     foo = foo.resize((1024, 768), Image.ANTIALIAS)  # 40 ，20 是要调整的大小

        foo.save(Imgsave_dir + "/" + f1.replace(".jpg" or ".jpeg" or ".png"), quality=95)
    else:
        # foo.save(Global_conf_dir_new + "/" + "p1.jpg", quality=60)
        continue
