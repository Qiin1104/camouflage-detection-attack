import os

image_root = "H:/hjc20220923/YOLOV7/yolov7-main/coco/edge/all/edge/"
gt_root = "H:/hjc20220923/YOLOV7/yolov7-main/coco/labels/all/"
images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.txt')  ]#f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')
size = len(images)
i = 0
while i < size:
    a = images[i].split('/')[-1]
    b = gts[i].split('/')[-1]
    assert len(a) == len(b) or len(a) - 1 == len(b)
    print(a + '\n' + b)
    i += 1
    print(i)
