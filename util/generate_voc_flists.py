import os
import sys

path = "/unsullied/sharefs/_research_detection/GeneralDetection/VOC/VOCdevkit/VOC2012"
image_path = "/unsullied/sharefs/_research_detection/GeneralDetection/VOC/VOCdevkit/VOC2012/JPEGImages"
annotation_path = "/unsullied/sharefs/_research_detection/GeneralDetection/VOC/VOCdevkit/VOC2012/Annotations"
cls_name  = sys.argv[1]
train_file_name = os.path.join(path, "ImageSets/Main/{}_train.txt".format(cls_name))
val_file_name = os.path.join(path, "ImageSets/Main/{}_val.txt".format(cls_name))
#test_file_name = os.path.join(path, "ImageSets/Main/{}_val.txt".format(cls_name))
all_cls = ['bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'motorbike', 'sheep', 'sofa', 'tvmonitor']
cls_dict = {term:i for i, term in enumerate(all_cls)}

def generate_flist(file_name, cls_name, mode):
    with open(file_name, 'r') as f:
        fterms = f.read().splitlines()
    with open("/unsullied/sharefs/linhangyu/Inpainting/Data/VOCData/voc_all_{}_flist.txt".format(mode), 'a') as f1, \
        open("/unsullied/sharefs/linhangyu/Inpainting/Data/VOCData/voc_all_{}_bbox_flist.txt".format(mode), "a") as f2:
        for term in fterms:
            terms = term.strip().split()
            file = terms[0]
            e = terms[1]
            print(terms, file, e)
            if e == '1':
                # f1.write(os.path.join(image_path, file+'.jpg') + '\t{}\n'.format(cls_dict[cls_name]))
                # f2.write(os.path.join(annotation_path, file+'.xml') + '\t{}\n'.format(cls_dict[cls_name]))
                f1.write(os.path.join(image_path, file+'.jpg') + '\n')
                f2.write(os.path.join(annotation_path, file+'.xml') + '\n')
for cls_name in all_cls:
    train_file_name = os.path.join(path, "ImageSets/Main/{}_train.txt".format(cls_name))
    val_file_name = os.path.join(path, "ImageSets/Main/{}_val.txt".format(cls_name))
    generate_flist(train_file_name, cls_name, 'train')
    generate_flist(val_file_name, cls_name, 'val')
