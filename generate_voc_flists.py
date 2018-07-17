import os

cls_name = "horse"
path = "/unsullied/sharefs/_research_detection/GeneralDetection/VOC/VOCdevkit/VOC2012"
image_path = "/unsullied/sharefs/_research_detection/GeneralDetection/VOC/VOCdevkit/VOC2012/JPEGImages"
annotation_path = "/unsullied/sharefs/_research_detection/GeneralDetection/VOC/VOCdevkit/VOC2012/Annotations"
file_name = os.path.join(path, "ImageSets/Main/{}_trainval.txt".format(cls_name))

with open(file_name, 'r') as f:
    fterms = f.read().splitlines()

with open("/unsullied/sharefs/linhangyu/Inpainting/Data/VOCData/voc_{}_flist.txt".format(cls_name), 'w') as f1, \
    open("/unsullied/sharefs/linhangyu/Inpainting/Data/VOCData/voc_{}_bbox_flist.txt".format(cls_name), "w") as f2:
    for term in fterms:
        terms = term.strip().split()
        file = terms[0]
        e = terms[1]
        print(terms, file, e)
        if e == '1':
            f1.write(os.path.join(image_path, file+'.jpg') + '\n')
            f2.write(os.path.join(annotation_path, file+'.xml') + '\n')
