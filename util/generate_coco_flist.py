import os
import sys
#import pycocotools
from pycocotools.coco import COCO
import pickle as pkl
# Load cls name
cls_name  = sys.argv[1]
dataDir = '/unsullied/sharefs/linhangyu/Inpainting/Data/COCO/annotations_trainval2017'
dataType = 'train2017'
prefix = 'instances'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
coco = COCO(annFile)


catIds = coco.getCatIds(catNms=[cls_name]);
#print(catIds, sep=' ', end='n', file=sys.stdout, flush=False)
imgIds = coco.getImgIds(catIds=catIds[0] );
#imgs = coco.loadImgs(imgIds)
annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)

print(len(anns))
imgId_dict = {}
#imgIds = coco.getImgIds(imgIds = [324158])
with open("coco_{}_{}_flist.txt".format(cls_name, dataType), 'w') as f ,\
     open("coco_{}_{}_bbox_flist.txt".format(cls_name, dataType), 'w') as fb:
    for ann in anns:
        imgId = ann['image_id']
        if imgId in imgId_dict:
            continue
        else:
            imgId_dict[imgId] = 1
        img = coco.loadImgs(imgId)[0]
        print(img)
        file_path = '/unsullied/sharefs/linhangyu/Inpainting/Data/COCO/images/%s/%s/%s'%(dataType,dataType,img['file_name'])
        if not os.path.exists(file_path):
            print(file_path)
        f.write(file_path+'\n')
        bbox_path = '/unsullied/sharefs/linhangyu/Inpainting/Data/COCO/bboxes/%s/%s'%(dataType, img['file_name'].replace('jpg', 'pkl'))
        pkl.dump({"bbox":ann['bbox'], 'shape':(img['width'], img['height'])}, open(bbox_path, 'wb'))
        fb.write(bbox_path+'\n')
