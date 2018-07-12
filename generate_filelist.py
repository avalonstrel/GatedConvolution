import os
import tarfile

def norilist2list(filename, noriname, prefix):
    with open(filename, 'w') as wf:
        with open(noriname, 'r') as rf:
            for line in rf:
                terms = line.split('\t')
                wf.write(prefix+terms[2]+'\n')


def tar2list(filename, tarname, prefix):
    places_path = '/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData'
    tar = tarfile.open(tarname)
    with open(filename, 'w') as f:
        for tarinfo in tar:
            if tarinfo.isfile():
                f.write(os.path.join(prefix, tarinfo.name)+"\n")

def places2list(filename, dataset_path='/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/data_256/'):
    with open(filename, mode='w') as f:
        # a, b, c
        for root, dirs, files in os.walk(dataset_path):
            for d in dirs:
                # airfield
                for dr, dds, dfs in os.walk(os.path.join(dataset_path, d)):
                    for dd in dds:
                        for file in os.listdir(os.path.join(dataset_path,d,dd)):
                            if file[-3:] == 'jpg':
                                #print(os.path.join(dataset_path,d,dd,file))
                                f.write(os.path.join(dataset_path,d,dd,file)+"\n")

def placesval2list(filename, dataset_path='/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/val_256/'):
    with open(filename, mode='w') as f:
        # a, b, c
        for root, dirs, files in os.walk(dataset_path):
            for d in dirs:
                # airfield
                for dr, dds, dfs in os.walk(os.path.join(dataset_path, d)):
                    for dd in dds:
                        for file in os.listdir(os.path.join(dataset_path,d,dd)):
                            if file[-3:] == 'jpg':
                                #print(os.path.join(dataset_path,d,dd,file))
                                f.write(os.path.join(dataset_path,d,dd,file)+"\n")

#tar2list("/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/train_list.txt","/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/train_256_places365standard.tar")
#tar2list("/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/train_large_list.txt","/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/train_large_places365standard.tar")
#tar2list("/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/val_list.txt","/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/val_256.tar")
#tar2list("/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/test_list.txt","/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/test_256.tar")
#tar2list("/unsullied/sharefs/linhangyu/Inpainting/Data/MaskData/irrmask_flist.txt", "/unsullied/sharefs/linhangyu/Inpainting/Data/BenchmarkData/ILSVRC2012/IRRMASK/irrmask.tar", prefix="/unsullied/sharefs/linhangyu/Inpainting/Data/BenchmarkData/ILSVRC2012/IRRMASK")
norilist2list("/unsullied/sharefs/linhangyu/Inpainting/Data/MaskData/mask_flist.txt",
"/unsullied/sharefs/linhangyu/Inpainting/Data/MaskData/mask.512.train.nori.list", prefix="/unsullied/sharefs/linhangyu/Inpainting/Data/MaskData")
