rlaunch --cpu=4 -P1 --gpu=1 --memory=60000  -- python3 test_batch_seg.py --image /unsullied/sharefs/linhangyu/Inpainting/Data/SegmentationData/imgs --mask /unsullied/sharefs/linhangyu/Inpainting/Data/SegmentationData/masks --output test_result/imagenet_seg --checkpoint_dir model_logs/release_places2_256
# CelebA 256x256 input
