rlaunch --cpu=4 -P1 --gpu=4 --memory=60000  -- python3 test_batch_seg.py --image /unsullied/sharefs/linhangyu/Inpainting/SegmentationData/imgs --mask /unsullied/sharefs/linhangyu/Inpainting/SegmentationData/masks --output test_result/imagenet_seg --checkpoint_dir model_logs/release_imagenet_256
# CelebA 256x256 input
