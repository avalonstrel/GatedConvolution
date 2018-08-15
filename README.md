
# An Reimplemented version of DeepFillv2.

**Update (Aug, 2018)**:
The main files I modify is inpaint_ops.py, inpaint_model_gc.py, and train.py. I add mask_from_fnames.py for add masks from Data(voc or coco). And I will refactor this project soon. ( To use this Project, you can refer the official version of DeepFillv1 since it is modified from the DeepFillv1)


## Run (From [DeepFillv1](https://github.com/JiahuiYu/generative_inpainting))
0. Requirements:
    * Install python3.
    * Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0).
    * Install tensorflow toolkit [neuralgym](https://github.com/JiahuiYu/neuralgym) (run `pip install git+https://github.com/JiahuiYu/neuralgym`).
1. Training:
    * Prepare training images filelist ([example](https://github.com/JiahuiYu/generative_inpainting/issues/15)).
    * Modify [inpaint.yml](/inpaint.yml) to set DATA_FLIST, LOG_DIR, IMG_SHAPES and other parameters.
    * Run `python train.py`.
2. Resume training:
    * Modify MODEL_RESTORE flag in [inpaint.yml](/inpaint.yml). E.g., MODEL_RESTORE: 20180115220926508503_places2_model.
    * Run `python train.py`.
3. Testing:
    * Run `python test.py --image examples/input.png --mask examples/mask.png --output examples/output.png --checkpoint model_logs/your_model_dir`.(I have not test)

4. Still have questions?
    * If you still have questions (e.g.: How filelist looks like? How to use multi-gpus? How to do batch testing?), please first search over closed issues. If the problem is not solved, please open a new issue.(Refer the [DeepFillv1](https://github.com/JiahuiYu/generative_inpainting))

## Results(Still Testing)

## Pretrained Model(Still Testing)

## Acknowledgments
My project acknowledge the official code DeepFillv1 and SNGAN. Especially, thanks for the authors of this amazing algorithm.
