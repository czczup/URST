# Fast Neural Style Transfer in PyTorch

<p align="center">
    <img src="assets/zurich.jpg" width="900"\>
</p>

PyTorch implementation of [Fast Neural Style Transfer](https://cs.stanford.edu/people/jcjohns/eccv16/) ([official Lua implementation](https://github.com/jcjohnson/fast-neural-style)).

## Requirements

- Python 3.6
- PyTorch 1.1+
- TorchVision
- Pillow
- tqdm

## Train

```shell
python train.py  --dataset_path <path-to-dataset> \
                 --style_image <path-to-style-image> \
                 --epochs 2 \
                 --batch_size 4 \
                 --image_size 512
                 --style_size 1024
```


## Test

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <path-to-image> \
                                             --model <path-to-checkpoint> \
                                             --URST
```

for example:

```shell
CUDA_VISIBLE_DEVICES=0 python test.py --content ../examples/content/pexels-andrea-piacquadio-3830880.jpg \
                                      --model models/mosaic_1024_10000.pth \
                                      --URST
```

Some options:

* `--patch_size`: The maximum size of each patch. The default setting is 1000.
* `--thumb_size`: The size of the thumbnail image. The default setting is 1024.
* `--URST`: Use our URST framework to process ultra-high resolution images.