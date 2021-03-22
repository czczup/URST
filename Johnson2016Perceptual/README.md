# Johnson2016Perceptual

This is a PyTorch implementation for the paper "[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)", which comes from the repository [Fast-Neural-Style-Transfer](https://github.com/eriklindernoren/Fast-Neural-Style-Transfer).

<p align="center">
    <img src="assets/zurich.jpg" width="900"\>
</p>

## Test (Ultra-high Resolution Style Transfer)

Use `--content` and `--style` to provide the respective path to the content and style image.

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <content_path> \
                                             --model <model_path> \
                                             --URST
```

For example:

```shell
CUDA_VISIBLE_DEVICES=0 python test.py --content ../examples/content/pexels-paulo-marcelo-martins-2412603.jpg \
                                      --model models/mosaic_1024_10000.pth \
                                      --URST
```

Some options:

* `--patch_size`: The maximum size of each patch. The default setting is 1000.
* `--thumb_size`: The size of the thumbnail image. The default setting is 1024.
* `--URST`: Use our URST framework to process ultra-high resolution images.

## Train

Please refer to [Fast-Neural-Style-Transfer](https://github.com/eriklindernoren/Fast-Neural-Style-Transfer).