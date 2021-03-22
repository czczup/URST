# Wang2020Collaborative
This is a PyTorch implementation for the paper "[Collaborative Distillation for Ultra-Resolution Universal Style Transfer](https://arxiv.org/abs/2003.08436)", which comes from the repository [Collaborative-Distillation](https://github.com/MingSun-Tse/Collaborative-Distillation).

## Test (Ultra-high Resolution Style Transfer)

Use `--content` and `--style` to provide the respective path to the content and style image.

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <content_path> \
                                             --style <style_path> \
                                             --URST
```

For example:

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content ../../examples/content/pexels-andrea-piacquadio-3830880.jpg \
                                             --style ../../examples/style/mosaic.png \
                                             --URST
```

Some options:

* `--patch_size`: The maximum size of each patch. The default setting is 1000.
* `--style_size`: The size of the style image. The default setting is 1024.
* `--thumb_size`: The size of the thumbnail image. The default setting is 1024.
* `--URST`: Use our URST framework to process ultra-high resolution images.
* `--alpha`: Adjust the degree of stylization. It should be a value between 0.0 and 1.0 (default).

## Train

Please refer to [Collaborative-Distillation](https://github.com/MingSun-Tse/Collaborative-Distillation).