# Li2017Universal

This is a PyTorch implementation for the paper "[Universal Style Transfer via Feature Transforms](https://arxiv.org/pdf/1705.08086)", which comes from the repository [PytorchWCT](https://github.com/sunshineatnoon/PytorchWCT).

## Test (Ultra-high Resolution Style Transfer)

Use `--content` and `--style` to provide the respective path to the content and style image.

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <content_path> \
                                             --style <style_path> \
                                             --URST
```

For example:

```shell
CUDA_VISIBLE_DEVICES=0 python test.py --content ../examples/content/pexels-andrea-piacquadio-3830880.jpg \
                                      --style ../examples/style/line2.png \
                                      --padding 64
                                      --URST
```

Some options:

* `--patch_size`: The maximum size of each patch. The default setting is 1000.
* `--style_size`: The size of the style image. The default setting is 1024.
* `--thumb_size`: The size of the thumbnail image. The default setting is 1024.
* `--URST`: Use our URST framework to process ultra-high resolution images.
* `--alpha`: Adjust the degree of stylization. It should be a value between 0.0 and 1.0 (default).

For more details please refer to [PytorchWCT](https://github.com/sunshineatnoon/PytorchWCT).