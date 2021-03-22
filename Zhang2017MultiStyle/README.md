# Zhang2017MultiStyle

This is a PyTorch implementation for the paper "[Multi-style Generative Network for Real-time Transfer](https://arxiv.org/abs/1703.06953)", which comes from the repository [PyTorch-Multi-Style-Transfer](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer).

<img src ="images/1.jpg" width="260px" /> <img src ="images/2.jpg" width="260px" />
<img src ="images/3.jpg" width="260px" />
<img src ="images/4.jpg" width="260px" />
<img src ="images/5.jpg" width="260px" />
<img src ="images/6.jpg" width="260px" />
<img src ="images/7.jpg" width="260px" />
<img src ="images/8.jpg" width="260px" />
<img src ="images/9.jpg" width="260px" />



## Test (Ultra-high Resolution Style Transfer)

Use `--content` and `--style to provide the respective path to the content image and model.

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <content_path> \
                                             --style <style_path> \
                                             --URST
```

For example:

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content ../examples/content/pexels-andrea-piacquadio-3830880.jpg \
                                             --style ../examples/style/mosaic.png \
                                             --URST
```

Some options:

* `--patch_size`: The maximum size of each patch. The default setting is 1000.
* `--style_size`: The size of the style image. The default setting is 1024.
* `--thumb_size`: The size of the thumbnail image. The default setting is 1024.
* `--URST`: Use our URST framework to process ultra-high resolution images.

## Train

Please refer to [PyTorch-Multi-Style-Transfer](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer).