# Collaborative-Distillation
PyTorch code for CVPR-20 poster paper "[Collaborative Distillation for Ultra-Resolution Universal Style Transfer](https://arxiv.org/abs/2003.08436)", where Wang et al. propose a new knowledge distillation method to reduce VGG-19 filters, realizing the ultra-resolution universal style transfer on a single 12GB GPU. They focus on model compression instead of new stylization schemes. For stylization, their method builds upon [WCT](https://papers.nips.cc/paper/6642-universal-style-transfer-via-feature-transforms.pdf).

## Environment
- Python 3.6
- PyTorch 1.1+
- TorchVision
- Pillow
- tqdm
- torchfile

## Test

Use `--content` and `--style` to provide the respective path to the content and style image.

```shell
cd PytorchWCT
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