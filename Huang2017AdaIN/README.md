# pytorch-AdaIN

This is a pytorch implementation of a paper, Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [Huang+, ICCV2017].

![Results](results.png)

## Requirements
- Python 3.6
- PyTorch 1.1+
- TorchVision
- Pillow
- tqdm

(optional, for training)

- TensorboardX

## Usage

### Test
Use `--content` and `--style` to provide the respective path to the content and style image.
```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content ../examples/content/pexels-andrea-piacquadio-3830880.jpg \
                                             --style ../examples/style/line2.png \
                                             --decoder models/decoder_stroke_perceptual_loss_1.pth.tar \
                                             --URST
```

Some options:
* `--patch_size`: The maximum size of each patch. The default setting is 1000.
* `--style_size`: The size of the style image. The default setting is 1024.
* `--thumb_size`: The size of the thumbnail image. The default setting is 1024.
* `--decoder`: Path to the decoder. The default decoder is the original model trained without our stroke perceptual loss. 
* `--URST`: Use our URST framework to process ultra-high resolution images.
* `--alpha`: Adjust the degree of stylization. It should be a value between 0.0 and 1.0 (default).
* `--preserve_color`: Preserve the color of the content image.


### Train
Use `--content_dir` and `--style_dir` to provide the respective directory to the content and style images.
```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python trainv2.py --content_dir <content_dir> --style_dir <style_dir>
```

For more details and parameters, please refer to --help option.

## References
- [1]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.
- [2]: [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
