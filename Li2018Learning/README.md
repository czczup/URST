## Learning Linear Transformations for Fast Image and Video Style Transfer
**[[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Learning_Linear_Transformations_for_Fast_Image_and_Video_Style_Transfer_CVPR_2019_paper.pdf)** **[[Project Page]](https://sites.google.com/view/linear-style-transfer-cvpr19/)**

<img src="doc/images/chicago_paste.png" height="149" hspace="5"><img src="doc/images/photo_content.png" height="150" hspace="5"><img src="doc/images/content.gif" height="150" hspace="5">
<img src="doc/images/chicago_27.png" height="150" hspace="5"><img src="doc/images/in5_result.png" height="150" hspace="5"><img src="doc/images/test.gif" height="150" hspace="5">

## Requirements

- Python 3.6
- PyTorch 1.1+
- TorchVision
- Pillow
- tqdm

## Style Transfer
Use `--content` and `--style` to provide the respective path to the content and style image.

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content ../examples/content/pexels-andrea-piacquadio-3830880.jpg \
                                             --style ../examples/style/line2.png \
                                             --decoder models/dec_r41_stroke_perceptual_loss_1.pth
                                             --URST
```

Some options:

* `--patch_size`: The maximum size of each patch. The default setting is 1000.
* `--style_size`: The size of the style image. The default setting is 1024.
* `--thumb_size`: The size of the thumbnail image. The default setting is 1024.
* `--decoder`: Path to the decoder. The default decoder is the original model trained without our stroke perceptual loss. 
* `--URST`: Use our URST framework to process ultra-high resolution images.

## Model Training
### Data Preparation
- MSCOCO

  ```shell
  wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
  ```
- WikiArt
  - Either manually download from [kaggle](https://www.kaggle.com/c/painter-by-numbers).
  - Or install [kaggle-cli](https://github.com/floydwch/kaggle-cli) and download by running:
  ```shell
  kg download -u <username> -p <password> -c painter-by-numbers -f train.zip
  ```

### Training
To train a model with our proposed stroke perceptual loss:
```shell
python trainv2.py --contentPath PATH_TO_MSCOCO \
                  --stylePath PATH_TO_WikiArt \
                  --outf checkpoints/
```
