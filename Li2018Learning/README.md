## Li2018Learning
This is a PyTorch implementation for the paper "[Learning Linear Transformations for Fast Image and Video Style Transfer](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Learning_Linear_Transformations_for_Fast_Image_and_Video_Style_Transfer_CVPR_2019_paper.html)", which comes from the repository [LinearStyleTransfer](https://github.com/sunshineatnoon/LinearStyleTransfer).

<img src="doc/images/chicago_paste.png" height="149" hspace="5"><img src="doc/images/photo_content.png" height="150" hspace="5"><img src="doc/images/content.gif" height="150" hspace="5">
<img src="doc/images/chicago_27.png" height="150" hspace="5"><img src="doc/images/in5_result.png" height="150" hspace="5"><img src="doc/images/test.gif" height="150" hspace="5">

## Setup
```
# python 3.6.9
pip install -r requirements.txt
```

## Models
Download model at [link](https://drive.google.com/file/d/119zpTaDvBa-otAk3DPiBanY2AmH5hW98/view?usp=share_link)

## Test (Ultra-high Resolution Style Transfer)

Use `--content` and `--style` to provide the respective path to the content and style image.

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <content_path> \
                                             --style <style_path> \
                                             --decoder <decoder_path> \
                                             --URST
```

Some options:

* `--patch_size`: The maximum size of each patch. The default setting is 1000.
* `--style_size`: The size of the style image. The default setting is 1024.
* `--thumb_size`: The size of the thumbnail image. The default setting is 1024.
* `--decoder`: Path to the decoder. The default decoder is the original model trained without our stroke perceptual loss. 
* `--URST`: Use our URST framework to process ultra-high resolution images.

## Train (Enlarge the Stroke Size)

Use `--contentPath` and `--stylePath` to provide the respective directory to the content and style images.

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python trainv2.py --contentPath <coco_path> --stylePath <wikiart_path>
```

For more details and parameters, please refer to `--help` option.