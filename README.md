# Towards Ultra-Resolution Neural Style Transfer via Thumbnail Instance Normalization (AAAI'2022)

Official PyTorch implementation for our URST (Ultra-Resolution Style Transfer) framework.

URST is a versatile framework for ultra-high resolution style transfer under limited memory resources, which can be easily plugged in most existing neural style transfer methods.

With the growth of the input resolution, the memory cost of our URST hardly increases. Theoretically, it supports style transfer of arbitrary high-resolution images. 

<center><img src="assets/ultra_high_result.jpg" width="1000" hspace="10"></center>
<p align="center">
  One ultra-high resolution stylized result of 12000 x 8000 pixels (i.e., 96 megapixels).
</p>

This repository is developed based on six representative style transfer methods, which are [Johnson et al.](https://arxiv.org/abs/1603.08155), [MSG-Net](https://arxiv.org/abs/1703.06953),  [AdaIN](https://arxiv.org/abs/1703.06868), [WCT](https://arxiv.org/abs/1705.08086), [LinearWCT](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Learning_Linear_Transformations_for_Fast_Image_and_Video_Style_Transfer_CVPR_2019_paper.html), and [Wang et al. (Collaborative Distillation)](https://arxiv.org/abs/2003.08436).

For details see [Towards Ultra-Resolution Neural Style Transfer via Thumbnail Instance Normalization](https://arxiv.org/abs/2103.11784).

If you use this code for a paper please cite:

```
@inproceedings{chen2022towards,
  title={Towards Ultra-Resolution Neural Style Transfer via Thumbnail Instance Normalization},
  author={Chen, Zhe and Wang, Wenhai and Xie, Enze and Lu, Tong and Luo, Ping},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

## Environment

- python3.6, pillow, tqdm, torchfile, pytorch1.1+ (for inference)

  ```shell
  pip install pillow
  pip install tqdm
  pip install torchfile
  conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
  ```

- tensorboardX (for training)

  ```shell
  pip install tensorboardX
  ```

Then, clone the repository locally:

```shell
git clone https://github.com/czczup/URST.git
```

## Test (Ultra-high Resolution Style Transfer)

**Step 1: Prepare images**

- Content images and style images are placed in `examples/`.
- Since the ultra-high resolution images are quite large, we not place them in this repository. Please download them from this [google drive](https://drive.google.com/file/d/1TFWHb-PQ57qYCaNm2lKUxtxouq3PJO0i/view?usp=sharing).
- All content images used in this repository are collected from [pexels.com](https://www.pexels.com/). 

**Step 2: Prepare models**

- Download models from this [google drive](https://drive.google.com/file/d/1f-G5RsYMUqJlTgNBV7_MfRP_JIb6Oz2b/view?usp=sharing). Unzip and merge them into this repository.

**Step 3: Stylization**

First, choose a specific style transfer method and enter the directory. 

Then, please run the corresponding script. The stylized results will be saved in `output/`.

- For Johnson et al., we use the PyTorch implementation [Fast-Neural-Style-Transfer](https://github.com/eriklindernoren/Fast-Neural-Style-Transfer).

  ```shell
  cd Johnson2016Perceptual/
  CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <content_path> --model <model_path> --URST
  ```

- For MSG-Net, we use the official PyTorch implementation [PyTorch-Multi-Style-Transfer](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer).

  ```shell
  cd Zhang2017MultiStyle/
  CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <content_path> --style <style_path> --URST
  ```

- For AdaIN, we use the PyTorch implementation [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN).

  ```shell
  cd Huang2017AdaIN/
  CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <content_path> --style <style_path> --URST
  ```

- For WCT, we use the PyTorch implementation [PytorchWCT](https://github.com/sunshineatnoon/PytorchWCT).

  ```shell
  cd Li2017Universal/
  CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <content_path> --style <style_path> --URST
  ```

- For LinearWCT, we use the official PyTorch implementation [LinearStyleTransfer](https://github.com/sunshineatnoon/LinearStyleTransfer).

  ```shell
  cd Li2018Learning/
  CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <content_path> --style <style_path> --URST
  ```

- For Wang et al. (Collaborative Distillation), we use the official PyTorch implementation [Collaborative-Distillation](https://github.com/MingSun-Tse/Collaborative-Distillation).

  ```shell
  cd Wang2020Collaborative/PytorchWCT/
  CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content <content_path> --style <style_path> --URST
  ```

Optional options:

- `--patch_size`: The maximum size of each patch. The default setting is 1000.
- `--style_size`: The size of the style image. The default setting is 1024.
- `--thumb_size`: The size of the thumbnail image. The default setting is 1024.
- `--URST`: Use our URST framework to process ultra-high resolution images.

## Train (Enlarge the Stroke Size)

**Step 1: Prepare datasets**

Download the [MS-COCO 2014 dataset](http://cocodataset.org/#download) and [WikiArt dataset](https://www.kaggle.com/c/painter-by-numbers).

- MS-COCO

  ```shell
  wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
  ```

- WikiArt

  - Either manually download from [kaggle](https://www.kaggle.com/c/painter-by-numbers).
  - Or install [kaggle-cli](https://github.com/floydwch/kaggle-cli) and download by running:

  ```shell
  kg download -u <username> -p <password> -c painter-by-numbers -f train.zip
  ```

**Step 2: Prepare models**

As same as the Step 2 in the test phase.

**Step 3: Train the decoder with our stroke perceptual loss**

- For AdaIN:

  ```shell
  cd Huang2017AdaIN/
  CUDA_VISIBLE_DEVICES=<gpu_id> python trainv2.py --content_dir <coco_path> --style_dir <wikiart_path>
  ```

- For LinearWCT:

  ```shell
  cd Li2018Learning/
  CUDA_VISIBLE_DEVICES=<gpu_id> python trainv2.py --contentPath <coco_path> --stylePath <wikiart_path>
  ```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/czczup/URST/blob/main/LICENSE.md) file.
