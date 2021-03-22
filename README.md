# Towards Ultra-Resolution Neural Style Transfer via Thumbnail Instance Normalization

Official PyTorch implementation for our URST (Ultra-Resolution Style Transfer) framework.

URST is a versatile framework for ultra-high resolution style transfer under limited memory resources, which can be easily plugged in most existing neural style transfer methods.

With the growth of the input resolution, the memory cost of our URST hardly increases. Theoretically, it supports style transfer of arbitrary high-resolution images. 

This repository is developed based on six representative style transfer methods, which are [Johnson et al.](https://arxiv.org/abs/1603.08155), [MSG-Net](https://arxiv.org/abs/1703.06953),  [AdaIN](https://arxiv.org/abs/1703.06868), [WCT](https://arxiv.org/abs/1705.08086), [LinearWCT](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Learning_Linear_Transformations_for_Fast_Image_and_Video_Style_Transfer_CVPR_2019_paper.html), and [Wang et al. (Collaborative Distillation)](https://arxiv.org/abs/2003.08436).

<center><img src="assets/ultra_high_result.jpg" width="1000" hspace="10"></center>
<center>One ultra-high resolution stylized result of 12000 x 8000 pixels (i.e., 96 megapixels)</center>