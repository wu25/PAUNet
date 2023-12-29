# PA-Unet: **Phase-Aware Unet for Image Denoising**

<hr />

> **Abstract:** In recent years, the rapid development of neural networks has made great contributions to the field of image denoising. However, preserving image details during denoising remains a significant challenge. To address this problem, we propose a innovative method called Phase-Aware Unet (PA-Unet) that extracts relevant features between pixels while preserving the contextual information of the image. Our approach comprises two key components. First, we introduce a novelty Phase-Aware Attention (PAA) that effectively captures underlying relationships and dependencies between pixels. Secondly, we adopt a Residual Phase-Aware Attention Module (RPAA) to combine PAA with Simplified Channel Attention (SCA) to further enhance the extraction of relevant features. We evaluate the performance of our method on both synthetic and real datasets, and the experimental results show that our approach achieves competitive performance compared to state-of-the-art methods while effectively retaining more image texture information.

## Installation
The model is built in PyTorch 1.1.8 and tested on Ubuntu 16.04 environment (Python3.8, CUDA11.8, cuDNN7.5).

For installing, follow these intructions
```
conda create -n pytorch1 python=3.8
conda activate pytorch1
conda install pytorch=1.1.8 torchvision=0.15 cudatoolkit=11.1 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

## Training

#### Gaussian Denoising 

1. Download the DIV2K and CBSD68 dataset
2. Generate image patches

```
generate_patches_div2k+div2k+cbsd.py  
```

3. Download validation images of SIDD and place them in `../SIDD_patches/val`

4. Train your model with default arguments by running

```
python train.py -opt options/train/color.yml
```

### Real Denoising

1. Download the SIDD dataset
2. Generate image patches

```
python generate_patches_SIDD.py  
```

3. Download train images of SIDD and place them in `../SIDD_patches/val`

4. Train your model with default arguments by running

```
python train.py -opt options/train/color.yml
```

## Evaluation

Real Denoising

1. Download the SIDD-Medium dataset 
2. Generate image patches
```
python generate_patches_SIDD.py
```
3. Download validation images of SIDD and place them in `../SIDD_patches/val`

   4.Train your model with default arguments by running

```
python train.py -opt options/train/real.yml
```

## Evaluation
You can download, at once, the complete repository of PA-Unet (including pre-trained models, datasets, results, etc) from this Google Drive  [link](https://drive.google.com/drive/folders/1C2XCufoxxckQ29EkxERFPxL8R3Kx68ZG?usp=sharing), or evaluate individual tasks with the following instructions:

### Gaussian Denoising 
- Download the [model](https://drive.google.com/file/d/13PGkg3yaFQCvz6ytN99Heh_yyvfxRCdG/view?usp=sharing) and place it in ./pretrained_models/denoising/

- Download sRGB images of  Set12, CBSD68, BSD68, McMaster, Kodak, Urban100 and place them in ./datasets

- run

  python test.py -opt options/test/color.yml

### Real Denoising

- Download the [model](https://drive.google.com/file/d/13PGkg3yaFQCvz6ytN99Heh_yyvfxRCdG/view?usp=sharing) and place it in ./pretrained_models/denoising/

- Download sRGB [images](https://drive.google.com/drive/folders/1j5ESMU0HJGD-wU6qbEdnt569z7sM3479?usp=sharing) of SIDD and PolyU and place them in ./datasets/
- Run
```
python test_denoising_sidd.py --save_images
```
