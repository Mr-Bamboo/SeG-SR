# SeG-SR (IEEE TGRS 2025)
### [**Paper**](https://ieeexplore.ieee.org/document/11175207) | [**Arxiv**](https://arxiv.org/abs/2505.23010)

PyTorch codes for "[SeG-SR: Integrating Semantic Knowledge Into Remote Sensing Image Super-Resolution via Vision-Language Model](https://ieeexplore.ieee.org/document/11175207)", **IEEE Transactions on Geoscience and Remote Sensing(TGRS)**, 2025.

## Abstract
> High-resolution (HR) remote sensing imagery plays a vital role in a wide range of applications, including urban planning and environmental monitoring. However, due to limitations in sensors and data transmission links, the images acquired in practice often suffer from resolution degradation. Remote sensing image super-resolution (RSISR) aims to reconstruct HR images from low-resolution (LR) inputs, providing a cost-effective and efficient alternative to direct HR image acquisition. Existing RSISR methods primarily focus on low-level characteristics in pixel space, while neglecting the high-level understanding of remote sensing scenes. This may lead to semantically inconsistent artifacts in the reconstructed results. Motivated by this observation, our work aims to explore the role of high-level semantic knowledge in improving RSISR performance. We propose a semantic-guided super-resolution (SR) framework, semantic-guided super-resolution (SeG-SR), which leverages vision-language models (VLMs) to extract semantic knowledge from input images and uses it to guide the SR process. Specifically, we first design a semantic feature extraction module (SFEM) that utilizes a pretrained VLM to extract semantic knowledge from remote sensing images. Next, we propose a semantic localization module (SLM), which derives a series of semantic guidance from the extracted semantic knowledge. Finally, we develop a learnable modulation module (LMM) that uses semantic guidance to modulate the features extracted by the SR network, effectively incorporating high-level scene understanding into the SR pipeline. We validate the effectiveness and generalizability of SeG-SR through extensive experiments: SeG-SR achieves state-of-the-art performance on three datasets, and consistently improves performance across various SR architectures. Notably, for the x4 SR task on the UCMerced dataset, it attained a PSNR of 29.3042 dB and an SSIM of 0.7961.
## Pipeline  
 ![image](/figs/SeG-SR.png)
 
## Install
```
git clone https://github.com/Mr-Bamboo/SeG-SR.git
python setup.py develop
```

## Environment
 > * CUDA 11.8
 > * Python >=3.7.0
 > * PyTorch >= 1.9.0
 > * basicsr >= 1.4.2


## Usage

### Train
- Single GPU For Training:
```
python segsr/train.py -opt options/train/train_SEGSR_SRx4_UCMerced.yml --auto_resume
```
- The command for multi-GPU training is same as [HAT](https://github.com/XPixelGroup/HAT).

### Test
```
python segsr/test.py -opt options/test/test_SEGSR_SRx4_UCMerced.yml
```
The weight file (pth) for UCMerced has been released at the link below： [Google Drive](https://drive.google.com/drive/folders/1p9Q_3_9doJieEmg5YC1gazxDYsHO2xtg?usp=sharing) 

However, please note that we do not recommend using this weight file directly for performance reporting, since your dataset splits may differ from those used in our work. For performance reporting, we strongly suggest training the model yourself using the provided training scripts and then evaluating with the weights obtained from your own training process.

## Acknowledgments
Our SeG-SR mainly borrows from [HAT](https://github.com/XPixelGroup/HAT), [CLIP](https://github.com/openai/CLIP/tree/main/clip) and [TTST](https://github.com/XY-boy/TTST). Thanks for these excellent open-source works!

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: chenbowen@buaa.edu.cn

## Citation
If you find our work helpful in your research, please consider citing it. Your support is greatly appreciated! 😊

```
@ARTICLE{11175207,
  author={Chen, Bowen and Chen, Keyan and Yang, Mohan and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SeG-SR: Integrating Semantic Knowledge Into Remote Sensing Image Super-Resolution via Vision-Language Model}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  keywords={Semantics;Remote sensing;Image reconstruction;Superresolution;Feature extraction;Transformers;Modulation;Diffusion models;Data mining;Computer architecture;Remote sensing;semantic guidance;super-resolution (SR);vision-language model (VLM)},
  doi={10.1109/TGRS.2025.3612420}}
```
