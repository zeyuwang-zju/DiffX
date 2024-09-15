# DiffX: Guide Your Layout to Cross-Modal Generative Modeling

[[Paper](https://arxiv.org/abs/2407.15488)] [[Code](https://github.com/zeyuwang-zju/DiffX)]

> **DiffX: Guide Your Layout to Cross-Modal Generative Modeling**
>
> **Authors:** Zeyu Wang^1*^, Jingyu Lin^2*^, Yifei Qian^3^, Yi Huang^4^, Shicen Tian^1^, Bosong Chai^1^, Juncan Deng^1^, Qu Yang^5^, Lan Du^2^, Cunjian Chen^2^, Yufei Guo^4†^, Kejie Huang^1†^ (^*^These authors contributed equally. ^†^Corresponding authors. )
> 1Zhejiang University, 2Monash University, 3University of Nottingham, 4Peking University, 5Wuhan University  

>**Abstract**:
> Diffusion models have made significant strides in languagedriven and layout-driven image generation. However, most diffusion models are limited to visible RGB image generation. In fact, human perception of the world is enriched by diverse viewpoints, such as chromatic contrast, thermal illumination, and depth information. In this paper, we introduce a novel diffusion model for general layoutguided cross-modal generation, called DiffX. Notably, our DiffX presents a simple yet effective cross-modal generative modeling pipeline, which conducts diffusion and denoising processes in the modality shared latent space. Moreover, we introduce the Joint-Modality Embedder (JME) to enhance the interaction between layout and text conditions by incorporating a gated attention mechanism. To facilitate the user-instructed training, we construct the cross-modal image datasets with detailed text captions by the LargeMultimodal Model (LMM) and our human-in-the-loop refinement. Through extensive experiments, our DiffX demonstrates robustness in cross-modal “RGB+X” image generation on FLIR, MFNet, and COME15K datasets, guided by various layout conditions. It also shows the potential for the adaptive generation of “RGB+X+Y(+Z)” images or more diverse modalities on COME15K and MCXFace datasets. Our code and constructed cross-modal image datasets are available at https://github.com/zeyuwang-zju/DiffX.  

<img src="img/model.png" style="zoom:65%;" />

## Updates

**15/09/2024** Code released!

## TODO:
- [x] Release code!
- [ ] Complete the usage instruction
- [ ] Release text captions in datasets
- [ ] Release pre-trained model weights

## Setup

To set up our environment, please run:

```
pip install -r requirements.txt
```

## Usage

We will soon complete this section.

## Pretrained Models / Data

Pretained models coming soon.


## Citation

If you make use of our work, please cite our paper:

```
@misc{wang2024diffxguidelayoutcrossmodal,
      title={DiffX: Guide Your Layout to Cross-Modal Generative Modeling}, 
      author={Zeyu Wang and Jingyu Lin and Yifei Qian and Yi Huang and Shicen Tian and Bosong Chai and Juncan Deng and Qu Yang and Lan Du and Cunjian Chen and Yufei Guo and Kejie Huang},
      year={2024},
      eprint={2407.15488},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.15488}, 
}
```

## Results
**Results1:** Qualitative results of cross-modal “RGB+X” generation on FLIR, MFNet, and COME15K datasets:

<img src="img/result1.png" alt="0" style="zoom:120%;" />



**Results2:** Qualitative results on “SOD → RGB+D+Edge” task on COME15K dataset:

<img src="img/result2.png" alt="0.7" style="zoom:50%;" />



**Results3:** Qualitative results on “3DDFA → RGB+NIR+SWIR+T” task on MCXFace dataset  :

<img src="img/result3.png" style="zoom:55%;" />
