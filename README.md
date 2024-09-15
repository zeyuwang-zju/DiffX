# DiffX: Guide Your Layout to Cross-Modal Generative Modeling

[[Paper](https://arxiv.org/abs/2407.15488)] [[Code](https://github.com/zeyuwang-zju/DiffX)]

> **DiffX: Guide Your Layout to Cross-Modal Generative Modeling**
>
> **Authors:** Zeyu Wang*, Jingyu Lin*, Yifei Qian, Yi Huang, Shicen Tian, Bosong Chai, Juncan Deng, Qu Yang, Lan Du, Cunjian Chen, Yufei Guo, Kejie Huang (*These authors contributed equally. †Corresponding authors. )
> 

>**Abstract**:
> Diffusion models have made significant strides in languagedriven and layout-driven image generation. However, most diffusion models are limited to visible RGB image generation. In fact, human perception of the world is enriched by diverse viewpoints, such as chromatic contrast, thermal illumination, and depth information. In this paper, we introduce a novel diffusion model for general layoutguided cross-modal generation, called DiffX. Notably, our DiffX presents a simple yet effective cross-modal generative modeling pipeline, which conducts diffusion and denoising processes in the modality shared latent space. Moreover, we introduce the Joint-Modality Embedder (JME) to enhance the interaction between layout and text conditions by incorporating a gated attention mechanism. To facilitate the user-instructed training, we construct the cross-modal image datasets with detailed text captions by the LargeMultimodal Model (LMM) and our human-in-the-loop refinement. Through extensive experiments, our DiffX demonstrates robustness in cross-modal “RGB+X” image generation on FLIR, MFNet, and COME15K datasets, guided by various layout conditions. It also shows the potential for the adaptive generation of “RGB+X+Y(+Z)” images or more diverse modalities on COME15K and MCXFace datasets. Our code and constructed cross-modal image datasets are available at https://github.com/zeyuwang-zju/DiffX.  

<img src="img/model.png" style="zoom:65%;" />

## Updates

**[15/09/2024]** Code released!

## TODO
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
**1. Dataset Preparation**...
**2. Training:**
   
   For the four types of ''RGB+X'' generation tasks:
   ```
   # Firstly, train the MP-VAE:
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_autoencoder.py  --yaml_file=configs/flir_text.yaml  --DATA_ROOT=./DATA/flir/   --batch_size=2   --save_every_iters 1000   --name flir
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_autoencoder.py  --yaml_file=configs/mfnet.yaml  --DATA_ROOT=./DATA/mfnet/   --batch_size=2   --save_every_iters 1000   --name mfnet
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_autoencoder.py  --yaml_file=configs/come.yaml  --DATA_ROOT=./DATA/come/   --batch_size=2   --save_every_iters 1000   --name come
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_autoencoder.py  --yaml_file=configs/come_sobel.yaml  --DATA_ROOT=./DATA/come/   --batch_size=2   --save_every_iters 1000   --name come_sobel

   # Secondly, train the DiffX-UNet:
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_diffusion.py  --yaml_file=configs/flir_text.yaml  --DATA_ROOT=./DATA/flir/   --batch_size=8   --save_every_iters 1000   --name flir
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_diffusion.py  --yaml_file=configs/mfnet.yaml  --DATA_ROOT=./DATA/mfnet/   --batch_size=8   --save_every_iters 1000   --name mfnet
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_diffusion.py  --yaml_file=configs/come.yaml  --DATA_ROOT=./DATA/come/   --batch_size=8   --save_every_iters 1000   --name come
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_diffusion.py  --yaml_file=configs/come_sobel.yaml  --DATA_ROOT=./DATA/come/   --batch_size=8   --save_every_iters 1000   --name come_sobel
   ```
## Pretrained Models
Pretained models coming soon.
## Acknowledgements
This code is built on [GLIGEN (PyTorch)](https://github.com/gligen/GLIGEN). We thank the authors for sharing the codes.
## Contact
If you have any questions, please contact me by email (wangzeyu2020@zju.edu.cn).
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

<img src="img/result1.png" alt="0" style="zoom:110%;" />



**Results2:** Qualitative results on “SOD → RGB+D+Edge” task on COME15K dataset:

<img src="img/result2.png" alt="0.7" style="zoom:40%;" />



**Results3:** Qualitative results on “3DDFA → RGB+NIR+SWIR+T” task on MCXFace dataset  :

<img src="img/result3.png" style="zoom:40%;" />
