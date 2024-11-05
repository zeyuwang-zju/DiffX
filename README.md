# DiffX

> **TITLE:** DiffX: Guide Your Layout to Cross-Modal Generative Modeling [[Paper](https://arxiv.org/abs/2407.15488)] [[Page](https://jonyond-lin.github.io/DiffX_page/)]
>
> **AUTHORS:** Zeyu Wang*, Jingyu Lin*, Yifei Qian, Yi Huang, Shicen Tian, Bosong Chai, Juncan Deng, Qu Yang, Lan Du, Cunjian Chen#, Kejie Huang# (*These authors contributed equally; #Corresponding authors).
> 
>**ABSTRACT**:
> Diffusion models have made significant strides in language-driven and layout-driven image generation. However, most diffusion models are limited to visible RGB image generation. In fact, human perception of the world is enriched by diverse viewpoints, such as chromatic contrast, thermal illumination, and depth information. In this paper, we introduce a novel diffusion model for general layout-guided cross-modal generation, called DiffX. Notably, our DiffX presents a compact and effective cross-modal generative modeling pipeline, which conducts diffusion and denoising processes in the modality-shared latent space. Moreover, we introduce the Joint-Modality Embedder (JME) to enhance the interaction between layout and text conditions by incorporating a gated attention mechanism. To facilitate the user-instructed training, we construct the cross-modal image datasets with detailed text captions by the Large-Multimodal Model (LMM) and our human-in-the-loop refinement. Through extensive experiments, our DiffX demonstrates robustness in cross-modal “RGB+X” image generation on FLIR, MFNet, and COME15K datasets, guided by various layout conditions. Meanwhile, it shows the strong potential for the adaptive generation of “RGB+X+Y(+Z)” images or more diverse modalities on FLIR, MFNet, COME15K, and MCXFace datasets. To our knowledge, DiffX is the first model for layout-guided cross-modal image generation. Our code and constructed cross-modal image datasets are available at https://github.com/zeyuwang-zju/DiffX.

![model](https://github.com/user-attachments/assets/ea6c81ad-e8b1-423c-ac63-e9354329c385)

![image](https://github.com/user-attachments/assets/974a0a7d-ca79-4de9-b7e4-c1e46a4079cf)

## 📜 Updates
🚀 **[20/10/2024]** New experiments on RGB+T+D generation are added!

🚀 **[03/10/2024]** [Page](https://jonyond-lin.github.io/DiffX_page/) is released!

🚀 **[15/09/2024]** Instruction is completed!

🚀 **[15/09/2024]** Code is released!

🚀 **[28/07/2024]** [Paper](https://arxiv.org/abs/2407.15488) is released!

## 👨‍💻 TODO
- [x] Release the code!
- [x] Complete the instruction!
- [x] Release the page!
- [ ] Release text captions in datasets
- [ ] Release pre-trained model weights

## 🛠️ Usage

**1. Repo Clone & Environment Setup:**

Please first clone our repo from github by running the following command.
```
git clone https://github.com/zeyuwang-zju/DiffX.git
cd DiffX
```

To set up our environment, please run:
```
pip install -r requirements.txt
```

**2. Data Preparation**... (This part will come soon)

**3. Long-CLIP Model Preparation:**

Download the checkpoint of [Long-CLIP](https://huggingface.co/BeichenZhang/LongCLIP-L) and place it under `./ldm/modules/encoders/long_clip/checkpoints/`.

**4. Training:**

   For the ''RGB+X'' generation tasks on FLIR, MFNet, and COME15K datasets:
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

   For the ''RGB+T+D'' generation tasks on FLIR and MFNet datasets:
   ```
   # Firstly, train the MP-VAE:
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_autoencoder3.py  --yaml_file=configs/flir_text_triple.yaml  --DATA_ROOT=./DATA/flir/   --batch_size=1   --save_every_iters 1000   --name flir3
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_autoencoder3.py  --yaml_file=configs/mfnet_triple.yaml  --DATA_ROOT=./DATA/mfnet/   --batch_size=1   --save_every_iters 1000   --name mfnet3
   # Secondly, train the DiffX-UNet:
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_diffusion3.py  --yaml_file=configs/flir_text_triple.yaml  --DATA_ROOT=./DATA/flir/   --batch_size=8   --save_every_iters 1000   --name flir3
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_diffusion3.py  --yaml_file=configs/mfnet_triple.yaml  --DATA_ROOT=./DATA/mfnet/   --batch_size=8   --save_every_iters 1000   --name mfnet3
   ```

   For the ''3DDFA → RGB+NIR+SWIR+T'' task on MCXFace dataset:
   ```
   # Firstly, train the MP-VAE:
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_autoencoder4.py  --yaml_file=configs/mcxface.yaml  --DATA_ROOT=./DATA/mcxface/   --batch_size=1   --save_every_iters 1000   --name mcxface
   # Secondly, train the DiffX-UNet:
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/train_diffusion4.py  --yaml_file=configs/mcxface.yaml  --DATA_ROOT=./DATA/mcxface/   --batch_size=2   --save_every_iters 1000   --name mcxface
   ```

**5. Inference:**

   For the ''RGB+X'' generation tasks on FLIR, MFNet, and COME15K datasets:
   ```
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 scripts/inference.py  --yaml_file=configs/flir_text.yaml  --DATA_ROOT=./DATA/flir/  --name flir
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 scripts/inference.py  --yaml_file=configs/mfnet.yaml  --DATA_ROOT=./DATA/mfnet/  --name mfnet
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 scripts/inference.py  --yaml_file=configs/come.yaml  --DATA_ROOT=./DATA/come/  --name come
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 scripts/inference.py  --yaml_file=configs/come_sobel.yaml  --DATA_ROOT=./DATA/come/  --name come_sobel
   ```

   For the ''RGB+T+D'' generation tasks on FLIR and MFNet datasets:
   ```
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 scripts/inference3.py  --yaml_file=configs/flir_text_triple.yaml  --DATA_ROOT=./DATA/flir/  --name flir3
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 scripts/inference3.py  --yaml_file=configs/mfnet_triple.yaml  --DATA_ROOT=./DATA/mfnet/  --name mfnet3
   ```

   For the ''3DDFA → RGB+NIR+SWIR+T'' task on MCXFace dataset:
   ```
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 scripts/inference4.py  --yaml_file=configs/mcxface.yaml  --DATA_ROOT=./DATA/mcxface/  --batch_size=1  --name mcxface
   ```

## Pre-trained Models
Pre-trained models coming soon.

## Acknowledgements
This code is built on [GLIGEN (PyTorch)](https://github.com/gligen/GLIGEN) and [Long-CLIP (PyTorch)](https://github.com/beichenzbc/Long-CLIP). We thank the authors for sharing the codes.

## Licencing

Copyright (C) 2024 Zeyu Wang

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

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

![image](https://github.com/user-attachments/assets/44588c75-0b24-4fbd-8b70-6412adb5823a)



**Results2:** Qualitative results on cross-modal “RGB+X+Y” generation on FLIR, MFNet, and COME15K datasets:

![image](https://github.com/user-attachments/assets/c934757d-c78a-4e7a-a9a7-e70af46222e7)



**Results3:** Qualitative results on “3DDFA → RGB+NIR+SWIR+T” task on MCXFace dataset  :

![result3](https://github.com/user-attachments/assets/1c74c3f3-3c31-4a36-b81f-fc3605fc24a5)

More results can be found in our paper and appendix!
