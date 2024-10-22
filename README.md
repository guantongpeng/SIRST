## Single-frame Infrared Small Target Detction

### Supported Datasets

- SIRST
- NUDT-SIRST
- IRSTD-1k
- MWIRSTD

### Supported Models

- ACM
- ACLNet
- ISNet
- UIUNet
- DNANet
- SCTransNet
- RDIAN
- EGEUNet
- EffiSegNet
- UNet Series
- ...

### Train And Test

**Train Code Example**
```shell
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'SIRST' --model_name 'SCTransNet' --train 0 --test 1 --base_size 256 256 --crop_size 256 --save_pred_img True --pth_path  
```
**Test Code Example**
```shell
python train.py --dataset 'SIRST' --model_name 'SCTransNet' --train 0 --test 1 --base_size 256 256 --crop_size 256 --save_pred_img  True --pth_path your_pth_path
```

For the SIRST and NUDT-SIRST datasets, it is recommended to employ the parameters `--base_size 256 256 --crop_size 256`. Conversely, for the IRSTD-1k dataset, the suggested parameters are `--base_size 512 512 --crop_size 512`.

### References
I sincerely appreciate the following outstanding work and code !

**Paper List:** [Awesome Infrared Small Targets](https://github.com/Tianfang-Zhang/awesome-infrared-small-targets)

**DTUM:** [[code]](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM) | [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10321723)

**UIUNet:** [[code]](https://github.com/danfenghong/IEEE_TIP_UIU-Net) | [[paper]](https://ieeexplore.ieee.org/document/9989433)

**SCTransNet:** [[code]](https://github.com/xdFai/SCTransNet) | [[paper]](https://ieeexplore.ieee.org/document/10486932)

**DNANet:** [[code]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) | [[paper]](https://arxiv.org/pdf/2106.00487)

**MSHNet:** [[code]](https://github.com/Lliu666/MSHNet) | [[paper]](https://arxiv.org/abs/2403.19366)

**SIRST-5K:** [[code]](https://github.com/luy0222/SIRST-5K) | [[paper]](https://arxiv.org/abs/2403.05416)

**MiM-ISTD:** [[code]](https://github.com/txchen-USTC/MiM-ISTD) | [[paper]](https://arxiv.org/abs/2403.02148)

**IRSAM:** [[code]](https://github.com/IPIC-Lab/IRSAM) | [[paper]](https://arxiv.org/abs/2407.07520)
