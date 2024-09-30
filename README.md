**支持数据集**

- SIRST
- NUDT-SIRST
- IRSTD-1k
- MIRSDT
- MWIRSTD

**支持算法**

- ACM
- ACLNet
- ISNet
- UIUNet
- DNANet
- SCTransNet
- RevCol

**单GPU训练**
```shell
torchrun --nproc_per_node=1 train.py --datapath '../datasets/' --dataset 'NUDT-MIRSDT' --model 'SCTransNet' --loss_func 'fullySup1' --train 1 --test 0 --fullySupervised True --deep_supervision False --batchsize 16 --epochs 10 --lr 0.0001 
```
**单机多卡GPU训练**
```shell
torchrun --nproc_per_node=4 train.py --datapath '../datasets/' --dataset 'NUDT-MIRSDT' --model 'SCTransNet' --loss_func 'fullySup1' --train 1 --test 0 --fullySupervised True --SpatialDeepSu
p False --batchsize 16 --epochs 10 --lr 0.0001 --DataParallel True
```
**多机多卡GPU训练**
```shell
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 train.py --datapath '../datasets/' --dataset 'NUDT-MIRSDT' --model 'SCTransNet' --loss_func 'fullySup1' --train 1 --test 0 --fullySupervised True --deep_supervision False --batchsize 16 --epochs 10 --lr 0.0001 --DataParallel True
```
根据实际情况修改nnodes和node_rank参数
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
