torchrun --standalone --nproc_per_node=4 train.py  --dataset 'NUDT-SIRST' --model 'DNANet' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --deep_supervision True --batchsize 64 --epochs 100 --lr 0.01 --base_size 256 --crop_size 256

torchrun train.py --model 'DNANet' --dataset "IRSTD-1k" --loss_func 'fullySup1' --train 0 --test 1 --pth_path pth_path

rm -rf /home/guantp/Infrared/SIRST/results/*

python train.py --dataset 'NUDT-SIRST' --model 'DNANet' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --deep_supervision True --batchsize 32 --epochs 1000 --lr 0.01 --base_size 256 256 --crop_size 256

PYTHONPATH=/home/guantp/Infrared/SIRST/models/RevCol/mmsegmentation python train.py  --dataset 'SIRST' --model 'RevCol' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --deep_supervision True --batchsize 16 --epochs 30 --lr 0.01

PYTHONPATH=/home/guantp/Infrared/SIRST/models/RevCol/mmsegmentation python tools/train.py /home/guantp/Infrared/SIRST/models/RevCol/mmsegmentation/configs/revcol/upernet_revcol_tiny_fp16_512x512_160k_ade20k.py

CUDA_VISIBLE_DEVICES=0