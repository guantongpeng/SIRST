torchrun --standalone --nproc_per_node=4 train.py  --dataset 'NUDT-SIRST' --model_name 'DNANet' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --deep_supervision True --batchsize 64 --epochs 100 --lr 0.01 --base_size 256 --crop_size 256

torchrun train.py --model_name 'DNANet' --dataset "IRSTD-1k" --loss_func 'fullySup1' --train 0 --test 1 --pth_path pth_path

rm -rf /home/guantp/Infrared/SIRST/results/*

python train.py --dataset 'NUDT-SIRST' --model_name 'DNANet' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --deep_supervision True --batchsize 32 --epochs 1000 --lr 0.01 --base_size 256 256 --crop_size 256

PYTHONPATH=/home/guantp/Infrared/SIRST/models/RevCol/mmsegmentation python train.py  --dataset 'SIRST' --model_name 'RevCol' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --deep_supervision True --batchsize 16 --epochs 30 --lr 0.01

PYTHONPATH=/home/guantp/Infrared/SIRST/models/RevCol/mmsegmentation python tools/train.py /home/guantp/Infrared/SIRST/models/RevCol/mmsegmentation/configs/revcol/upernet_revcol_tiny_fp16_512x512_160k_ade20k.py

CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset 'NUDT-SIRST' --model_name 'SCTransNet' --loss_func 'fullySup' --train 1 --test 0 --deep_supervision True --batchsize 128 --epochs 1000 --lr 0.01 --base_size 256 256 --crop_size 256 --optimizer_name 'Adam' --test_epoch 1 --save_pred_img True

CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'SIRST' --model_name 'SCTransNet' --train 0 --test 1 --base_size 256 256 --crop_size 256 --save_pred_img True --pth_path 

python train.py --dataset 'NUDT-SIRST' --model_name 'SCTransNet' --train 0 --test 1 --base_size 256 256 --crop_size 256 --save_pred_img True --pth_path 

dataset MWIRSTD  NUDT-SIRST
model SCTransNet

python train.py --dataset 'SIRST' --model_name 'MSHNet' --train 0 --test 1 --base_size 256 256 --crop_size 256 --save_pred_img True --pth_path /home/guantp/Infrared/SIRST/results/SIRST_MSHNet/models/16_202410182234_net_epoch_1000_loss_0.2041_IoU-0.7593_nIoU-0.7896.pth