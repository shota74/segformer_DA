python -m torch.distributed.launch --nproc_per_node=2 --master_port=29511 dist_test.py --name GTA5_cityscapes_1000 --resume ./run/GTA5_to_cityscapes_1000/experiment_2022-01-28_16:15:54.983505/model_best.pth.tar --result ./result/GTA5_cityscapes_1000/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29511 dist_test.py --name GTA5_cityscapes_200 --resume ./run/GTA5_to_cityscapes_200/experiment_2022-01-26_00:33:55.959965/model_best.pth.tar --result ./result/GTA5_cityscapes_200/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29511 dist_test.py --name GTA5_cityscapes_100 --resume ./run/GTA5_to_cityscapes_100/experiment_2022-01-23_14:06:16.691282/model_best.pth.tar --result ./result/GTA5_cityscapes_100/





