python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name label_16 --resume ./run/cityscapes_16_1000/experiment_2021-11-21_13:47:33.552503/model_best.pth.tar --result ./result/label_16/

