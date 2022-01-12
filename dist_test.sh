python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name SYNTHIA_cityscapes_16_1000 --resume ./run/SYNTHIA_to_cityscapes_1000/experiment_2021-12-25_22:28:53.049763/model_best.pth.tar --result ./result/SYNTHIA_cityscapes_16_1000/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name SYNTHIA_cityscapes_16_500 --resume ./run/SYNTHIA_to_cityscapes_500/experiment_2021-12-26_23:24:57.840727/model_best.pth.tar --result ./result/SYNTHIA_cityscapes_16_500/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name SYNTHIA_cityscapes_16_200 --resume ./run/SYNTHIA_to_cityscapes_200/experiment_2021-12-28_17:35:12.631340/model_best.pth.tar --result ./result/SYNTHIA_cityscapes_16_200/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name SYNTHIA_cityscapes_16_100 --resume ./run/SYNTHIA_to_cityscapes_100/experiment_2021-12-30_16:49:53.106564/model_best.pth.tar --result ./result/SYNTHIA_cityscapes_16_100/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name SYNTHIA_cityscapes_16 --resume ./run/SYNTHIA_to_cityscapes/experiment_2021-12-24_22:37:55.346283/model_best.pth.tar --result ./result/SYNTHIA_cityscapes_16/

python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name cityscapes_16_1000 --resume ./run/cityscapes_16_1000/experiment_2021-11-21_13:47:33.552503/model_best.pth.tar --result ./result/cityscapes_16_1000/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name cityscapes_16_500 --resume ./run/cityscapes_16_500/experiment_2021-11-22_14:15:13.875121/model_best.pth.tar --result ./result/cityscapes_16_500/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name cityscapes_16_200 --resume ./run/cityscapes_16_200/experiment_2021-12-02_13:00:22.805074/model_best.pth.tar --result ./result/cityscapes_16_200/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name cityscapes_16_100 --resume ./run/cityscapes_16_100/experiment_2021-12-04_10:31:23.912126/model_best.pth.tar --result ./result/cityscapes_16_100/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name cityscapes_16     --resume ./run/cityscapes_16/experiment_2021-11-08_10\:27\:12.718191/model_best.pth.tar --result ./result/cityscapes_16/




