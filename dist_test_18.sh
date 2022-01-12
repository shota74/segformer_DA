

python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name synscapes_cityscapes_1000 --resume ./run/synscapes_to_cityscapes_1000/experiment_2021-12-07_17:38:27.385309/model_best.pth.tar --result ./result/synscapes__cityscapes_1000/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name synscapes__cityscapes_500 --resume ./run/synscapes_to_cityscapes_500/experiment_2021-12-08_18:52:57.866496/model_best.pth.tar  --result ./result/synscapes__cityscapes_500/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name synscapes__cityscapes_200 --resume ./run/synscapes_to_cityscapes_200/experiment_2021-12-10_12:45:31.832534/model_best.pth.tar  --result ./result/synscapes__cityscapes_200/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name synscapes__cityscapes_100 --resume ./run/synscapes_to_cityscapes_100/experiment_2021-12-12_22:05:01.847324/model_best.pth.tar  --result ./result/synscapes__cityscapes_100/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name synscapes__cityscapes     --resume ./run/synscapes_to_cityscapes/experiment_2021-12-06_10:52:57.658987/model_best.pth.tar      --result ./result/synscapes__cityscapes/

python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name cityscapes_1000 --resume ./run/cityscapes_1000/experiment_2021-11-01_13:45:39.683970/model_best.pth.tar --result ./result/cityscapes_1000/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name cityscapes_500 --resume ./run/cityscapes_500/experiment_2021-11-02_16:10:56.034951/model_best.pth.tar --result ./result/cityscapes_500/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name cityscapes_200 --resume ./run/cityscapes_200/experiment_2021-11-03_18:18:13.678245/model_best.pth.tar --result ./result/cityscapes_200/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name cityscapes_100 --resume ./run/cityscapes_100/experiment_2021-11-05_00:17:48.609447/model_best.pth.tar --result ./result/cityscapes_100/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name cityscapes   --resume ./run/cityscapes/model_best.pth.tar --result ./result/cityscapes/

python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name GTA5_cityscapes_1000 --resume ./run/GTA5_to_cityscapes_1000/experiment_2021-12-17_01:38:37.277046/model_best.pth.tar --result ./result/GTA5_cityscapes_1000/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name GTA5_cityscapes_500 --resume ./run/GTA5_to_cityscapes_500/experiment_2021-12-18_15:18:06.162984/model_best.pth.tar --result ./result/GTA5_cityscapes_500/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name GTA5_cityscapes_200 --resume ./run/GTA5_to_cityscapes_200/experiment_2021-12-20_11:09:36.997006/model_best.pth.tar --result ./result/GTA5_cityscapes_200/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name GTA5_cityscapes_100 --resume ./run/GTA5_to_cityscapes_100/experiment_2021-12-22_07:52:04.494684/model_best.pth.tar --result ./result/GTA5_cityscapes_100/
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 dist_test.py --name GTA5_cityscapes     --resume ./run/GTA5_to_cityscapes/experiment_2021-12-15_12:26:25.018763/model_best.pth.tar --result ./result/GTA5_cityscapes/





