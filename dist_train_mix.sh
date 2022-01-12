
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29508 dist_train_mix.py --dataset Synscapes_to_cityscapes_100   --dataset_sorcee synscapes  --dataset_target cityscapes_100

python -m torch.distributed.launch --nproc_per_node=8 --master_port=29508 dist_train_mix.py --dataset Synscapes_to_cityscapes_200   --dataset_sorcee synscapes  --dataset_target cityscapes_200

python -m torch.distributed.launch --nproc_per_node=8 --master_port=29508 dist_train_mix.py --dataset Synscapes_to_cityscapes_500   --dataset_sorcee synscapes  --dataset_target cityscapes_500

python -m torch.distributed.launch --nproc_per_node=8 --master_port=29508 dist_train_mix.py --dataset Synscapes_to_cityscapes_1000   --dataset_sorcee synscapes  --dataset_target cityscapes_1000


python -m torch.distributed.launch --nproc_per_node=8 --master_port=29508 dist_train_mix.py --dataset GTA5_to_cityscapes_100   --dataset_sorcee GTA5  --dataset_target cityscapes_100
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29508 dist_train_mix.py --dataset GTA5_to_cityscapes_200   --dataset_sorcee GTA5  --dataset_target cityscapes_200
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29508 dist_train_mix.py --dataset GTA5_to_cityscapes_1000   --dataset_sorcee GTA5  --dataset_target cityscapes_1000