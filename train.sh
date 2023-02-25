CUDA_VISIBLE_DEVICES=0 python main.py --with_image
CUDA_VISIBLE_DEVICES=0 python main.py --with_image --lr 0.001
CUDA_VISIBLE_DEVICES=0 python main.py --with_image --lr 0.001 --mlp
CUDA_VISIBLE_DEVICES=5 python main.py --with_image --lr 0.001 --mlp --pre_train

# normal
CUDA_VISIBLE_DEVICES=1 python main.py --with_image --lr 0.001 

# neglink
CUDA_VISIBLE_DEVICES=1 python main.py --with_image --lr 0.001 --mlp --neglink_num 0.1
CUDA_VISIBLE_DEVICES=1 python main.py --with_image --lr 0.001 --mlp --neglink_num 0.3
CUDA_VISIBLE_DEVICES=1 python main.py --with_image --lr 0.001 --mlp --neglink_num 0.5
CUDA_VISIBLE_DEVICES=1 python main.py --with_image --lr 0.001 --mlp --neglink_num 1

# image sim
CUDA_VISIBLE_DEVICES=5 python main.py --with_image --lr 0.001 --mlp --neglink_num 1 --img_sim_thr 0.7 --num-epoch 5
CUDA_VISIBLE_DEVICES=6 python main.py --with_image --lr 0.001 --mlp --neglink_num 1 --img_sim_thr 0.8 --num-epoch 5
CUDA_VISIBLE_DEVICES=2 python main.py --with_image --lr 0.001 --mlp --neglink_num 1 --img_sim_thr 0.9 --num-epoch 5
CUDA_VISIBLE_DEVICES=2 python main.py --with_image --lr 0.001 --mlp --neglink_num 1 --img_sim_thr 0.75 --num-epoch 5
CUDA_VISIBLE_DEVICES=3 python main.py --with_image --lr 0.001 --mlp --neglink_num 1 --img_sim_thr 0.85 --num-epoch 5