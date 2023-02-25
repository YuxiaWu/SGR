CUDA_VISIBLE_DEVICES=0 python main.py --load_model --test --with_image --step 20
CUDA_VISIBLE_DEVICES=1 python main.py --load_model --test --with_image --test_GCN --step 20
CUDA_VISIBLE_DEVICES=1 python main.py --load_model --test --with_image --request_refine --step 20
CUDA_VISIBLE_DEVICES=3 python main.py --load_model --test --with_image --test_GCN --test_GCN_epoch 50 --step 3 --mlp
CUDA_VISIBLE_DEVICES=3 python main.py --load_model --test --with_image --test_GCN --test_GCN_epoch 50 --step 3 --mlp --lr 0.001
CUDA_VISIBLE_DEVICES=1 python main.py --load_model --test --with_image --step 10 --mlp --lr 0.001 --pre_train



CUDA_VISIBLE_DEVICES=0 python main.py --load_model --test --with_image --lr 0.001 --mlp --pre_train --step 5

# test
CUDA_VISIBLE_DEVICES=0 python main.py --load_model --test --with_image --lr 0.001 --mlp --step 5

# test neg link
# test
CUDA_VISIBLE_DEVICES=0 python main.py --load_model --test --with_image --lr 0.001 --mlp --step 3 --neglink_num 0.2
CUDA_VISIBLE_DEVICES=1 python main.py --load_model --test --with_image --lr 0.001 --mlp --step 3 --neglink_num 0.6
CUDA_VISIBLE_DEVICES=1 python main.py --load_model --test --with_image --lr 0.001 --mlp --step 3 --neglink_num 1.0
CUDA_VISIBLE_DEVICES=3 python main.py --load_model --test --with_image --lr 0.001 --mlp --step 3 --neglink_num 1

CUDA_VISIBLE_DEVICES=1 python test_domain.py --load_model --test --with_image --lr 0.001 --mlp --step 3 --neglink_num 1 --img_sim_thr 0.7

CUDA_VISIBLE_DEVICES=6 python main.py --load_model --test --with_image --lr 0.001 --mlp --step 3 --neglink_num 1 --img_sim_thr 0.7
CUDA_VISIBLE_DEVICES=6 python main.py --load_model --test --with_image --lr 0.001 --mlp --step 3 --neglink_num 1 --img_sim_thr 0.9
CUDA_VISIBLE_DEVICES=5 python main.py --load_model --test --with_image --lr 0.001 --mlp --step 3 --neglink_num 1 --img_sim_thr 0.8

