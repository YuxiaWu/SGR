CUDA_VISIBLE_DEVICES=0 python online_evaluate_task.py --with_image --mlp --lr 0.001
CUDA_VISIBLE_DEVICES=1 python online_evaluate_8.py --with_image --mlp --lr 0.001

CUDA_VISIBLE_DEVICES=3 python online_evaluate_task.py --with_image --mlp --lr 0.001 --neglink_num 1

CUDA_VISIBLE_DEVICES=1 python online_evaluate_task_domain.py --with_image --mlp --lr 0.001 --neglink_num 1 --step 3 --img_sim_thr 0.7 
CUDA_VISIBLE_DEVICES=1 python online_evaluate_task_domain_1225.py --with_image --mlp --lr 0.001 --neglink_num 1 --step 3 

# debug

CUDA_VISIBLE_DEVICES=0 python -m pdb online_evaluate_task.py --with_image --mlp --lr 0.001