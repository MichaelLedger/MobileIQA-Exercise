PYTORCH_ENABLE_MPS_FALLBACK=1 HF_ENDPOINT=https://hf-mirror.com python3 -u train.py --gpu_id 1 --seed 3407 --batch_size 4 --dataset spaq --loss MSE --model MobileVit_IQA --save_path ./Running_Test --epochs 1

PYTORCH_ENABLE_MPS_FALLBACK=1 HF_ENDPOINT=https://hf-mirror.com python3 -u train.py --gpu_id 1 --seed 3407 --batch_size 4 --dataset spaq --loss MSE --save_path ./Running_Distill --teacher_pkl ./Running_Test/best_model.pkl --epochs 1
