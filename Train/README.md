# Training Instructions

This document outlines the steps to train the MobileIQA models using the SPAQ dataset.

## Prerequisites
- Python 3.x
- Required packages from requirements.txt
- SPAQ dataset properly set up

## Training Process

### 1. Teacher Model Training
```bash
# Run the first command in train.sh
HF_ENDPOINT=https://hf-mirror.com python3 -u train.py --gpu_id 1 --seed 3407 --batch_size 8 --dataset spaq --loss MSE --model MobileVit_IQA --save_path ./Running_Test --epochs 3
```
This will train the teacher model for 3 epochs. The best model will be saved as `best_model.pkl` in the `Running_Test` directory.

### 2. Knowledge Distillation
```bash
# Run the second command in train.sh
HF_ENDPOINT=https://hf-mirror.com python3 -u train.py --gpu_id 1 --seed 3407 --batch_size 8 --dataset spaq --loss MSE --save_path ./Running_Distill --teacher_pkl ./Running_Test/best_model.pkl --epochs 3
```
This will perform knowledge distillation for 3 epochs using the trained teacher model.

## Model Size Optimization

To minimize the size of the trained model (.pkl files):
1. During training, only the best model is saved (based on validation performance)
2. Previous checkpoints are automatically removed
3. The model architecture is designed to be lightweight while maintaining performance

## Output Files
- Teacher model: `./Running_Test/best_model.pkl`
- Student model: `./Running_Distill/best_model.pkl`
- Training logs can be found in the respective directories

## Notes
- The training process uses GPU ID 1. Adjust the `--gpu_id` parameter if needed.
- The random seed is set to 3407 for reproducibility.
- Batch size is set to 8, adjust based on your GPU memory.
