#!/usr/bin/env bash
sbatch train.slurm --experiment_name BaseModel --transformer_d_model 128 --transformer_n_heads 32 --transformer_dim_feedforward 128 --transformer_n_layers  24
sbatch train.slurm --experiment_name SModel --transformer_d_model 64 --transformer_n_heads 16 --transformer_dim_feedforward 64 --transformer_n_layers  12
sbatch train.slurm --experiment_name XSModel --transformer_d_model 16 --transformer_n_heads 4 --transformer_dim_feedforward 32 --transformer_n_layers  6
sbatch train.slurm --experiment_name MModel --transformer_d_model 150 --transformer_n_heads 50 --transformer_dim_feedforward 144 --transformer_n_layers  12
sbatch train.slurm --experiment_name LModel --transformer_d_model 256 --transformer_n_heads 32 --transformer_dim_feedforward 144 --transformer_n_layers  32
sbatch train.slurm --experiment_name XLModel --transformer_d_model 512 --transformer_n_heads 64 --transformer_dim_feedforward 128 --transformer_n_layers  16
sbatch train.slurm --experiment_name XSFullModel --transformer_d_model 16 --transformer_n_heads 4 --transformer_dim_feedforward 32 --transformer_n_layers  6 --pretrained_dtm_path "save/dtm_0.628.pl"
