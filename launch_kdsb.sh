#!/bin/bash

# TODO: adapt to multi-node setting:
#   - requires a dynamic ACCELERATE_CONFIG, notably machine_rank

echo -e "\n<--------------------------------------- launch_script_DDIM.sh --------------------------------------->\n"

# ------------------------------------------> Variables <------------------------------------------
# experiment and output folder name; common to all runs in the same experiment
exp_name=FineTune_KDSB

# wandb run display name
run_name=FineTune_KDSB_SMALL

exp_dirs_parent_folder=./experiments
model_configs_folder=./models_configs

num_GPUS=1 # *total* number of processes (i.e. accross all nodes) = number of GPUs

# --------------------------------------> Accelerate config <--------------------------------------
if [[ "$num_GPUS" -gt 1 ]]; then
    acc_cfg="--multi_gpu"
else
    acc_cfg=""
fi

ACCELERATE_CONFIG="
${acc_cfg}
--machine_rank=0
--mixed_precision=fp16
--num_machines=1
--num_processes=${num_GPUS}
--rdzv_backend=static
--same_network
--dynamo_backend=no
--gpu_ids 2
--main_process_port=29502
"
# --main_process_port=29501

# ----------------------------------------> Script + args <----------------------------------------
MAIN_SCRIPT=train.py

MAIN_SCRIPT_ARGS="
$1
--exp_output_dirs_parent_folder ${exp_dirs_parent_folder}
--experiment_name ${exp_name}
--run_name ${run_name}
--model_type DDIM
--components_to_train denoiser
--denoiser_config_path ${model_configs_folder}/denoiser/super_small.json
--noise_scheduler_config_path ${model_configs_folder}/noise_scheduler/3k_steps_clipping_rescaling.json
--num_inference_steps 30
--train_data_dir /projects/deepdevpath2/Kian/datasets/KDSB/
--train_batch_size 128

--fine_tune_with_paired_dataset_mode translation
--fine_tune_experiment_by_paired_training /projects/deepdevpath2/Kian/PhenDiff/experiments/PhenDiff_KDSB/PhenDiff_KDSB_main/checkpoints/step_40000
--paired_train_data_dir /projects/deepdevpath2/Kian/datasets/KDSB/KDSB-paired-small/
--test_data_dir /projects/deepdevpath2/Kian/datasets/KDSB/KDSB-test/
--source_class_for_paired_training images
--paired_training_loss bce
--paired_train_batch_size 2
--visual_inspection_interval 250

--denoiser_in_channels 3
--denoiser_out_channels 3
--definition 128
--eval_batch_size 256
--max_num_steps 41500
--learning_rate 3e-4
--mixed_precision fp16
--eval_save_model_every_epochs 50
--nb_generated_images 1024
--checkpoints_total_limit 2
--checkpointing_steps 1000
--use_ema
--proba_uncond 0.1
--compute_fid
--compute_isc
--compute_kid
--wandb_entity kian-team
"

# --fine_tune_with_paired_dataset_mode translation
# --fine_tune_experiment_by_paired_training /projects/deepdevpath2/Kian/PhenDiff/experiments/PhenDiff/KDSB/checkpoints/step_50000
# --paired_train_data_dir /projects/deepdevpath2/Kian/datasets/TissueNet/TN-binary-paired-tiny/nuc/
# --test_data_dir /projects/deepdevpath2/Kian/datasets/KDSB/KDSB-test/
# --source_class_for_paired_training images
# --paired_training_loss bce
# --paired_train_batch_size 2

# ----------------------------------------> Echo commands <----------------------------------------
echo -e "START TIME: $(date)\n"
echo -e "EXPERIMENT NAME: ${exp_name}\n"
echo -e "RUN NAME: ${run_name}\n"
echo -e "EXP_DIRS_PARENT_FOLDER: ${exp_dirs_parent_folder}\n"
echo -e "ACCELERATE_CONFIG: ${ACCELERATE_CONFIG}\n"
echo -e "MAIN_SCRIPT: ${MAIN_SCRIPT}\n"
echo -e "MAIN_SCRIPT_ARGS: ${MAIN_SCRIPT_ARGS}\n"

# Launch the job
accelerate launch ${ACCELERATE_CONFIG} ${MAIN_SCRIPT} ${MAIN_SCRIPT_ARGS}

exit 0