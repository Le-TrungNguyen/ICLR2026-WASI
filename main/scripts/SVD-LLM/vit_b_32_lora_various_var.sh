#!/bin/bash

saved_location="runs"
truncation_thresholds=(8)

checkpoints=(
  SVD_LLM_checkpoints/vit_b_32_whitening_only_0.1.pt    # equal to epsilon = 0.4
  SVD_LLM_checkpoints/vit_b_32_whitening_only_0.15.pt   # equal to epsilon = 0.5
  SVD_LLM_checkpoints/vit_b_32_whitening_only_0.22.pt   # equal to epsilon = 0.6
  SVD_LLM_checkpoints/vit_b_32_whitening_only_0.29.pt   # equal to epsilon = 0.7
  SVD_LLM_checkpoints/vit_b_32_whitening_only_0.4.pt    # equal to epsilon = 0.8
  SVD_LLM_checkpoints/vit_b_32_whitening_only_0.57.pt   # equal to epsilon = 0.9
)

# Dataset configurations
declare -A datasets
datasets=(
  ["pets"]=37
  ["flowers102"]=102
  ["cub200"]=200
  ["cifar10"]=10
  ["cifar100"]=100
)

# Model names
models=("vit_b_32")

num_of_finetune_list=(all)

# Common setup
perplexity_pkl="--model.perplexity_pkl runs/setupA/vit_b_32/imagenet_perplexity_HOSVD_var/perplexity_test_var_0.4to0.9_imagenet/perplexity.pkl"
common_data_args=""
common_model_args=""
common_trainer_args=""
common_seed_args="--seed_everything 233"

# Loop through truncation thresholds
for checkpoint in "${checkpoints[@]}"; do
  echo "Running checkpoint: ${checkpoint}"
  compression_rate=$(echo "$checkpoint" | sed -n 's/.*only_\([0-9.]*\)\.pt/\1/p')

  for var_index in "${!truncation_thresholds[@]}"; do
  # for truncation_threshold in "${truncation_thresholds[@]}"; do
    truncation_threshold="${truncation_thresholds[var_index]}"
    echo "Running with truncation_threshold: ${truncation_threshold}"

    common_methods="--model.checkpoint $checkpoint --model.with_lora True --model.truncation_threshold $truncation_threshold"
    common_args="$perplexity_pkl $common_data_args $common_model_args $common_trainer_args $common_seed_args $common_methods"
    # Loop through models and datasets
    for model in "${models[@]}"; do
      echo "Processing model: $model"

      # model_config_args="--config configs/${model}_config_use_checkpoint.yaml"
      model_config_args="--config configs/general_cls_config_finetuned.yaml"

      for dataset in "${!datasets[@]}"; do
        num_classes=${datasets[$dataset]}
        echo "  Processing dataset: $dataset with num_classes: $num_classes"
        specific_logger_args="--logger.save_dir ${saved_location}/$model/$dataset/LoRA_${truncation_threshold}/"
        specific_data_args="--data.name $dataset --data.data_dir data/$dataset"
        specific_model_args="--model.num_classes $num_classes"
        specific_args="$specific_logger_args $specific_data_args $specific_model_args"

        all_args="$model_config_args $common_args $specific_args"
        # echo $all_args

        for i in "${!num_of_finetune_list[@]}"; do
          num_of_finetune="${num_of_finetune_list[i]}"

          echo "    Running with num_of_finetune: $num_of_finetune, rank: $truncation_threshold"
          python trainer.py ${all_args} --model.backbone $model --logger.exp_name LoRA_r${truncation_threshold}_compress${compression_rate} --model.num_of_finetune $num_of_finetune
        done
      done
    done
  done
done
