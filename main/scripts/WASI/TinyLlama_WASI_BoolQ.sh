saved_location="runs"
truncation_thresholds=(0.4 0.5 0.6 0.7 0.8 0.9)

# Dataset configurations
declare -A datasets
datasets=(
  ["BoolQ"]=2
  # ["AG_News"]=4
)

# Model names
models=("TinyLlama")

# List of num_of_finetune values and corresponding budgets
num_of_finetune_list=(1 2 3 4 5)

declare -A budgets=(
  ["BoolQ"]="5"
  # ["AG_News"]="5"
)

# Common setup
common_data_args="--data.train_workers 10 --data.val_workers 10 --data.max_length 128 --data.batch_size 8"
common_model_args=""
common_trainer_args="" # --trainer.gpus 3 --trainer.strategy ddp"
common_seed_args="--seed_everything 233"

# Loop through truncation thresholds
for var_index in "${!truncation_thresholds[@]}"; do
  truncation_threshold="${truncation_thresholds[var_index]}"
  echo "Running with truncation_threshold: ${truncation_threshold}"

  common_methods="--model.with_WASI True --model.truncation_threshold $truncation_threshold --model.just_log True"
  common_args="$perplexity_pkl $common_data_args $common_model_args $common_trainer_args $common_seed_args $common_methods"
  # Loop through models and datasets
  for model in "${models[@]}"; do
    echo "Processing model: $model"

    model_config_args="--config configs/general_cls_config_finetuned.yaml"

    for dataset in "${!datasets[@]}"; do
      num_classes=${datasets[$dataset]}
      echo "  Processing dataset: $dataset with num_classes: $num_classes"
      specific_logger_args="--logger.save_dir ${saved_location}/$model/$dataset/WASI/var${truncation_threshold}/"
      specific_data_args="--data.name $dataset --data.data_dir data/$dataset"
      specific_model_args="--model.num_classes $num_classes"
      specific_args="$specific_logger_args $specific_data_args $specific_model_args"

      all_args="$model_config_args $common_args $specific_args"
      # echo $all_args

      for i in "${!num_of_finetune_list[@]}"; do
        num_of_finetune="${num_of_finetune_list[i]}"

        echo "    Running with num_of_finetune: $num_of_finetune"
        python trainer.py ${all_args} ${all_args} --logger.exp_name WASI_${truncation_threshold}_l${num_of_finetune} --model.num_of_finetune $num_of_finetune
      done
    done
  done
done
