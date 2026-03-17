saved_location="runs"
truncation_thresholds=(0.4 0.5 0.6 0.7 0.8 0.9)

# Dataset configurations
declare -A datasets
datasets=(
  ["pets"]=37
  ["flowers102"]=102
  ["cub200"]=200
  ["cifar10"]=10
  ["cifar100"]=100
  ["isic2018"]=7
  # ["imagenet"]=1000
)

# Model names
models=("vit_b_32" "swinT")

# List of num_of_finetune values and corresponding budgets
num_of_finetune_list=(all)

# Common setup
declare -A perplexity_pkls=(
  ["vit_b_32"]="--model.perplexity_pkl perplexity/perplexity_vit_b_32.pkl"
  ["swinT"]="--model.perplexity_pkl perplexity/perplexity_swinT.pkl"
)


common_data_args=""
common_model_args=""
common_trainer_args=""
common_seed_args="--seed_everything 233"

# Loop through truncation thresholds
for var_index in "${!truncation_thresholds[@]}"; do
# for truncation_threshold in "${truncation_thresholds[@]}"; do
  truncation_threshold="${truncation_thresholds[var_index]}"
  echo "Running with truncation_threshold: ${truncation_threshold}"

  common_methods="--model.with_WASI True --model.truncation_threshold $truncation_threshold"
  common_args="$common_data_args $common_model_args $common_trainer_args $common_seed_args $common_methods"
  # Loop through models and datasets
  for model in "${models[@]}"; do
    echo "Processing model: $model"

    perplexity_pkl="${perplexity_pkls[$model]}"

    model_config_args="--config configs/general_cls_config_finetuned.yaml"

    for dataset in "${!datasets[@]}"; do
      num_classes=${datasets[$dataset]}
      echo "  Processing dataset: $dataset with num_classes: $num_classes"

      if [ "$dataset" == "imagenet" ]; then
        data_dir="~/imagenet"
      else
        data_dir="data/$dataset"
      fi

      specific_logger_args="--logger.save_dir ${saved_location}/$model/$dataset/WASI/var${truncation_threshold}/finetuning"
      specific_data_args="--data.name $dataset --data.data_dir $data_dir"
      specific_model_args="--model.num_classes $num_classes $perplexity_pkl"
      specific_args="$specific_logger_args $specific_data_args $specific_model_args"

      all_args="$model_config_args $common_args $specific_args"

      for i in "${!num_of_finetune_list[@]}"; do
        num_of_finetune="${num_of_finetune_list[i]}"

        echo "    Running with num_of_finetune: $num_of_finetune"
        python trainer.py ${all_args} --model.backbone $model --logger.exp_name WASI_var${truncation_threshold}_l${num_of_finetune} --model.num_of_finetune $num_of_finetune --model.budget 0.01
      done
    done
  done
done
