saved_location="runs"

# Dataset configurations
declare -A datasets
datasets=(
  ["pets"]=37
  ["flowers102"]=102
  ["cub200"]=200
  ["cifar10"]=10
  ["cifar100"]=100
  ["isic2018"]=7
)

# Model names
models=("vit_b_32" "swinT" "mcunet")

# List of num_of_finetune values and corresponding budgets
num_of_finetune_list=(all)

# Common setup
common_data_args=""
common_model_args=""
common_trainer_args=""
common_seed_args="--seed_everything 233"
common_args="$common_data_args $common_model_args $common_trainer_args $common_seed_args"
# Loop through models and datasets
for model in "${models[@]}"; do
  echo "Processing model: $model"
  model_config_args="--config configs/general_cls_config_finetuned.yaml"
  for dataset in "${!datasets[@]}"; do
    num_classes=${datasets[$dataset]}
    echo "  Processing dataset: $dataset with num_classes: $num_classes"
    specific_logger_args="--logger.save_dir ${saved_location}/$model/$dataset/vanilla/finetuning"
    specific_data_args="--data.name $dataset --data.data_dir data/$dataset"
    specific_model_args="--model.num_classes $num_classes"
    specific_args="$specific_logger_args $specific_data_args $specific_model_args"

    all_args="$model_config_args $common_args $specific_args"

    for i in "${!num_of_finetune_list[@]}"; do
      num_of_finetune="${num_of_finetune_list[i]}"
      python trainer.py ${all_args} --model.backbone $model --logger.exp_name vanilla_l${num_of_finetune} --model.num_of_finetune $num_of_finetune
    done
  done
done
