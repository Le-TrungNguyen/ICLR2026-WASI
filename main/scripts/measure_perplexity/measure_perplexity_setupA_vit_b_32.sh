#!/bin/bash

saved_location="runs"

imagenet_link="data/imagenet/"

# Model names
models=("vit_b_32")

# Common setup

common_data_args="--data.train_workers 24 --data.val_workers 24 --data.partition 1 --data.batch_size 128 --data.num_train_batch 1 --data.num_val_batch 1"
common_model_args=""
common_trainer_args=""
common_seed_args="--seed_everything 233"
common_methods="--model.measure_perplexity_HOSVD_var True"

common_args="$common_data_args $common_model_args $common_trainer_args $common_seed_args $common_methods"

# Loop through models and datasets
for i in "${!models[@]}"; do
  model="${models[i]}"
  echo "Processing model: $model"

  model_config_args="--config configs/general_cls_config_finetuned.yaml"

  num_classes=1000
  echo "  Processing with num_classes: $num_classes"
  specific_logger_args="--logger.save_dir ${saved_location}/$model/imagenet_perplexity_HOSVD_var"
  specific_data_args="--data.name imagenet --data.data_dir $imagenet_link"
  specific_model_args="--model.num_classes $num_classes"
  specific_args="$specific_logger_args $specific_data_args $specific_model_args"

  all_args="$model_config_args $model_args $common_args $specific_args"
  echo $all_args

  # python trainer.py ${all_args} --model.backbone $model --logger.exp_name perplexity_test_var_0.8 --set_of_epsilons 0.8
  # python trainer.py ${all_args} --model.backbone $model --logger.exp_name perplexity_test_var_0.9 --set_of_epsilons 0.9
  python trainer.py ${all_args} --model.backbone $model --logger.exp_name perplexity_test_var_0.4to0.9 --set_of_epsilons 0.4,0.5,0.6,0.7,0.8,0.9
done
