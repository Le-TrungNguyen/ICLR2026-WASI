saved_location="runs"
dataset="BoolQ"
num_classes="2"
usr_group_kl="full_pretrain_imagenet"

num_of_finetune=5

# usr_group_kl=15.29
# load_args="--model.load pretrained_ckpts/swinT/c10_epoch=24-val-acc=0.809.ckpt"

general_config_args="--config configs/general_cls_config_finetuned.yaml"
logger_args="--logger.save_dir $saved_location/TinyLlama/$dataset/base"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 24 --data.val_workers 24 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy"
trainer_args="--trainer.max_epochs 50" # --trainer.gpus 2 --trainer.strategy ddp"
model_args="--model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233 --data.batch_size 8 --model.just_log False --data.max_length 128"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args"

echo $common_args


python trainer.py ${all_args} ${common_args} --model.backbone $model --logger.exp_name base_l${num_of_finetune}_${usr_group_kl} --model.num_of_finetune $num_of_finetune
# python trainer.py ${all_args} ${common_args} --model.backbone $model --logger.exp_name base_all_${usr_group_kl} --model.num_of_finetune all