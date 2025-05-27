accelerate launch train_taksie.py \
 --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
 --gradient_accumulation_steps=4 --cache_dir=.cache/.huggingface --checkpoints_total_limit=1  \
 --resolution=256 \
 --learning_rate=1e-5 \
 --train_batch_size=32 --num_train_epochs 10000 --checkpointing_steps 5000 --report_to="wandb" --output_dir=test --tracker_project_name=Taksie --running_name=train_d --validation_step=100 --if_normalize_conditioning --debug_data_num=-1 --keyframe_info_path=keyframe_segment.npy --data_dir=example/example_trajectory --feature_dir=cache_features/clip --lstm_feature=768 
