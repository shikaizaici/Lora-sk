accelerate launch --num_cpu_threads_per_process 1 train_network.py
    --pretrained_model_name_or_path="D:\BaiduNetdiskDownload\sd-webui-aki\sd-webui-aki-v4.9.1\models\Stable-diffusion\v1-5-pruned.safetensors"
    --dataset_config="D:\BaiduNetdiskDownload\sd-scripts-main\Lora.toml"
    --output_dir="D:\BaiduNetdiskDownload\sd-scripts-main\lora_sk"
    --output_name="sk"
    --save_model_as=safetensors
    --prior_loss_weight=1.0
    --max_train_steps=400
    --learning_rate=1e-4
    --optimizer_type="AdamW8bit"
    --xformers
    --mixed_precision="fp16"
    --cache_latents
    --gradient_checkpointing
    --save_every_n_epochs=1
    --network_module=networks.lora

