# Lora trian
## 安装库 requirements.txt
该文件未包含PyTorch的版本要求，因为PyTorch的版本依赖于环境配置，因此未在文件中列出。请根据您的环境先行安装PyTorch:
```
pip install -r requirement.txt
```

## 使用文档链接


[由 darkstorm2150 提供的英文翻译在此](https://github.com/darkstorm2150/sd-scripts#links-to-usage-documentation).

* [训练指南 - 常见](./docs/train_README-ja.md) : 数据描述，选项等...
  * [中文版本](./docs/train_README-zh.md)
* [SDXL 训练](./docs/train_SDXL-en.md) (英文版本)
* [数据集 config](./docs/config_README-ja.md) 
  * [英文版本](./docs/config_README-en.md)
* [DreamBooth 训练 指导](./docs/train_db_README-ja.md)
* [逐步微调指南](./docs/fine_tune_README_ja.md):
* [Lora 训练](./docs/train_network_README-ja.md)
* [训练文本反转](./docs/train_ti_README-ja.md)
* [图片生成](./docs/gen_img_README-ja.md)
* note.com [模型转换](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windows 所需依赖项
Virtual environment: Python 3.10.6

## Windows 安装
打开一个普通的Powershell终端，在里面输入以下命令：:
```
git clone git@github.com:xxxxxxxxxxxxxxxxxxx20gex/Lora.git
cd sd-scripts
conda activate ‘你创建的虚拟环境’
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

accelerate config

##1、开始训练
1.收集照片8张，包括正脸照片4张（3张佩戴眼镜，1张未佩戴）  侧脸照片四张（左右脸各2张）
 

##2.处理照片：使用工具Photosho对照片进行抠图，去除背景

##3.对数据集进行处理
3.1 使用了BLIP生成captions
脚本：python "D:\BaiduNetdiskDownload\sd-scripts-main\finetune\make_captions.py" "D:\BaiduNetdiskDownload\sd-scripts-main\picture\sk\5_zkz"




##3.2 使用了WD14Tagger 生成标签（因为比较精确）
脚本：python finetune/tag_images_by_wd14_tagger.py --onnx --repo_id SmilingWolf/wd-swinv2-tagger-v3 --batch_size 4 D:\BaiduNetdiskDownload\sd-scripts-main\picture\sk-1



 
##3.3 将 captions 与标签进行预处理形成 metadata.json 元数据文件并且对文件进行清洗
脚本：
python "D:\BaiduNetdiskDownload\sd-scripts-main\finetune\merge_dd_tags_to_metadata.py" --full_path "D:\BaiduNetdiskDownload\sd-scripts-main\picture\sk\5_zkz" --in_json "D:\BaiduNetdiskDownload\sd-scripts-main\picture\sk\5_zkz\metadata.json" "D:\BaiduNetdiskDownload\sd-scripts-main\picture\sk\5_zkz\metadata.json"

python "D:\BaiduNetdiskDownload\sd-scripts-main\finetune\clean_captions_and_tags.py" "D:\BaiduNetdiskDownload\sd-scripts-main\picture\sk\5_zkz\metadata.json" "meta clean.json"

图1.4 清洗的元文件
4.训练阶段：
4.1 数据集配置如下：
[general]
enable_bucket = true                        # 是否使用Aspect Ratio Bucketing

[[datasets]]
resolution = 512                            # 训练分辨率
batch_size = 8                             # 批次大小

[[datasets.subsets]]
image_dir = 'D:\BaiduNetdiskDownload\sd-scripts-main\picture\sk\5_zkz'                     # 指定包含训练图像的文件夹
metadata_file = 'D:/BaiduNetdiskDownload/sd-scripts-main/picture/sk/5_zkz/metadata.json'
#class_tokens = 'sk'                # 指定标识符类
#caption_extension = '.txt'            # 若使用txt文件,更改此项
 num_repeats = 10         
4.2 训练脚本参数如下：
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


4.3 训练阶段：


4.4训练得到的Lora会在文件夹同一路径sk-Lora中保存



