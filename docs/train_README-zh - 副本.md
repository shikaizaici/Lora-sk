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
  