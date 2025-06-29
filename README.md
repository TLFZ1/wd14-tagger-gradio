# WD1.4 图像批量打标工具 (Gradio Web UI)

这是一个基于 Gradio 的图形化Web界面，用于对本地图片文件夹进行批量动漫风格的标签识别。它使用 ONNX Runtime 部署了 SmilingWolf/wd-v1-4-tagger 系列模型，并可利用 TensorRT 或 CUDA 后端进行高性能推理。

![image](https://github.com/user-attachments/assets/f7d0b747-ddf1-455b-80c9-0447aaaeaedb)


## ✨ 功能特性

- **图形化界面**：通过 Gradio 构建，无需命令行操作，简单易用。
- **多种模型支持**：内置多个 `SmilingWolf` 和 `deepghs` 系列的 ONNX Tagger 模型可供选择。
- **高性能推理**：使用 ONNX Runtime，并可配置优先使用 TensorRT 或 CUDA 执行，速度快。
- **批量处理**：支持对整个文件夹（包括子文件夹）的图片进行批量处理。
- **灵活的输出选项**：支持跳过、覆盖或追加（可去重）已有的标签文件。
- **丰富的标签控制**：可分别设置通用和角色标签的阈值，并提供多种标签格式化选项。

## ⚙️ 环境配置

建议使用 Conda 创建独立的虚拟环境。

### 第一步：获取代码
首先，将本仓库克隆（下载）到您的本地电脑，并进入项目目录。
```bash
git clone https://github.com/TLFZ1/wd14-tagger-gradio.git
```

### 第二步：创建并激活 Conda 环境
我们为项目创建一个独立的 Python 环境，以避免与您电脑上其他项目的依赖产生冲突。
```bash
conda create -n wd14 python=3.10.8
conda activate wd14
```

### 第三步：安装依赖库
接着，使用 pip 安装运行此工具所必需的所有 Python 库。
```bash
cd .....\wd14-tagger-gradio
pip install -r requirements.txt
```
提示: 如果下载速度过慢，可以尝试使用国内镜像，例如：
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/


### 第四步：运行程序
最后，启动工具的图形化界面。
```bash
python wd14.py
```
