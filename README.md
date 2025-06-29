# WD1.4 图像批量打标工具 (Gradio Web UI)

这是一个基于 Gradio 的图形化Web界面，用于对本地图片文件夹进行批量动漫风格的标签识别。它使用 ONNX Runtime 部署了 SmilingWolf/wd-v1-4-tagger 系列模型，并可利用 TensorRT 或 CUDA 后端进行高性能推理。

![Uploading image.png…]()


## ✨ 功能特性

- **图形化界面**：通过 Gradio 构建，无需命令行操作，简单易用。
- **多种模型支持**：内置多个 `SmilingWolf` 和 `deepghs` 系列的 ONNX Tagger 模型可供选择。
- **高性能推理**：使用 ONNX Runtime，并可配置优先使用 TensorRT 或 CUDA 执行，速度快。
- **批量处理**：支持对整个文件夹（包括子文件夹）的图片进行批量处理。
- **灵活的输出选项**：支持跳过、覆盖或追加（可去重）已有的标签文件。
- **丰富的标签控制**：可分别设置通用和角色标签的阈值，并提供多种标签格式化选项。

## ⚙️ 环境配置

本脚本依赖 Python 3.8+ 环境。建议使用 Conda 创建独立的虚拟环境。

```bash
conda create -n wd14 python=3.10.8
conda activate wd14
