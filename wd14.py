# -*- coding: utf-8 -*-
# 导入必要的库
import argparse  # 用于解析命令行参数（在此脚本中未使用，但可能为未来扩展保留）
import csv  # 用于读写CSV文件，如此处的标签文件
import os  # 用于与操作系统交互，如文件路径操作
import gradio as gr  # Gradio库，用于快速创建Web UI界面
from pathlib import Path  # 面向对象的文件系统路径库，方便地处理文件和目录
import cv2  # OpenCV库，用于图像处理（在此脚本中未直接使用，但ONNX模型内部可能依赖）
import numpy as np  # NumPy库，用于高效的数值计算，特别是多维数组
import torch  # PyTorch库，主要用于数据加载（DataLoader）部分
from huggingface_hub import hf_hub_download  # 从Hugging Face Hub下载文件的函数
from PIL import Image  # Python Imaging Library (PIL) 的一个分支，用于打开、操作和保存多种图像文件格式
from tqdm import tqdm  # 用于显示进度条
import time  # 时间相关的函数
import gc  # 垃圾回收模块，用于手动管理内存
import pandas as pd  # Pandas库，用于数据分析和处理，如此处读取CSV
import onnxruntime as rt  # ONNX运行时库，用于执行ONNX格式的深度学习模型

# =================================================================================
# D E F A U L T   S E T T I N G S   A R E A  (默认设置区)
# =================================================================================
# 在这里修改变量，Gradio界面的默认值会自动更新

# --- 基础设置 ---
DEFAULT_INPUT_DIR = ""  # 默认的输入图片文件夹路径
DEFAULT_CAPTION_EXTENSION = ".txt"  # 生成的标签文件的默认扩展名
DEFAULT_RECURSIVE_SEARCH = True  # 是否默认递归搜索子文件夹中的图片
DEFAULT_FILE_EXIST_MODE = "Skip"  # 当标签文件已存在时的默认处理模式，可选: "Skip" (跳过), "Overwrite" (覆盖), "Append" (追加)

# --- 模型设置 ---
DEFAULT_MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"  # 默认使用的Hugging Face模型仓库ID
DEFAULT_MODEL_DIR = "./wd14_tagger_model"  # 下载的模型文件存放的本地目录

# --- 性能设置 ---
DEFAULT_BATCH_SIZE = 64  # 默认的批处理大小，一次处理多少张图片
# 关键提示：为了让“强制停止”按钮有效，必须将 Num Workers 设为 0。
# > 0 会显著提升速度，但会导致停止按钮失效（只能刷新页面来终止）。
DEFAULT_NUM_WORKERS = 6  # 默认的数据加载器工作进程数，用于并行加载数据

# --- 标签逻辑设置 ---
DEFAULT_GENERAL_THRESHOLD = 0.3  # 通用标签的置信度阈值，低于此值的标签将被忽略
DEFAULT_CHARACTER_THRESHOLD = 0.3  # 角色标签的置信度阈值
DEFAULT_ADD_RATING_TAGS = True  # 是否默认添加分级标签 (e.g., general, sensitive, questionable, explicit)
DEFAULT_REMOVE_UNDERSCORE = True  # 是否默认将标签中的下划线替换为空格
DEFAULT_ENABLE_GENERAL_TAGS = True  # 是否默认启用通用标签的生成
DEFAULT_ENABLE_CHARACTER_TAGS = True  # 是否默认启用角色标签的生成
DEFAULT_REMOVE_DUPLICATES_ON_APPEND = True # 新增：在追加模式下是否自动移除重复的标签

# =================================================================================
# E N D   O F   D E F A U L T   S E T T I N G S
# =================================================================================

# --- 全局常量和辅助函数 ---
# 可用的模型仓库列表，用于Gradio界面的下拉菜单
MODEL_REPOS_LIST = [
    "SmilingWolf/wd-swinv2-tagger-v3", "SmilingWolf/wd-convnext-tagger-v3", "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3", "SmilingWolf/wd-eva02-large-tagger-v3", "SmilingWolf/wd-v1-4-moat-tagger-v2",
    "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "SmilingWolf/wd-v1-4-convnext-tagger-v2", "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "SmilingWolf/wd-v1-4-vit-tagger-v2", "deepghs/idolsankaku-eva02-large-tagger-v1", "deepghs/idolsankaku-swinv2-tagger-v1"
]
MODEL_FILENAME = "model.onnx"  # 模型文件的标准名称
LABEL_FILENAME = "selected_tags.csv"  # 标签文件的标准名称
# Kaomoji (颜文字) 列表，这些标签中的下划线不会被替换
KAOMOJI_LIST = ["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"]

def glob_images_pathlib(dir_path, recursive):
    """
    使用 pathlib 查找指定目录下的所有图片文件。
    :param dir_path: 要搜索的目录路径。
    :param recursive: 是否递归搜索子目录。
    :return: 包含所有图片路径(Path对象)的列表。
    """
    dir_path = Path(dir_path)  # 将字符串路径转换为Path对象
    if not dir_path.is_dir(): return []  # 如果路径不是一个目录，返回空列表
    image_paths = []
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']  # 支持的图片文件扩展名
    glob_pattern = '**/*' if recursive else '*'  # 如果递归，则搜索所有子目录；否则只搜索当前目录
    for file_path in dir_path.glob(glob_pattern):
        # 检查文件是否是文件且扩展名在支持列表中
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image_paths.append(file_path)
    return image_paths

def preprocess_image(image: Image.Image, target_size: int):
    """
    预处理单张图片以符合模型输入要求。
    :param image: PIL.Image 对象。
    :param target_size: 模型输入的目标尺寸 (正方形)。
    :return: 预处理后的 NumPy 数组。
    """
    # 如果图片是RGBA模式（带透明通道），则创建一个白色背景并粘贴上去
    if image.mode == 'RGBA':
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(image, mask=image.split()[3]) # 使用alpha通道作为遮罩
        image = canvas
    else:
        image = image.convert("RGB") # 转换为RGB模式

    image_shape = image.size
    max_dim = max(image_shape) # 获取图片的最长边

    # 计算填充量，使图片变为正方形
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    # 创建一个白色背景的正方形画布，并将原图粘贴到中心
    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # 如果最长边不等于目标尺寸，则缩放图片
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.Resampling.BICUBIC)

    # 将PIL图片转换为NumPy数组，并转换为float32类型
    image_array = np.asarray(padded_image, dtype=np.float32)
    # 将颜色通道从RGB转换为BGR（模型通常需要这种格式）
    image_array = image_array[:, :, ::-1]
    return image_array

class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    """
    一个自定义的PyTorch数据集，用于加载和预处理图片。
    这允许使用DataLoader进行多进程并行加载，提高效率。
    """
    def __init__(self, image_paths, target_size):
        self.images = image_paths
        self.target_size = target_size

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.images)

    def __getitem__(self, idx):
        # 根据索引获取单个样本
        img_path = self.images[idx]
        try:
            image = Image.open(img_path)
            prepared_image = preprocess_image(image, self.target_size)
        except Exception as e:
            # 如果加载或处理图片时出错（如文件损坏），则打印错误并返回None
            print(f"无法加载图片: {img_path}, 错误: {e}")
            return None
        return (prepared_image, img_path)

def collate_fn_remove_corrupted(batch):
    """
    自定义的 collate_fn 函数，用于在DataLoader组合批次时，过滤掉值为None的项（损坏的图片）。
    """
    return list(filter(lambda x: x is not None, batch))

class Tagger:
    """
    封装了模型加载和推理逻辑的类。
    """
    def __init__(self, model_repo):
        self.model_repo = model_repo
        self.model_session, self.tag_names, self.model_target_size = None, None, None
        self.rating_indexes, self.general_indexes, self.character_indexes = [], [], []

    def load_model_if_needed(self, model_dir, force_download):
        """
        如果模型未加载，则从Hugging Face Hub下载模型和标签文件，并初始化ONNX运行时会话。
        """
        if self.model_session: return # 如果模型已加载，则直接返回
        # 从Hugging Face Hub下载标签CSV文件
        csv_path = hf_hub_download(repo_id=self.model_repo, filename=LABEL_FILENAME, cache_dir=model_dir, force_download=force_download)
        # 从Hugging Face Hub下载ONNX模型文件
        model_path = hf_hub_download(repo_id=self.model_repo, filename=MODEL_FILENAME, cache_dir=model_dir, force_download=force_download)
        
        # 使用pandas读取标签CSV文件
        dataframe = pd.read_csv(csv_path)
        # 处理标签名称：将下划线替换为空格，但保留颜文字的下划线
        name_series = dataframe["name"].map(lambda x: x.replace("_", " ") if x not in KAOMOJI_LIST else x)
        self.tag_names = name_series.tolist()
        
        # 根据标签类别（category）分离出不同类型的标签索引
        self.rating_indexes = list(np.where(dataframe["category"] == 9)[0])  # 分级标签
        self.general_indexes = list(np.where(dataframe["category"] == 0)[0]) # 通用标签
        self.character_indexes = list(np.where(dataframe["category"] == 4)[0]) # 角色标签
        
        print(f"正在加载 ONNX 模型: {model_path}")
        # 定义ONNX运行时的执行提供程序（Providers），按顺序尝试使用
        providers = [
            # 优先尝试TensorRT，开启FP16和缓存以获得最佳性能
            ('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': os.path.join(os.path.dirname(model_path), 'trt_cache')}),
            # 其次尝试CUDA
            'CUDAExecutionProvider',
            # 最后使用CPU
            'CPUExecutionProvider'
        ]
        # 创建ONNX运行时会话
        self.model_session = rt.InferenceSession(model_path, providers=providers)
        # 获取模型的输入尺寸
        _, self.model_target_size, _, _ = self.model_session.get_inputs()[0].shape
        print(f"模型加载成功！后端: {self.model_session.get_providers()[0]}, 输入尺寸: {self.model_target_size}x{self.model_target_size}")

    def process_single_prediction(self, probs, general_thresh, character_thresh, add_rating_tags, remove_underscore, enable_general_tags, enable_character_tags):
        """
        处理单个图片的模型预测结果（概率），并根据设置生成最终的标签字符串。
        """
        # 将标签名和对应的概率打包成元组列表
        labels = list(zip(self.tag_names, probs.astype(float)))
        
        # 提取分级标签及其概率
        rating_res = dict([labels[i] for i in self.rating_indexes])
        
        final_tags = []
        # 如果启用添加分级标签，并且存在分级结果，则添加概率最高的那个
        if add_rating_tags and rating_res:
            final_tags.append(max(rating_res, key=rating_res.get))
        
        # 如果启用通用标签
        if enable_general_tags:
            # 筛选出通用标签
            general_names = [labels[i] for i in self.general_indexes]
            # 根据通用阈值过滤，并创建字典
            general_res = dict(x for x in general_names if x[1] > general_thresh)
            # 按概率从高到低排序，并添加到最终结果
            sorted_general_tags = sorted(general_res.keys(), key=general_res.get, reverse=True)
            final_tags.extend(sorted_general_tags)
        
        # 如果启用角色标签
        if enable_character_tags:
            # 筛选出角色标签
            character_names = [labels[i] for i in self.character_indexes]
            # 根据角色阈值过滤，并创建字典
            character_res = dict(x for x in character_names if x[1] > character_thresh)
            # 按概率从高到低排序，并添加到最终结果
            final_tags.extend(sorted(character_res.keys(), key=character_res.get, reverse=True))

        # 如果启用移除下划线
        if remove_underscore:
            final_tags = [tag.replace("_", " ") if tag not in KAOMOJI_LIST else tag for tag in final_tags]
            
        # 使用 dict.fromkeys 来做一次最终的去重，以防万一模型输出本身有重复
        # 这几乎不会发生，但作为安全措施是好的。它能保持原始顺序。
        final_tags = list(dict.fromkeys(final_tags))
        # 将所有标签用 ", " 连接成一个字符串
        output_string = ", ".join(final_tags)
        return output_string

# 全局变量，用于缓存已加载的Tagger实例，避免重复加载模型
loaded_taggers_cache = {}

def run_tagging_ui(
    input_dir, model_repo, file_exist_mode,
    batch_size, num_workers, general_thresh, character_thresh,
    caption_extension, recursive,
    add_rating_tags, remove_underscore,
    enable_general_tags, enable_character_tags,
    remove_duplicates_on_append,
    progress=gr.Progress(track_tqdm=True) # Gradio进度条对象
):
    """
    Gradio界面点击“开始”后执行的核心函数。这是一个生成器函数，会逐步yield状态更新。
    """
    # 检查输入文件夹路径是否有效
    if not input_dir or not os.path.isdir(input_dir):
        raise gr.Error("错误：输入文件夹路径无效。")
    
    # 初始化状态信息和控制台输出
    status_message = "状态：正在准备模型..."
    console_output = ""
    yield status_message, console_output # 更新UI

    try:
        # 使用模型仓库ID作为缓存的键
        model_key = model_repo
        # 如果模型已在缓存中，则直接使用
        if model_key in loaded_taggers_cache:
            tagger = loaded_taggers_cache[model_key]
        else:
            # 否则，首次加载模型
            status_message = f"状态：首次加载模型 {model_repo}..."
            yield status_message, console_output
            tagger = Tagger(model_repo)
            tagger.load_model_if_needed(DEFAULT_MODEL_DIR, force_download=False)
            loaded_taggers_cache[model_key] = tagger # 加载后存入缓存
            
        status_message = "状态：正在扫描图片文件..."
        yield status_message, console_output
        # 扫描图片文件
        image_paths = glob_images_pathlib(input_dir, recursive=recursive)
        if not image_paths:
            yield "完成：未找到任何图片。", "未找到图片"
            return
        
        total_images = len(image_paths)
        status_message = f"状态：找到 {total_images} 张图片..."
        yield status_message, console_output
        # 创建数据集和数据加载器
        dataset = ImageLoadingPrepDataset(image_paths, tagger.model_target_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=int(num_workers), collate_fn=collate_fn_remove_corrupted, drop_last=False
        )
        
        # 初始化计数器
        processed_count = 0
        skipped_count = 0
        appended_count = 0
        overwritten_count = 0
        
        # 获取ONNX模型的输入和输出名称
        input_name = tagger.model_session.get_inputs()[0].name
        label_name = tagger.model_session.get_outputs()[0].name

        # 遍历数据加载器，批量处理图片
        for batch_data in progress.tqdm(dataloader, desc="批量处理中"):
            if not batch_data: continue # 如果批次为空（可能所有图片都损坏），则跳过
            
            # 从批次数据中分离出预处理后的图像数组和原始路径
            image_arrays = np.array([img for img, path in batch_data])
            paths = [path for img, path in batch_data]
            
            # 运行ONNX模型进行推理
            preds_batch = tagger.model_session.run([label_name], {input_name: image_arrays})[0]
            
            # 遍历批次中的每个结果
            for i in range(len(paths)):
                img_path, probs = paths[i], preds_batch[i]
                # 构建对应的标签文件路径
                caption_file_path = img_path.with_suffix(caption_extension)
                
                # 如果文件存在且模式为'Skip'，则跳过
                if caption_file_path.exists() and file_exist_mode == 'Skip':
                    skipped_count += 1
                    console_output += f"已跳过 (文件存在): {img_path.name}\n"
                    continue

                # 处理模型输出，生成标签字符串
                output_string = tagger.process_single_prediction(
                    probs, general_thresh, character_thresh, 
                    add_rating_tags, remove_underscore,
                    enable_general_tags, enable_character_tags
                )
                
                # 如果文件存在且模式为'Append'，则追加内容
                if caption_file_path.exists() and file_exist_mode == 'Append':
                    existing_text = caption_file_path.read_text(encoding='utf-8').strip()
                    final_text = ""
                    
                    # 如果启用了追加时去重
                    if remove_duplicates_on_append:
                        # 去重逻辑
                        existing_tags = [tag.strip() for tag in existing_text.split(',') if tag.strip()]
                        new_tags = [tag.strip() for tag in output_string.split(',') if tag.strip()]
                        
                        # 合并并使用 dict.fromkeys 保持顺序去重
                        combined_tags = existing_tags + new_tags
                        unique_tags = list(dict.fromkeys(combined_tags))
                        final_text = ", ".join(unique_tags)
                    else:
                        # 不去重的原始逻辑
                        if existing_text and output_string:
                            final_text = existing_text + ", " + output_string
                        else:
                            final_text = existing_text or output_string
                    
                    caption_file_path.write_text(final_text, encoding='utf-8')
                    appended_count += 1
                    log_msg = "已追加到 (去重):" if remove_duplicates_on_append else "已追加到:"
                    console_output += f"{log_msg} {img_path.name}\n"
                else:  # 'Overwrite'模式或文件不存在
                    caption_file_path.write_text(output_string, encoding='utf-8')
                    overwritten_count += 1
                    console_output += f"已处理 (创建/覆盖): {img_path.name}\n"

            # 更新处理进度
            processed_count = appended_count + overwritten_count
            current_total = processed_count + skipped_count
            yield f"状态：处理中...({current_total}/{total_images})", console_output

        # 最终总结信息
        final_summary = f"完成！共写入 {overwritten_count} 个新文件, 追加 {appended_count} 个文件, 跳过 {skipped_count} 个文件。"
        yield final_summary, console_output
    except Exception as e:
        # 捕获并报告任何在处理过程中发生的错误
        raise gr.Error(f"处理过程中发生错误: {e}")

def create_ui():
    """
    创建并返回Gradio用户界面。
    """
    with gr.Blocks(title="wd1.4图像打标工具") as demo:
        gr.Markdown("# wd1.4图像打标工具")
        gr.Markdown("融合了**高性能架构**与**高级文件和标签处理逻辑**。您可以在脚本顶部修改各项默认值。")

        with gr.Row():
            # 左侧主设置区域
            with gr.Column(scale=2):
                gr.Markdown("### **1. 基础设置**")
                input_dir = gr.Textbox(label="输入图片文件夹路径", value=DEFAULT_INPUT_DIR, placeholder="例如: D:\\dataset\\my_images")
                
                file_exist_mode_input = gr.Radio(
                    label="对于已存在的标签文件",
                    choices=["Skip", "Overwrite", "Append"],
                    value=DEFAULT_FILE_EXIST_MODE,
                    info="选择如何处理已存在的同名.txt文件"
                )
                
                recursive_search_input = gr.Checkbox(label="递归搜索子文件夹", value=DEFAULT_RECURSIVE_SEARCH)
                caption_extension = gr.Textbox(label="标签文件扩展名", value=DEFAULT_CAPTION_EXTENSION)

                gr.Markdown("### **2. 模型设置**")
                model_repo_input = gr.Dropdown(MODEL_REPOS_LIST, value=DEFAULT_MODEL_REPO, label="选择打标模型")
                
                gr.Markdown("### **3. 性能设置**")
                batch_size_input = gr.Slider(1, 512, value=DEFAULT_BATCH_SIZE, step=1, label="批处理大小 (Batch Size)")
                num_workers_input = gr.Slider(
                    0, 32, value=DEFAULT_NUM_WORKERS, step=1, 
                    label="数据加载器工作进程数 (Num Workers)",
                    info="重要：为了让“强制停止”按钮有效，此值必须为 0。大于0会加速，但无法中途停止。"
                )

                # 可折叠的精细化设置区域
                with gr.Accordion("精细化标签设置", open=True):
                    general_thresh_input = gr.Slider(0, 1, value=DEFAULT_GENERAL_THRESHOLD, step=0.01, label="通用标签阈值")
                    character_thresh_input = gr.Slider(0, 1, value=DEFAULT_CHARACTER_THRESHOLD, step=0.01, label="角色标签阈值")
                    
                    gr.Markdown("---")
                    add_rating_tags_input = gr.Checkbox(label="添加分级标签", value=DEFAULT_ADD_RATING_TAGS)
                    remove_underscore_input = gr.Checkbox(label="移除标签中的下划线", value=DEFAULT_REMOVE_UNDERSCORE)
                    remove_duplicates_input = gr.Checkbox(label="追加模式下自动去重", value=DEFAULT_REMOVE_DUPLICATES_ON_APPEND, info="合并新旧标签时，移除重复的标签。")
                    
                    with gr.Row():
                        enable_general_tags_input = gr.Checkbox(label="启用通用标签", value=DEFAULT_ENABLE_GENERAL_TAGS)
                        enable_character_tags_input = gr.Checkbox(label="启用角色标签", value=DEFAULT_ENABLE_CHARACTER_TAGS)
                
                with gr.Row():
                    start_button = gr.Button("开始批量打标", variant="primary", size="lg")
                    stop_button = gr.Button("强制停止", variant="stop", size="lg", visible=False) # 默认不可见

            # 右侧状态和日志区域
            with gr.Column(scale=1):
                gr.Markdown("### **4. 运行状态**")
                status_output = gr.Textbox(label="当前状态", interactive=False, lines=2, max_lines=2)
                console_output = gr.Textbox(label="处理日志", interactive=False, lines=20)
        
        # --- 事件处理逻辑 ---
        
        # 将所有输入UI组件收集到一个列表中，方便传递给处理函数
        run_inputs = [
            input_dir, model_repo_input, file_exist_mode_input,
            batch_size_input, num_workers_input, general_thresh_input, character_thresh_input,
            caption_extension, recursive_search_input,
            add_rating_tags_input, remove_underscore_input,
            enable_general_tags_input, enable_character_tags_input,
            remove_duplicates_input
        ]

        def start_running():
            """点击开始按钮时，更新按钮的可见性。"""
            return {
                start_button: gr.update(visible=False),
                stop_button: gr.update(visible=True)
            }
        
        def stop_running(status_message=""):
            """任务结束或手动停止时，恢复按钮的可见性，并可选择更新状态消息。"""
            update_dict = {
                start_button: gr.update(visible=True),
                stop_button: gr.update(visible=False)
            }
            if status_message:
                update_dict[status_output] = gr.update(value=status_message)
            return update_dict

        # “开始”按钮的点击事件链
        process_event = start_button.click(
            fn=start_running, # 第一步：调用start_running函数
            outputs=[start_button, stop_button] # 更新这两个按钮的状态
        ).then(
            fn=run_tagging_ui, # 第二步：调用核心处理函数
            inputs=run_inputs, # 传入所有输入UI组件的值
            outputs=[status_output, console_output] # 将函数的输出流式更新到这两个文本框
        ).then(
            fn=lambda: stop_running(), # 第三步：处理函数结束后，调用stop_running恢复按钮状态
            outputs=[start_button, stop_button]
        )
        
        # “停止”按钮的点击事件
        stop_button.click(
            fn=lambda: stop_running("状态：任务已手动停止。"), # 调用stop_running并设置自定义消息
            outputs=[start_button, stop_button, status_output], # 更新按钮和状态文本框
            cancels=[process_event] # 关键：取消正在运行的process_event事件链
        )
        
        return demo

# 主程序入口
if __name__ == "__main__":
    ui = create_ui() # 创建UI实例
    ui.launch(inbrowser=True) # 启动Gradio应用，并在浏览器中打开新标签页