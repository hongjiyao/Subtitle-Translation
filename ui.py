#!/usr/bin/env python3
"""
视频转字幕工具 - 简洁现代的Gradio UI实现
代码简单、功能丰富、方便调试
所有调试信息和错误信息都会打印到终端
"""
import os
import sys

# 确保在导入任何库之前设置HF-Mirror作为下载源
# 这些环境变量需要在导入其他库之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.dirname(__file__), "models")
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = os.path.join(os.path.dirname(__file__), "models")

import json
import gradio as gr
from config import DEFAULT_CONFIG, TEMP_DIR, OUTPUT_DIR
from utils.queue_manager import QueueManager

# 确保必要的目录存在
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 翻译模型映射
translator_models = {
    "m2m100_418M": "facebook/m2m100_418M",
    "m2m100_1.2B": "facebook/m2m100_1.2B"
}

# 模型选项
model_options = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
translator_options = list(translator_models.keys())

# 创建队列管理器实例
queue_manager = QueueManager()

# 参数保存和加载相关函数
PARAMS_FILE = os.path.join(os.path.dirname(__file__), "saved_params.json")

def save_params(model, translator, beam_size, vad_filter, word_timestamps, 
                condition_on_previous_text, translation_beam_size, 
                translation_max_length, translation_early_stopping, device):
    """保存当前参数设置
    
    Args:
        所有参数设置项
    
    Returns:
        str: 保存状态消息
    """
    try:
        params = {
            "model": model,
            "translator": translator,
            "beam_size": int(beam_size),
            "vad_filter": vad_filter,
            "word_timestamps": word_timestamps,
            "condition_on_previous_text": condition_on_previous_text,
            "translation_beam_size": int(translation_beam_size),
            "translation_max_length": int(translation_max_length),
            "translation_early_stopping": translation_early_stopping,
            "device": device
        }
        
        with open(PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        
        print(f"[参数保存] 成功保存参数到 {PARAMS_FILE}")
        return "参数保存成功！"
    except Exception as e:
        error_msg = f"保存参数时出错: {str(e)}"
        print(f"[错误信息] {error_msg}")
        return "保存失败：" + str(e)

def load_params():
    """加载保存的参数设置
    
    Returns:
        tuple: 所有参数值，如果没有保存的参数则返回None
    """
    try:
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                params = json.load(f)
            
            print(f"[参数加载] 成功从 {PARAMS_FILE} 加载参数")
            return (
                params.get("model"),
                params.get("translator"),
                params.get("beam_size"),
                params.get("vad_filter"),
                params.get("word_timestamps"),
                params.get("condition_on_previous_text"),
                params.get("translation_beam_size"),
                params.get("translation_max_length"),
                params.get("translation_early_stopping"),
                params.get("device")
            )
        else:
            print("[参数加载] 没有找到保存的参数文件")
            return None
    except Exception as e:
        error_msg = f"加载参数时出错: {str(e)}"
        print(f"[错误信息] {error_msg}")
        return None

def reset_to_default():
    """恢复默认参数设置
    
    Returns:
        tuple: 默认参数值
    """
    print("[参数重置] 恢复默认参数设置")
    return (
        DEFAULT_CONFIG["speech_recognition_model"],
        "m2m100_418M",
        DEFAULT_CONFIG["speech_recognition_params"]["beam_size"],
        DEFAULT_CONFIG["speech_recognition_params"]["vad_filter"],
        DEFAULT_CONFIG["speech_recognition_params"]["word_timestamps"],
        DEFAULT_CONFIG["speech_recognition_params"]["condition_on_previous_text"],
        DEFAULT_CONFIG["translation_params"]["beam_size"],
        DEFAULT_CONFIG["translation_params"]["max_length"],
        DEFAULT_CONFIG["translation_params"]["early_stopping"],
        DEFAULT_CONFIG["device"]
    )

# 处理单个视频文件
def process_single_video(video_file, model, translator, beam_size, vad_filter, 
                        word_timestamps, condition_on_previous_text, 
                        translation_beam_size, translation_max_length, 
                        translation_early_stopping, device):
    """处理单个视频文件
    
    Args:
        video_file: 视频文件路径
        model: 语音识别模型
        translator: 翻译模型
        beam_size: 语音识别beam size
        vad_filter: 是否启用VAD过滤
        word_timestamps: 是否启用单词时间戳
        condition_on_previous_text: 是否基于先前文本
        translation_beam_size: 翻译beam size
        translation_max_length: 翻译最大长度
        translation_early_stopping: 是否启用翻译早停
        device: 设备选择
    
    Yields:
        tuple: (下载路径, 状态消息, 进度值)
    """
    # 打印所有UI信息到终端
    print("\n" + "="*80)
    print("[UI信息] 用户执行了'处理单个文件'操作")
    print(f"[UI信息] 处理的视频文件: {video_file}")
    print(f"[UI信息] 使用的模型: {model}")
    print(f"[UI信息] 使用的翻译模型: {translator}")
    print(f"[UI信息] 使用的设备: {device}")
    print(f"[UI信息] Beam size: {beam_size}")
    print(f"[UI信息] VAD filter: {vad_filter}")
    print(f"[UI信息] Word timestamps: {word_timestamps}")
    print(f"[UI信息] Condition on previous text: {condition_on_previous_text}")
    print(f"[UI信息] Translation beam size: {translation_beam_size}")
    print(f"[UI信息] Translation max length: {translation_max_length}")
    print(f"[UI信息] Translation early stopping: {translation_early_stopping}")
    print("="*80)
    
    try:
        # 处理视频文件参数
        if isinstance(video_file, list) and video_file:
            video_file = video_file[0]
            print(f"[UI信息] 从列表中提取第一个视频文件: {video_file}")
        
        # 构建参数字典
        # 将翻译模型选项映射到完整的模型路径
        translator_model = translator_models.get(translator, translator)
        params = {
            'model': model,
            'translator': translator_model,
            'beam_size': int(beam_size),
            'vad_filter': vad_filter,
            'word_timestamps': word_timestamps,
            'condition_on_previous_text': condition_on_previous_text,
            'translation_beam_size': int(translation_beam_size),
            'translation_max_length': int(translation_max_length),
            'translation_early_stopping': translation_early_stopping,
            'device': device
        }
        print(f"[UI信息] 映射后的翻译模型: {translator_model}")
        
        # 处理开始时，文件还不存在，返回None作为下载路径
        yield None, "处理中...", 0
        
        # 实时更新进度
        yield None, "正在提取音频...", 10
        
        # 调用QueueManager的process_video方法
        success, message, output_path, logs = queue_manager.process_video(video_file, params)
        
        # 打印消息到终端
        print(f"[处理结果] {message}")
        
        # 最终更新
        if success and output_path and os.path.exists(output_path):
            print(f"[下载信息] 字幕文件已生成：{output_path}")
            yield output_path, "处理成功！", 100
        else:
            print(f"[下载信息] 字幕文件生成失败")
            yield None, "处理失败：" + message, 0
    except Exception as e:
        # 处理整体异常
        error_msg = f"处理视频时出错: {str(e)}"
        print(f"[错误信息] {error_msg}")
        yield None, "处理失败：" + str(e), 0

# 添加文件到队列
def add_to_queue(video_files, model, translator, beam_size, vad_filter, 
                 word_timestamps, condition_on_previous_text, 
                 translation_beam_size, translation_max_length, 
                 translation_early_stopping, device):
    """添加文件到队列
    
    Args:
        video_files: 视频文件路径列表
        model: 语音识别模型
        translator: 翻译模型
        beam_size: 语音识别beam size
        vad_filter: 是否启用VAD过滤
        word_timestamps: 是否启用单词时间戳
        condition_on_previous_text: 是否基于先前文本
        translation_beam_size: 翻译beam size
        translation_max_length: 翻译最大长度
        translation_early_stopping: 是否启用翻译早停
        device: 设备选择
    
    Returns:
        tuple: (队列数据, 状态消息, 队列统计)
    """
    # 打印所有UI信息到终端
    print("\n" + "="*80)
    print("[UI信息] 用户执行了'添加到队列'操作")
    print("[UI信息] 选择的视频文件:")
    if isinstance(video_files, list):
        for i, file in enumerate(video_files):
            print(f"  {i+1}. {file}")
    else:
        print(f"  {video_files}")
    print(f"[UI信息] 使用的模型: {model}")
    print(f"[UI信息] 使用的翻译模型: {translator}")
    print(f"[UI信息] 使用的设备: {device}")
    print(f"[UI信息] VAD filter: {vad_filter}")
    print(f"[UI信息] Word timestamps: {word_timestamps}")
    print(f"[UI信息] Condition on previous text: {condition_on_previous_text}")
    print(f"[UI信息] Translation early stopping: {translation_early_stopping}")
    print("="*80)
    
    try:
        # 确保video_files是列表
        if isinstance(video_files, str):
            video_files = [video_files]
        elif not isinstance(video_files, list):
            return [], "请选择视频文件", ""
        
        # 处理文件对象格式
        processed_files = []
        for file_item in video_files:
            if isinstance(file_item, str):
                processed_files.append(file_item)
            elif hasattr(file_item, 'name'):
                processed_files.append(file_item.name)
            elif isinstance(file_item, dict) and 'path' in file_item:
                processed_files.append(file_item['path'])
            elif isinstance(file_item, dict) and 'name' in file_item:
                processed_files.append(file_item['name'])
        
        # 使用处理后的文件列表
        video_files = processed_files
        
        # 再次检查是否有有效文件
        if not video_files:
            return [], "请选择视频文件", ""
        
        # 构建参数字典
        # 将翻译模型选项映射到完整的模型路径
        translator_model = translator_models.get(translator, translator)
        params = {
            'model': model,
            'translator': translator_model,
            'beam_size': int(beam_size),
            'vad_filter': vad_filter,
            'word_timestamps': word_timestamps,
            'condition_on_previous_text': condition_on_previous_text,
            'translation_beam_size': int(translation_beam_size),
            'translation_max_length': int(translation_max_length),
            'translation_early_stopping': translation_early_stopping,
            'device': device
        }
        print(f"[UI信息] 映射后的翻译模型: {translator_model}")
        
        # 使用QueueManager添加文件到队列
        added_count = queue_manager.add_to_queue(video_files, params)
        
        # 获取队列状态
        queue_items = queue_manager.get_queue()
        queue_data = [[item['filename'], item['status']] for item in queue_items]
        status_message = f"成功添加 {added_count} 个文件到队列"
        
        # 计算队列统计信息
        total_files = len(queue_items)
        waiting_files = sum(1 for item in queue_items if item['status'] == '等待中')
        processing_files = sum(1 for item in queue_items if item['status'] == '处理中')
        completed_files = sum(1 for item in queue_items if item['status'] == '已完成')
        failed_files = sum(1 for item in queue_items if item['status'] == '失败')
        
        queue_stats_message = f"总计: {total_files} 个文件\n" \
                            f"等待中: {waiting_files} 个\n" \
                            f"处理中: {processing_files} 个\n" \
                            f"已完成: {completed_files} 个\n" \
                            f"失败: {failed_files} 个"
        
        return queue_data, status_message, queue_stats_message
    except Exception as e:
        # 处理整体异常
        error_msg = f"添加文件到队列时出错: {str(e)}"
        print(f"[错误信息] {error_msg}")
        return [], "添加失败：" + str(e), ""

# 从队列删除文件
def remove_from_queue(index):
    """从队列删除文件
    
    Args:
        index: 要删除的队列项索引
    
    Returns:
        tuple: (队列数据, 状态消息, 队列统计)
    """
    # 打印所有UI信息到终端
    print("\n" + "="*80)
    print("[UI信息] 用户执行了'从队列删除'操作")
    print(f"[UI信息] 删除索引: {index}")
    print("="*80)
    
    try:
        # 处理不同类型的索引输入
        if isinstance(index, str):
            index = index.strip()
            if not index.isdigit():
                queue_data = get_queue_status()
                return queue_data, "请输入有效的数字索引", get_queue_stats()
            index = int(index)
        elif not isinstance(index, (int, float)):
            queue_data = get_queue_status()
            return queue_data, "索引必须是数字", get_queue_stats()
        else:
            index = int(index)
        
        # 使用QueueManager从队列删除文件
        queue_manager.remove_from_queue(index)
        
        # 获取更新后的队列状态
        queue_data = get_queue_status()
        return queue_data, f"已删除索引为 {index} 的文件", get_queue_stats()
    except Exception as e:
        # 处理整体异常
        error_msg = f"从队列删除文件时出错: {str(e)}"
        print(f"[错误信息] {error_msg}")
        queue_data = get_queue_status()
        return queue_data, "删除失败：" + str(e), get_queue_stats()

# 清空队列
def clear_queue():
    """清空队列
    
    Returns:
        tuple: (队列数据, 状态消息, 队列统计)
    """
    # 打印所有UI信息到终端
    print("\n" + "="*80)
    print("[UI信息] 用户执行了'清空队列'操作")
    print("="*80)
    
    try:
        # 使用QueueManager清空队列
        cleared_count = queue_manager.clear_queue()
        
        if cleared_count > 0:
            return [], f"队列已清空，共删除 {cleared_count} 个文件", get_queue_stats()
        else:
            return [], "队列为空", get_queue_stats()
    except Exception as e:
        # 处理异常
        error_msg = f"清空队列时出错: {str(e)}"
        print(f"[错误信息] {error_msg}")
        return [], "清空失败：" + str(e), get_queue_stats()

# 获取队列状态
def get_queue_status():
    """获取队列状态
    
    Returns:
        list: 队列数据
    """
    queue_items = queue_manager.get_queue()
    return [[item['filename'], item['status']] for item in queue_items]

# 获取队列统计信息
def get_queue_stats():
    """获取队列统计信息
    
    Returns:
        str: 队列统计信息
    """
    queue_items = queue_manager.get_queue()
    total_files = len(queue_items)
    waiting_files = sum(1 for item in queue_items if item['status'] == '等待中')
    processing_files = sum(1 for item in queue_items if item['status'] == '处理中')
    completed_files = sum(1 for item in queue_items if item['status'] == '已完成')
    failed_files = sum(1 for item in queue_items if item['status'] == '失败')
    
    return f"总计: {total_files} 个文件\n" \
           f"等待中: {waiting_files} 个\n" \
           f"处理中: {processing_files} 个\n" \
           f"已完成: {completed_files} 个\n" \
           f"失败: {failed_files} 个"

# 处理队列
def process_queue():
    """处理队列中的所有视频文件
    
    Yields:
        tuple: (队列数据, 状态消息, 队列统计)
    """
    # 打印所有UI信息到终端
    print("\n" + "="*80)
    print("[UI信息] 用户执行了'开始处理队列'操作")
    print("="*80)
    
    try:
        # 获取当前队列状态
        queue_items = queue_manager.get_queue()
        if not queue_items:
            yield [], "队列为空，没有文件可处理", get_queue_stats()
            return
        
        # 使用QueueManager的process_queue方法处理队列
        for result in queue_manager.process_queue():
            queue_data, message, logs, progress, status_text = result
            # 实时更新UI
            yield queue_data, status_text if status_text else "处理中...", get_queue_stats()
        
        # 处理完成
        queue_items = queue_manager.get_queue()
        queue_data = [[item['filename'], item['status']] for item in queue_items]
        status_message = f"队列处理完成，共处理了 {len(queue_items)} 个文件"
        print(f"[队列状态] {status_message}")
        yield queue_data, status_message, get_queue_stats()
    except Exception as e:
        # 处理整体异常
        error_message = f"处理队列时出错：{str(e)}"
        print(f"[错误信息] {error_message}")
        yield get_queue_status(), "处理失败：" + str(e), get_queue_stats()

# 创建Gradio界面
with gr.Blocks(title="视频转字幕工具") as demo:
    gr.Markdown("""
    # 视频转字幕工具
    简洁高效的视频转字幕解决方案
    """)
    
    # 主布局
    with gr.Tabs():
        # 处理单个文件标签页
        with gr.TabItem("处理单个文件"):
            with gr.Row():
                with gr.Column(scale=2):
                    # 视频文件输入
                    video_input = gr.File(
                        label="选择视频文件", 
                        file_types=["video"], 
                        type="filepath", 
                        file_count="multiple"
                    )
                    gr.Markdown("支持多种视频格式：mp4、avi、mov、mkv等")
                    
                    # 主要操作按钮
                    with gr.Row():
                        process_btn = gr.Button("处理单个文件", variant="primary", size="lg")
                    
                    # 状态消息和下载
                    with gr.Row():
                        status_message = gr.Textbox(
                            label="状态消息", 
                            interactive=False, 
                            placeholder="操作状态将显示在这里...",
                            lines=3
                        )
                    
                    # 下载按钮
                    download_btn = gr.File(
                        label="生成的字幕文件",
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 参数设置")
                    
                    # 语音识别模型
                    model = gr.Dropdown(
                        label="语音识别模型",
                        choices=model_options,
                        value=DEFAULT_CONFIG["speech_recognition_model"],
                        info="模型越大，准确率越高，但处理速度越慢"
                    )
                    
                    # 翻译模型
                    translator = gr.Dropdown(
                        label="翻译模型",
                        choices=translator_options,
                        value="m2m100_418M",
                        info="m2m100_418M速度快，m2m100_1.2B准确率高"
                    )
                    
                    # 设备选择
                    device = gr.Dropdown(
                        label="设备选择",
                        choices=["auto", "cpu", "cuda"],
                        value=DEFAULT_CONFIG["device"],
                        info="auto自动选择，cpu使用CPU，cuda使用GPU（如果可用）"
                    )
                    
                    # 语音识别参数
                    with gr.Accordion("语音识别参数", open=False):
                        # Beam size设置
                        beam_size = gr.Slider(
                            label="语音识别beam size",
                            value=DEFAULT_CONFIG["speech_recognition_params"]["beam_size"],
                            minimum=1,
                            maximum=10,
                            step=1,
                            info="搜索宽度，值越大准确率越高但速度越慢"
                        )
                        # VAD过滤
                        vad_filter = gr.Checkbox(
                            label="VAD过滤",
                            value=DEFAULT_CONFIG["speech_recognition_params"]["vad_filter"],
                            info="开启后会过滤掉非语音部分，提高识别准确率"
                        )
                        # 单词时间戳
                        word_timestamps = gr.Checkbox(
                            label="单词时间戳",
                            value=DEFAULT_CONFIG["speech_recognition_params"]["word_timestamps"],
                            info="开启后会为每个单词添加时间戳，文件会更大"
                        )
                        # 基于先前文本
                        condition_on_previous_text = gr.Checkbox(
                            label="基于先前文本",
                            value=DEFAULT_CONFIG["speech_recognition_params"]["condition_on_previous_text"],
                            info="开启后会利用上下文信息提高识别准确率"
                        )
                    
                    # 翻译参数
                    with gr.Accordion("翻译参数", open=False):
                        translation_beam_size = gr.Slider(
                            label="翻译beam size",
                            value=DEFAULT_CONFIG["translation_params"]["beam_size"],
                            minimum=1,
                            maximum=10,
                            step=1,
                            info="搜索宽度，值越大翻译质量越高但速度越慢"
                        )
                        
                        translation_max_length = gr.Slider(
                            label="翻译最大长度",
                            value=DEFAULT_CONFIG["translation_params"]["max_length"],
                            minimum=50,
                            maximum=1000,
                            step=50,
                            info="翻译结果的最大长度"
                        )
                        
                        # 翻译早停
                        translation_early_stopping = gr.Checkbox(
                            label="翻译早停",
                            value=DEFAULT_CONFIG["translation_params"]["early_stopping"],
                            info="开启后翻译过程会在找到合适结果后提前停止"
                        )
                    
                    # 参数控制按钮
                    with gr.Row():
                        save_params_btn = gr.Button("保存参数", size="sm")
                        load_params_btn = gr.Button("加载参数", size="sm")
                        reset_default_btn = gr.Button("恢复默认", size="sm")
        
        # 批量处理标签页
        with gr.TabItem("批量处理"):
            with gr.Row():
                with gr.Column(scale=2):
                    # 视频文件输入（支持多文件选择）
                    batch_video_input = gr.File(
                        label="选择视频文件", 
                        file_types=["video"], 
                        type="filepath", 
                        file_count="multiple"
                    )
                    gr.Markdown("支持多种视频格式：mp4、avi、mov、mkv等")
                    
                    # 队列控制按钮
                    with gr.Row():
                        add_to_queue_btn = gr.Button("添加到队列", size="sm")
                        remove_from_queue_btn = gr.Button("从队列删除", size="sm")
                        clear_queue_btn = gr.Button("清空队列", size="sm")
                        process_queue_btn = gr.Button("开始处理队列", variant="primary", size="sm")
                    
                    # 队列操作
                    with gr.Row():
                        remove_index = gr.Number(
                            label="删除索引", 
                            value=0, 
                            minimum=0, 
                            precision=0, 
                            step=1
                        )
                    
                    # 队列文件列表
                    queue_list = gr.Dataframe(
                        label="队列文件列表",
                        headers=["文件名", "状态"],
                        datatype=["str", "str"],
                        row_count=8,
                        interactive=False,
                        wrap=True
                    )
                    
                    # 队列状态消息
                    queue_status_message = gr.Textbox(
                        label="队列状态", 
                        interactive=False, 
                        placeholder="队列操作状态将显示在这里...",
                        lines=3
                    )
                    
                    # 队列统计信息
                    queue_stats = gr.Textbox(
                        label="队列统计", 
                        interactive=False, 
                        placeholder="队列统计信息将显示在这里..."
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 批量处理参数")
                    
                    # 语音识别模型
                    batch_model = gr.Dropdown(
                        label="语音识别模型",
                        choices=model_options,
                        value=DEFAULT_CONFIG["speech_recognition_model"],
                        info="模型越大，准确率越高，但处理速度越慢"
                    )
                    
                    # 翻译模型
                    batch_translator = gr.Dropdown(
                        label="翻译模型",
                        choices=translator_options,
                        value="m2m100_418M",
                        info="m2m100_418M速度快，m2m100_1.2B准确率高"
                    )
                    
                    # 设备选择
                    batch_device = gr.Dropdown(
                        label="设备选择",
                        choices=["auto", "cpu", "cuda"],
                        value=DEFAULT_CONFIG["device"],
                        info="auto自动选择，cpu使用CPU，cuda使用GPU（如果可用）"
                    )
                    
                    # 语音识别参数
                    with gr.Accordion("语音识别参数", open=False):
                        # Beam size设置
                        batch_beam_size = gr.Slider(
                            label="语音识别beam size",
                            value=DEFAULT_CONFIG["speech_recognition_params"]["beam_size"],
                            minimum=1,
                            maximum=10,
                            step=1,
                            info="搜索宽度，值越大准确率越高但速度越慢"
                        )
                        # VAD过滤
                        batch_vad_filter = gr.Checkbox(
                            label="VAD过滤",
                            value=DEFAULT_CONFIG["speech_recognition_params"]["vad_filter"],
                            info="开启后会过滤掉非语音部分，提高识别准确率"
                        )
                        # 单词时间戳
                        batch_word_timestamps = gr.Checkbox(
                            label="单词时间戳",
                            value=DEFAULT_CONFIG["speech_recognition_params"]["word_timestamps"],
                            info="开启后会为每个单词添加时间戳，文件会更大"
                        )
                        # 基于先前文本
                        batch_condition_on_previous_text = gr.Checkbox(
                            label="基于先前文本",
                            value=DEFAULT_CONFIG["speech_recognition_params"]["condition_on_previous_text"],
                            info="开启后会利用上下文信息提高识别准确率"
                        )
                    
                    # 翻译参数
                    with gr.Accordion("翻译参数", open=False):
                        batch_translation_beam_size = gr.Slider(
                            label="翻译beam size",
                            value=DEFAULT_CONFIG["translation_params"]["beam_size"],
                            minimum=1,
                            maximum=10,
                            step=1,
                            info="搜索宽度，值越大翻译质量越高但速度越慢"
                        )
                        
                        batch_translation_max_length = gr.Slider(
                            label="翻译最大长度",
                            value=DEFAULT_CONFIG["translation_params"]["max_length"],
                            minimum=50,
                            maximum=1000,
                            step=50,
                            info="翻译结果的最大长度"
                        )
                        
                        # 翻译早停
                        batch_translation_early_stopping = gr.Checkbox(
                            label="翻译早停",
                            value=DEFAULT_CONFIG["translation_params"]["early_stopping"],
                            info="开启后翻译过程会在找到合适结果后提前停止"
                        )
    
    # 添加进度条
    with gr.TabItem("处理单个文件"):
        with gr.Row():
            progress_bar = gr.Slider(
                label="处理进度",
                minimum=0,
                maximum=100,
                value=0,
                interactive=False,
                visible=True
            )
    
    # 设置事件
    process_btn.click(
        fn=process_single_video,
        inputs=[
            video_input,
            model,
            translator,
            beam_size,
            vad_filter,
            word_timestamps,
            condition_on_previous_text,
            translation_beam_size,
            translation_max_length,
            translation_early_stopping,
            device
        ],
        outputs=[
            download_btn,
            status_message,
            progress_bar
        ]
    )
    
    # 队列操作事件
    add_to_queue_btn.click(
        fn=add_to_queue,
        inputs=[
            batch_video_input,
            batch_model,
            batch_translator,
            batch_beam_size,
            batch_vad_filter,
            batch_word_timestamps,
            batch_condition_on_previous_text,
            batch_translation_beam_size,
            batch_translation_max_length,
            batch_translation_early_stopping,
            batch_device
        ],
        outputs=[
            queue_list,
            queue_status_message,
            queue_stats
        ]
    )
    
    # 从队列删除文件
    remove_from_queue_btn.click(
        fn=remove_from_queue,
        inputs=[remove_index],
        outputs=[
            queue_list,
            queue_status_message,
            queue_stats
        ]
    )
    
    # 清空队列
    clear_queue_btn.click(
        fn=clear_queue,
        inputs=[],
        outputs=[
            queue_list,
            queue_status_message,
            queue_stats
        ]
    )
    
    # 开始处理队列
    process_queue_btn.click(
        fn=process_queue,
        inputs=[],
        outputs=[
            queue_list,
            queue_status_message,
            queue_stats
        ]
    )
    
    # 参数控制事件
    save_params_btn.click(
        fn=save_params,
        inputs=[
            model,
            translator,
            beam_size,
            vad_filter,
            word_timestamps,
            condition_on_previous_text,
            translation_beam_size,
            translation_max_length,
            translation_early_stopping,
            device
        ],
        outputs=[status_message]
    )
    
    load_params_btn.click(
        fn=load_params,
        inputs=[],
        outputs=[
            model,
            translator,
            beam_size,
            vad_filter,
            word_timestamps,
            condition_on_previous_text,
            translation_beam_size,
            translation_max_length,
            translation_early_stopping,
            device
        ]
    )
    
    reset_default_btn.click(
        fn=reset_to_default,
        inputs=[],
        outputs=[
            model,
            translator,
            beam_size,
            vad_filter,
            word_timestamps,
            condition_on_previous_text,
            translation_beam_size,
            translation_max_length,
            translation_early_stopping,
            device
        ]
    )

# 启动Gradio界面
if __name__ == "__main__":
    print("启动视频转字幕工具界面...")
    print("请在浏览器中打开以下URL:")
    print("http://localhost:7868")
    print("\n所有调试信息、错误信息都会打印到终端")
    print("="*80)
    
    # 启用队列机制，支持实时更新
    demo.queue(
        max_size=50,  # 队列最大长度
        api_open=False  # 关闭API访问
    )
    # 启动界面
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7868,
        theme=gr.themes.Soft()
    )
