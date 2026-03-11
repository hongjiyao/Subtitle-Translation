#!/usr/bin/env python3
"""
视频转字幕工具 - 简洁现代的Gradio UI实现
代码简单、功能丰富、方便调试
所有调试信息和错误信息都会打印到终端
"""
import os
import sys
import datetime

# 确保在导入任何库之前设置HF-Mirror作为下载源
# 这些环境变量需要在导入其他库之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.dirname(__file__), "models")
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "models")

# 导入必要的库
import torch
import warnings

# 禁用特定的警告
warnings.filterwarnings("ignore", category=UserWarning, message="Passing `gradient_checkpointing` to a config initialization is deprecated")

# 启用TF32以提高性能
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 导入必要的库
import json
import gradio as gr
from config import DEFAULT_CONFIG, TEMP_DIR, OUTPUT_DIR
from utils.queue_manager import QueueManager

# 确保必要的目录存在
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局配置
class Config:
    """应用配置类"""
    # 翻译模型映射
    translator_models = {
        "m2m100_418M": "facebook/m2m100_418M",
        "m2m100_1.2B": "facebook/m2m100_1.2B"
    }
    
    # 模型选项
    model_options = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    translator_options = list(translator_models.keys())
    
    # 支持的语言选项 (Whisper和M2M100支持的常见语言)
    language_options = [
        ("auto", "自动检测"),
        ("zh", "中文"),
        ("en", "英语"),
        ("ja", "日语"),
        ("ko", "韩语"),
        ("fr", "法语"),
        ("de", "德语"),
        ("es", "西班牙语"),
        ("ru", "俄语"),
        ("ar", "阿拉伯语"),
        ("hi", "印地语"),
        ("pt", "葡萄牙语"),
        ("it", "意大利语"),
        ("nl", "荷兰语"),
        ("pl", "波兰语")
    ]
    
    # 语言代码到名称的映射
    language_names = dict(language_options)
    
    # 默认语言设置
    default_source_language = "auto"
    default_target_language = "zh"
    
    # 默认语言选择模式 (auto_detect/manual)
    default_language_mode = "auto_detect"
    
    # 参数保存文件
    params_file = os.path.join(os.path.dirname(__file__), "saved_params.json")

# 工具函数
class Utils:
    """工具函数类"""
    @staticmethod
    def timestamp_print(message):
        """带时间戳的打印函数"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    @staticmethod
    def process_video_files(video_files):
        """处理视频文件输入，支持多种格式"""
        # 确保video_files是列表
        if isinstance(video_files, str):
            video_files = [video_files]
        elif not isinstance(video_files, list):
            return []
        
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
        
        return processed_files

# 全局实例
config = Config()
utils = Utils()
queue_manager = QueueManager()

# 参数管理
class ParameterManager:
    """参数管理类"""
    @staticmethod
    def save_params(model, translator, beam_size, vad_filter, word_timestamps, 
                   condition_on_previous_text, translation_beam_size, 
                   translation_max_length, device, 
                   source_language, target_language, language_mode,
                   speech_batch_size, translation_batch_size, vad_threshold, 
                   vad_min_speech, vad_max_speech, vad_min_silence):
        """保存当前参数设置"""
        try:
            # 转换语言模式为内部值
            internal_language_mode = "auto_detect" if language_mode == "自动检测" else "manual"
            
            # 安全地转换参数
            def safe_int(value, default):
                try:
                    return int(value) if value else default
                except (ValueError, TypeError):
                    return default
            
            beam_size = safe_int(beam_size, DEFAULT_CONFIG["speech_recognition_params"]["beam_size"])
            translation_beam_size = safe_int(translation_beam_size, DEFAULT_CONFIG["translation_params"]["beam_size"])
            translation_max_length = safe_int(translation_max_length, DEFAULT_CONFIG["translation_params"]["max_length"])
            speech_batch_size = safe_int(speech_batch_size, DEFAULT_CONFIG["whisperx_params"].get("batch_size", 16))
            translation_batch_size = safe_int(translation_batch_size, DEFAULT_CONFIG["whisperx_params"].get("batch_size", 16))
            
            params = {
                "model": model,
                "translator": translator,
                "beam_size": beam_size,
                "vad_filter": vad_filter,
                "word_timestamps": word_timestamps,
                "condition_on_previous_text": condition_on_previous_text,
                "translation_beam_size": translation_beam_size,
                "translation_max_length": translation_max_length,
                "device": device,
                "source_language": source_language,
                "target_language": target_language,
                "language_mode": internal_language_mode,
                "use_whisperx": True,  # 默认启用WhisperX
                "speech_batch_size": speech_batch_size,
                "translation_batch_size": translation_batch_size,
                "vad_threshold": vad_threshold,
                "vad_min_speech": vad_min_speech,
                "vad_max_speech": vad_max_speech,
                "vad_min_silence": vad_min_silence
            }
            
            with open(config.params_file, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            
            utils.timestamp_print(f"[参数保存] 成功保存参数到 {config.params_file}")
            return "参数保存成功！"
        except Exception as e:
            error_msg = f"保存参数时出错: {str(e)}"
            utils.timestamp_print(f"[错误信息] {error_msg}")
            return "保存失败：" + str(e)
    
    @staticmethod
    def load_params():
        """加载保存的参数设置"""
        try:
            if os.path.exists(config.params_file):
                with open(config.params_file, "r", encoding="utf-8") as f:
                    params = json.load(f)
                
                utils.timestamp_print(f"[参数加载] 成功从 {config.params_file} 加载参数")
                # 转换语言模式为UI值
                internal_language_mode = params.get("language_mode", config.default_language_mode)
                ui_language_mode = "自动检测" if internal_language_mode == "auto_detect" else "手动选择"
                
                return (
                    params.get("model"),
                    params.get("translator"),
                    params.get("beam_size"),
                    params.get("vad_filter"),
                    params.get("word_timestamps"),
                    params.get("condition_on_previous_text"),
                    params.get("translation_beam_size"),
                    params.get("translation_max_length"),
                    params.get("device"),
                    params.get("source_language", config.default_source_language),
                    params.get("target_language", config.default_target_language),
                    ui_language_mode,
                    params.get("speech_batch_size", 16),
                    params.get("translation_batch_size", 16),
                    params.get("vad_threshold", 0.5),
                    params.get("vad_min_speech", 250),
                    params.get("vad_max_speech", 30),
                    params.get("vad_min_silence", 100)
                )
            else:
                utils.timestamp_print("[参数加载] 没有找到保存的参数文件")
                return None
        except Exception as e:
            error_msg = f"加载参数时出错: {str(e)}"
            utils.timestamp_print(f"[错误信息] {error_msg}")
            return None
    
    @staticmethod
    def reset_to_default():
        """恢复默认参数设置"""
        utils.timestamp_print("[参数重置] 恢复默认参数设置")
        # 转换语言模式为UI值
        ui_language_mode = "自动检测" if config.default_language_mode == "auto_detect" else "手动选择"
        return (
                DEFAULT_CONFIG["speech_recognition_model"],
                "m2m100_418M",
                DEFAULT_CONFIG["speech_recognition_params"]["beam_size"],
                DEFAULT_CONFIG["speech_recognition_params"]["vad_filter"],
                DEFAULT_CONFIG["speech_recognition_params"]["word_timestamps"],
                DEFAULT_CONFIG["speech_recognition_params"]["condition_on_previous_text"],
                DEFAULT_CONFIG["translation_params"]["beam_size"],
                DEFAULT_CONFIG["translation_params"]["max_length"],
                DEFAULT_CONFIG["device"],
                config.default_source_language,
                config.default_target_language,
                ui_language_mode,
                DEFAULT_CONFIG["whisperx_params"].get("batch_size", 16),
                DEFAULT_CONFIG["whisperx_params"].get("batch_size", 16),
                0.5,  # 默认VAD阈值
                250,  # 默认最小语音持续时间
                30,   # 默认最大语音持续时间
                100   # 默认最小沉默持续时间
            )

# 队列管理
class QueueManagerUI:
    """队列管理UI类"""
    @staticmethod
    def add_to_queue(video_files, model, translator, beam_size, vad_filter, 
                    word_timestamps, condition_on_previous_text, 
                    translation_beam_size, translation_max_length, 
                    device, source_language, target_language, language_mode,
                    speech_batch_size, translation_batch_size, vad_threshold, 
                    vad_min_speech, vad_max_speech, vad_min_silence):
        """添加文件到队列"""
        # 打印所有UI信息到终端
        utils.timestamp_print("\n" + "="*80)
        utils.timestamp_print("[UI信息] 用户执行了'添加到队列'操作")
        utils.timestamp_print("[UI信息] 选择的视频文件:")
        
        # 处理视频文件
        processed_files = utils.process_video_files(video_files)
        if not processed_files:
            return [], "请选择视频文件"
        
        for i, file in enumerate(processed_files):
            utils.timestamp_print(f"  {i+1}. {file}")
        
        utils.timestamp_print(f"[UI信息] 使用的模型: {model}")
        utils.timestamp_print(f"[UI信息] 使用的翻译模型: {translator}")
        utils.timestamp_print(f"[UI信息] 使用的设备: {device}")
        utils.timestamp_print(f"[UI信息] VAD filter: {vad_filter}")
        utils.timestamp_print(f"[UI信息] Word timestamps: {word_timestamps}")
        utils.timestamp_print(f"[UI信息] Condition on previous text: {condition_on_previous_text}")
        utils.timestamp_print(f"[UI信息] 源语言: {source_language} ({config.language_names.get(source_language, source_language)})")
        utils.timestamp_print(f"[UI信息] 目标语言: {target_language} ({config.language_names.get(target_language, target_language)})")
        utils.timestamp_print(f"[UI信息] 语言模式: {language_mode}")
        utils.timestamp_print(f"[UI信息] 语音模型批处理大小: {speech_batch_size}")
        utils.timestamp_print(f"[UI信息] 翻译模型批处理大小: {translation_batch_size}")
        utils.timestamp_print(f"[UI信息] VAD阈值: {vad_threshold}")
        utils.timestamp_print(f"[UI信息] 最小语音持续时间: {vad_min_speech}ms")
        utils.timestamp_print(f"[UI信息] 最大语音持续时间: {vad_max_speech}s")
        utils.timestamp_print(f"[UI信息] 最小沉默持续时间: {vad_min_silence}ms")
        utils.timestamp_print("="*80)
        
        try:
            # 转换语言模式为内部值
            internal_language_mode = "auto_detect" if language_mode == "自动检测" else "manual"
            
            # 构建参数字典
            # 将翻译模型选项映射到完整的模型路径
            translator_model = config.translator_models.get(translator, translator)
            
            # 安全地转换参数
            def safe_int(value, default):
                try:
                    return int(value) if value else default
                except (ValueError, TypeError):
                    return default
            
            beam_size = safe_int(beam_size, DEFAULT_CONFIG["speech_recognition_params"]["beam_size"])
            translation_beam_size = safe_int(translation_beam_size, DEFAULT_CONFIG["translation_params"]["beam_size"])
            translation_max_length = safe_int(translation_max_length, DEFAULT_CONFIG["translation_params"]["max_length"])
            speech_batch_size = safe_int(speech_batch_size, DEFAULT_CONFIG["whisperx_params"].get("batch_size", 16))
            translation_batch_size = safe_int(translation_batch_size, DEFAULT_CONFIG["whisperx_params"].get("batch_size", 16))
            
            params = {
                'model': model,
                'translator': translator_model,
                'beam_size': beam_size,
                'vad_filter': vad_filter,
                'word_timestamps': word_timestamps,
                'condition_on_previous_text': condition_on_previous_text,
                'translation_beam_size': translation_beam_size,
                'translation_max_length': translation_max_length,
                'device': device,
                'source_language': source_language,
                'target_language': target_language,
                'language_mode': internal_language_mode,
                'use_whisperx': True,  # 默认启用WhisperX
                'speech_batch_size': speech_batch_size,
                'translation_batch_size': translation_batch_size,
                'vad_threshold': vad_threshold,
                'vad_min_speech_duration_ms': vad_min_speech,
                'vad_max_speech_duration_s': vad_max_speech,
                'vad_min_silence_duration_ms': vad_min_silence
            }
            utils.timestamp_print(f"[UI信息] 映射后的翻译模型: {translator_model}")
            
            # 使用QueueManager添加文件到队列
            added_count = queue_manager.add_to_queue(processed_files, params)
            
            # 获取队列状态
            queue_items = queue_manager.get_queue()
            queue_data = QueueManagerUI.get_queue_status()
            status_message = f"成功添加 {added_count} 个文件到队列"
            
            return queue_data, status_message
        except Exception as e:
            # 处理整体异常
            error_msg = f"添加文件到队列时出错: {str(e)}"
            utils.timestamp_print(f"[错误信息] {error_msg}")
            return [], "添加失败：" + str(e)
    
    @staticmethod
    def remove_from_queue(index):
        """从队列删除文件"""
        # 打印所有UI信息到终端
        utils.timestamp_print("\n" + "="*80)
        utils.timestamp_print("[UI信息] 用户执行了'从队列删除'操作")
        utils.timestamp_print(f"[UI信息] 删除索引: {index}")
        utils.timestamp_print("="*80)
        
        try:
            # 处理不同类型的索引输入
            if isinstance(index, str):
                index = index.strip()
                if not index.isdigit():
                    queue_data = QueueManagerUI.get_queue_status()
                    return queue_data, "请输入有效的数字索引"
                index = int(index)
            elif not isinstance(index, (int, float)):
                queue_data = QueueManagerUI.get_queue_status()
                return queue_data, "索引必须是数字"
            else:
                index = int(index)
            
            # 使用QueueManager从队列删除文件
            queue_manager.remove_from_queue(index)
            
            # 获取更新后的队列状态
            queue_data = QueueManagerUI.get_queue_status()
            return queue_data, f"已删除索引为 {index} 的文件"
        except Exception as e:
            # 处理整体异常
            error_msg = f"从队列删除文件时出错: {str(e)}"
            utils.timestamp_print(f"[错误信息] {error_msg}")
            queue_data = QueueManagerUI.get_queue_status()
            return queue_data, "删除失败：" + str(e)
    
    @staticmethod
    def clear_queue():
        """清空队列"""
        # 打印所有UI信息到终端
        utils.timestamp_print("\n" + "="*80)
        utils.timestamp_print("[UI信息] 用户执行了'清空队列'操作")
        utils.timestamp_print("="*80)
        
        try:
            # 使用QueueManager清空队列
            cleared_count = queue_manager.clear_queue()
            
            if cleared_count > 0:
                return [], f"队列已清空，共删除 {cleared_count} 个文件"
            else:
                return [], "队列为空"
        except Exception as e:
            # 处理异常
            error_msg = f"清空队列时出错: {str(e)}"
            utils.timestamp_print(f"[错误信息] {error_msg}")
            return [], "清空失败：" + str(e)
    
    @staticmethod
    def get_queue_status():
        """获取队列状态"""
        queue_items = queue_manager.get_queue()
        return [[item['filename'], item['status']] for item in queue_items]
    
    @staticmethod
    def process_queue():
        """处理队列中的所有视频文件"""
        # 打印所有UI信息到终端
        utils.timestamp_print("\n" + "="*80)
        utils.timestamp_print("[UI信息] 用户执行了'开始处理队列'操作")
        utils.timestamp_print("="*80)
        
        try:
            # 获取当前队列状态
            queue_items = queue_manager.get_queue()
            if not queue_items:
                yield [], "队列为空，没有文件可处理"
                return
            
            # 使用QueueManager的process_queue方法处理队列
            for result in queue_manager.process_queue():
                queue_data, message, logs, progress, status_text = result
                # 实时更新UI
                yield queue_data, status_text if status_text else "处理中..."
            
            # 处理完成
            queue_items = queue_manager.get_queue()
            queue_data = QueueManagerUI.get_queue_status()
            status_message = f"队列处理完成，共处理了 {len(queue_items)} 个文件"
            utils.timestamp_print(f"[队列状态] {status_message}")
            yield queue_data, status_message
        except Exception as e:
            # 处理整体异常
            error_message = f"处理队列时出错：{str(e)}"
            utils.timestamp_print(f"[错误信息] {error_message}")
            yield QueueManagerUI.get_queue_status(), "处理失败：" + str(e)

# 语言管理
class LanguageManager:
    """语言管理类"""
    @staticmethod
    def update_language_mode(source_language):
        """当用户选择特定语言时，自动切换语言模式为手动选择"""
        if source_language != "auto":
            utils.timestamp_print(f"[语言设置] 检测到手动选择语言: {source_language}，切换到手动模式")
            return "手动选择"
        return "自动检测"
    
    @staticmethod
    def confirm_language_settings(source_language, target_language, language_mode):
        """确认语言设置并更新显示"""
        source_name = config.language_names.get(source_language, source_language)
        target_name = config.language_names.get(target_language, target_language)
        mode_display = language_mode  # 直接使用UI值，因为它已经是"自动检测"或"手动选择"
        display_text = f"语言模式: {mode_display}\n源语言: {source_language} ({source_name})\n目标语言: {target_language} ({target_name})"
        utils.timestamp_print(f"[语言设置] 已确认语言设置: {display_text}")
        return display_text

# 创建Gradio界面
class VideoSubtitleUI:
    """视频字幕工具UI类"""
    def __init__(self):
        # 启动系统
        print("启动视频字幕工具...")
        
        self.demo = self.create_ui()
    
    def create_ui(self):
        """创建Gradio界面"""
        with gr.Blocks(title="视频转字幕工具") as demo:
            gr.Markdown("""
            # 视频转字幕工具
            简洁高效的视频转字幕解决方案
            """)
            
            # 主布局
            with gr.Row():
                # 左侧：文件管理和队列控制
                with gr.Column(scale=2, variant="panel"):
                    # 视频文件输入（支持多文件选择和拖拽）
                    batch_video_input = gr.File(
                        label="视频文件", 
                        file_types=["video"], 
                        type="filepath", 
                        file_count="multiple",
                        elem_id="video-input"
                    )
                    
                    # 队列控制按钮
                    with gr.Row(equal_height=True, elem_id="queue-controls"):
                        add_to_queue_btn = gr.Button("添加到队列", variant="secondary")
                        remove_from_queue_btn = gr.Button("从队列删除", variant="secondary")
                        clear_queue_btn = gr.Button("清空队列", variant="secondary")
                        process_queue_btn = gr.Button("开始处理", variant="primary")
                    
                    # 队列操作
                    with gr.Row(elem_id="remove-index-row"):
                        remove_index = gr.Number(
                            label="删除索引", 
                            value=0, 
                            minimum=0, 
                            precision=0, 
                            step=1
                        )
                    
                    # 队列文件列表
                    queue_list = gr.Dataframe(
                        label="处理队列",
                        headers=["文件名", "状态"],
                        datatype=["str", "str"],
                        row_count=8,
                        interactive=False,
                        wrap=True,
                        elem_classes=["queue-list"],
                        elem_id="queue-list"
                    )
                    
                    # 队列状态消息
                    queue_status_message = gr.Textbox(
                        label="操作状态", 
                        interactive=False, 
                        placeholder="操作状态将显示在这里...",
                        lines=2,
                        elem_id="status-message"
                    )
                    
                # 右侧：参数设置
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### 参数设置")
                    
                    # 模型选择
                    with gr.Tabs(elem_id="parameter-tabs"):
                        with gr.TabItem("基本设置"):
                            # 语音识别模型
                            batch_model = gr.Dropdown(
                                label="识别模型",
                                choices=config.model_options,
                                value=DEFAULT_CONFIG["speech_recognition_model"],
                                info="模型越大，准确率越高，但处理速度越慢",
                                elem_id="model-select"
                            )
                            
                            # 翻译模型
                            batch_translator = gr.Dropdown(
                                label="翻译模型",
                                choices=config.translator_options,
                                value="m2m100_418M",
                                info="m2m100_418M速度快，m2m100_1.2B准确率高",
                                elem_id="translator-select"
                            )
                            
                            # 设备选择
                            batch_device = gr.Dropdown(
                                label="设备",
                                choices=["auto", "cpu", "cuda"],
                                value=DEFAULT_CONFIG["device"],
                                info="auto自动选择，cpu使用CPU，cuda使用GPU",
                                elem_id="device-select"
                            )
                            
                            # 语言设置
                            with gr.Accordion("语言设置", open=True, elem_id="language-settings"):
                                # 语言模式选择
                                batch_language_mode = gr.Radio(
                                    label="语言模式",
                                    choices=["自动检测", "手动选择"],
                                    value="自动检测" if config.default_language_mode == "auto_detect" else "手动选择",
                                    elem_id="language-mode"
                                )
                                
                                # 源语言选择
                                batch_source_language = gr.Dropdown(
                                    label="源语言",
                                    choices=[lang[0] for lang in config.language_options],
                                    value=config.default_source_language,
                                    info="'auto'表示自动检测",
                                    elem_id="source-language"
                                )
                                
                                # 目标语言选择
                                batch_target_language = gr.Dropdown(
                                    label="目标语言",
                                    choices=[lang[0] for lang in config.language_options],
                                    value=config.default_target_language,
                                    elem_id="target-language"
                                )
                        
                        with gr.TabItem("高级设置"):
                            # 语音识别参数
                            with gr.Accordion("识别参数", open=False, elem_id="recognition-params"):
                                # 语音模型批量大小设置
                                with gr.Accordion("语音模型批量设置", open=False, elem_id="speech-batch-settings"):
                                    # 语音模型批处理大小
                                    batch_speech_batch_size = gr.Slider(
                                        label="语音模型批处理大小",
                                        value=DEFAULT_CONFIG["whisperx_params"].get("batch_size", 16),
                                        minimum=1,
                                        maximum=64,
                                        step=1,
                                        info="值越大处理速度越快但内存消耗越大",
                                        elem_id="speech-batch-size"
                                    )
                                    # 批量大小预设选项
                                    with gr.Row():
                                        gr.Button("小 (4)", variant="secondary").click(
                                            fn=lambda: 4,
                                            inputs=[],
                                            outputs=[batch_speech_batch_size]
                                        )
                                        gr.Button("中 (16)", variant="secondary").click(
                                            fn=lambda: 16,
                                            inputs=[],
                                            outputs=[batch_speech_batch_size]
                                        )
                                        gr.Button("大 (32)", variant="secondary").click(
                                            fn=lambda: 32,
                                            inputs=[],
                                            outputs=[batch_speech_batch_size]
                                        )
                                # 基础识别参数
                                with gr.Accordion("基础识别参数", open=False, elem_id="basic-recognition-params"):
                                    # Beam size设置
                                    batch_beam_size = gr.Slider(
                                        label="beam size",
                                        value=DEFAULT_CONFIG["speech_recognition_params"]["beam_size"],
                                        minimum=1,
                                        maximum=10,
                                        step=1,
                                        info="值越大准确率越高但速度越慢",
                                        elem_id="beam-size"
                                    )
                                    # VAD过滤
                                    batch_vad_filter = gr.Checkbox(
                                        label="VAD过滤",
                                        value=DEFAULT_CONFIG["speech_recognition_params"]["vad_filter"],
                                        info="过滤非语音部分",
                                        elem_id="vad-filter"
                                    )
                                    # Pyannote VAD参数
                                    with gr.Accordion("Pyannote VAD参数", open=False, elem_id="pyannote-vad-params"):
                                        # VAD阈值
                                        batch_vad_threshold = gr.Slider(
                                            label="VAD阈值",
                                            value=0.5,
                                            minimum=0.1,
                                            maximum=0.9,
                                            step=0.1,
                                            info="值越高，语音检测越严格",
                                            elem_id="vad-threshold"
                                        )
                                        # 最小语音持续时间
                                        batch_vad_min_speech = gr.Number(
                                            label="最小语音持续时间 (ms)",
                                            value=250,
                                            minimum=50,
                                            maximum=1000,
                                            step=50,
                                            info="低于此时间的语音将被过滤",
                                            elem_id="vad-min-speech"
                                        )
                                        # 最大语音持续时间
                                        batch_vad_max_speech = gr.Number(
                                            label="最大语音持续时间 (s)",
                                            value=30,
                                            minimum=5,
                                            maximum=60,
                                            step=5,
                                            info="超过此时间的语音将被分割",
                                            elem_id="vad-max-speech"
                                        )
                                        # 最小沉默持续时间
                                        batch_vad_min_silence = gr.Number(
                                            label="最小沉默持续时间 (ms)",
                                            value=100,
                                            minimum=50,
                                            maximum=500,
                                            step=50,
                                            info="低于此时间的沉默将被忽略",
                                            elem_id="vad-min-silence"
                                        )
                                    # 单词时间戳
                                    batch_word_timestamps = gr.Checkbox(
                                        label="单词时间戳",
                                        value=DEFAULT_CONFIG["speech_recognition_params"]["word_timestamps"],
                                        info="为每个单词添加时间戳",
                                        elem_id="word-timestamps"
                                    )
                                    # 基于先前文本
                                    batch_condition_on_previous_text = gr.Checkbox(
                                        label="基于先前文本",
                                        value=DEFAULT_CONFIG["speech_recognition_params"]["condition_on_previous_text"],
                                        info="利用上下文信息提高准确率",
                                        elem_id="condition-on-previous"
                                    )
                            # 翻译参数
                            with gr.Accordion("翻译参数", open=False, elem_id="translation-params"):
                                # 翻译模型批量大小设置
                                with gr.Accordion("翻译模型批量设置", open=False, elem_id="translation-batch-settings"):
                                    # 翻译模型批处理大小
                                    batch_translation_batch_size = gr.Slider(
                                        label="翻译模型批处理大小",
                                        value=DEFAULT_CONFIG["whisperx_params"].get("batch_size", 16),
                                        minimum=1,
                                        maximum=64,
                                        step=1,
                                        info="值越大处理速度越快但内存消耗越大",
                                        elem_id="translation-batch-size"
                                    )
                                    # 批量大小预设选项
                                    with gr.Row():
                                        gr.Button("小 (4)", variant="secondary").click(
                                            fn=lambda: 4,
                                            inputs=[],
                                            outputs=[batch_translation_batch_size]
                                        )
                                        gr.Button("中 (16)", variant="secondary").click(
                                            fn=lambda: 16,
                                            inputs=[],
                                            outputs=[batch_translation_batch_size]
                                        )
                                        gr.Button("大 (32)", variant="secondary").click(
                                            fn=lambda: 32,
                                            inputs=[],
                                            outputs=[batch_translation_batch_size]
                                        )
                                
                                batch_translation_beam_size = gr.Slider(
                                    label="beam size",
                                    value=DEFAULT_CONFIG["translation_params"]["beam_size"],
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    info="值越大翻译质量越高但速度越慢",
                                    elem_id="translation-beam-size"
                                )
                                
                                batch_translation_max_length = gr.Slider(
                                    label="最大长度",
                                    value=DEFAULT_CONFIG["translation_params"]["max_length"],
                                    minimum=50,
                                    maximum=1000,
                                    step=50,
                                    info="翻译结果的最大长度",
                                    elem_id="translation-max-length"
                                )
                                

                            # 参数控制按钮
                            with gr.Row(equal_height=True, elem_id="parameter-controls"):
                                save_params_btn = gr.Button("保存参数", variant="secondary")
                                load_params_btn = gr.Button("加载参数", variant="secondary")
                                reset_default_btn = gr.Button("恢复默认", variant="secondary")
            
            # 事件绑定
            self.bind_events(
                demo,
                batch_video_input,
                batch_model,
                batch_translator,
                batch_beam_size,
                batch_vad_filter,
                batch_word_timestamps,
                batch_condition_on_previous_text,
                batch_translation_beam_size,
                batch_translation_max_length,
                batch_device,
                batch_source_language,
                batch_target_language,
                batch_language_mode,
                batch_speech_batch_size,
                batch_translation_batch_size,
                batch_vad_threshold,
                batch_vad_min_speech,
                batch_vad_max_speech,
                batch_vad_min_silence,
                add_to_queue_btn,
                remove_from_queue_btn,
                clear_queue_btn,
                process_queue_btn,
                remove_index,
                queue_list,
                queue_status_message,
                save_params_btn,
                load_params_btn,
                reset_default_btn
            )
        
        return demo
    
    def bind_events(self, demo, batch_video_input, batch_model, batch_translator, batch_beam_size, 
                   batch_vad_filter, batch_word_timestamps, batch_condition_on_previous_text, 
                   batch_translation_beam_size, batch_translation_max_length, 
                   batch_device, batch_source_language, 
                   batch_target_language, batch_language_mode, 
                   batch_speech_batch_size, batch_translation_batch_size, 
                   batch_vad_threshold, batch_vad_min_speech, batch_vad_max_speech, batch_vad_min_silence, 
                   add_to_queue_btn, remove_from_queue_btn, clear_queue_btn, process_queue_btn, remove_index, 
                   queue_list, queue_status_message, save_params_btn, 
                   load_params_btn, reset_default_btn):
        """绑定UI事件"""
        # 源语言选择事件 - 自动更新语言模式
        batch_source_language.change(
            fn=LanguageManager.update_language_mode,
            inputs=[batch_source_language],
            outputs=[batch_language_mode]
        )
        
        # 队列操作事件
        add_to_queue_btn.click(
            fn=QueueManagerUI.add_to_queue,
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
                batch_device,
                batch_source_language,
                batch_target_language,
                batch_language_mode,
                batch_speech_batch_size,
                batch_translation_batch_size,
                batch_vad_threshold,
                batch_vad_min_speech,
                batch_vad_max_speech,
                batch_vad_min_silence
            ],
            outputs=[
                queue_list,
                queue_status_message
            ]
        )
        
        # 从队列删除文件
        remove_from_queue_btn.click(
            fn=QueueManagerUI.remove_from_queue,
            inputs=[remove_index],
            outputs=[
                queue_list,
                queue_status_message
            ]
        )
        
        # 清空队列
        clear_queue_btn.click(
            fn=QueueManagerUI.clear_queue,
            inputs=[],
            outputs=[
                queue_list,
                queue_status_message
            ]
        )
        
        # 开始处理队列
        process_queue_btn.click(
            fn=QueueManagerUI.process_queue,
            inputs=[],
            outputs=[
                queue_list,
                queue_status_message
            ]
        )
        
        # 参数控制事件
        save_params_btn.click(
            fn=ParameterManager.save_params,
            inputs=[
                batch_model,
                batch_translator,
                batch_beam_size,
                batch_vad_filter,
                batch_word_timestamps,
                batch_condition_on_previous_text,
                batch_translation_beam_size,
                batch_translation_max_length,
                batch_device,
                batch_source_language,
                batch_target_language,
                batch_language_mode,
                batch_speech_batch_size,
                batch_translation_batch_size,
                batch_vad_threshold,
                batch_vad_min_speech,
                batch_vad_max_speech,
                batch_vad_min_silence
            ],
            outputs=[queue_status_message]
        )
        
        load_params_btn.click(
            fn=ParameterManager.load_params,
            inputs=[],
            outputs=[
                batch_model,
                batch_translator,
                batch_beam_size,
                batch_vad_filter,
                batch_word_timestamps,
                batch_condition_on_previous_text,
                batch_translation_beam_size,
                batch_translation_max_length,
                batch_device,
                batch_source_language,
                batch_target_language,
                batch_language_mode,
                batch_speech_batch_size,
                batch_translation_batch_size,
                batch_vad_threshold,
                batch_vad_min_speech,
                batch_vad_max_speech,
                batch_vad_min_silence
            ]
        )
        
        reset_default_btn.click(
            fn=ParameterManager.reset_to_default,
            inputs=[],
            outputs=[
                batch_model,
                batch_translator,
                batch_beam_size,
                batch_vad_filter,
                batch_word_timestamps,
                batch_condition_on_previous_text,
                batch_translation_beam_size,
                batch_translation_max_length,
                batch_device,
                batch_source_language,
                batch_target_language,
                batch_language_mode,
                batch_speech_batch_size,
                batch_translation_batch_size,
                batch_vad_threshold,
                batch_vad_min_speech,
                batch_vad_max_speech,
                batch_vad_min_silence
            ]
        )
        
        # 视频文件输入变化时自动清空状态消息
        batch_video_input.change(
            fn=lambda x: "",
            inputs=[batch_video_input],
            outputs=[queue_status_message]
        )
        
        # 语言模式变化时的处理
        batch_language_mode.change(
            fn=lambda mode: "auto" if mode == "自动检测" else batch_source_language.value,
            inputs=[batch_language_mode],
            outputs=[batch_source_language]
        )
    
    def launch(self):
        """启动Gradio界面"""
        utils.timestamp_print("启动视频转字幕工具界面...")
        utils.timestamp_print("请在浏览器中打开以下URL:")
        utils.timestamp_print("http://localhost:7870")
        utils.timestamp_print("\n所有调试信息、错误信息都会打印到终端")
        utils.timestamp_print("="*80)
        
        # 启用队列机制，支持实时更新
        self.demo.queue(
            max_size=50,  # 队列最大长度
            api_open=False  # 关闭API访问
        )
        
        # 启动界面
        try:
            self.demo.launch(
                share=False,
                server_name="0.0.0.0",
                server_port=7870,
                theme=gr.themes.Soft()
            )
        except OSError as e:
            if "address already in use" in str(e).lower() or "10048" in str(e):
                utils.timestamp_print("[警告] 端口7870已被占用，尝试使用其他端口...")
                # 尝试使用不同端口
                for port in range(7871, 7880):
                    try:
                        self.demo.launch(
                            share=False,
                            server_name="0.0.0.0",
                            server_port=port,
                            theme=gr.themes.Soft()
                        )
                        break
                    except OSError:
                        continue
                else:
                    utils.timestamp_print("[错误] 无法找到可用端口，启动失败")
            else:
                raise


# 启动Gradio界面
if __name__ == "__main__":
    import argparse
    
    # 尝试加载保存的参数
    saved_params = ParameterManager.load_params()
    
    # 解析命令行参数，使用保存的参数作为默认值
    parser = argparse.ArgumentParser(description="视频转字幕工具 - 支持命令行参数")
    parser.add_argument("--model", type=str, default=saved_params[0] if saved_params else DEFAULT_CONFIG["speech_recognition_model"], 
                        help="语音识别模型 (tiny, base, small, medium, large-v2, large-v3)")
    parser.add_argument("--translator", type=str, default=saved_params[1] if saved_params else "m2m100_418M", 
                        help="翻译模型 (m2m100_418M, m2m100_1.2B)")
    parser.add_argument("--device", type=str, default=saved_params[8] if saved_params else DEFAULT_CONFIG["device"], 
                        help="设备选择 (auto, cpu, cuda)")
    parser.add_argument("--source-language", type=str, default=saved_params[9] if saved_params else config.default_source_language, 
                        help="源语言 (auto, zh, en, ja, ko, fr, de, es, ru, ar, hi, pt, it, nl, pl)")
    parser.add_argument("--target-language", type=str, default=saved_params[10] if saved_params else config.default_target_language, 
                        help="目标语言 (zh, en, ja, ko, fr, de, es, ru, ar, hi, pt, it, nl, pl)")
    parser.add_argument("--language-mode", type=str, default="auto_detect" if (not saved_params or saved_params[11] == "自动检测") else "manual", 
                        help="语言模式 (auto_detect, manual)")
    parser.add_argument("--speech-batch-size", type=int, default=saved_params[12] if saved_params else DEFAULT_CONFIG["whisperx_params"].get("batch_size", 16), 
                        help="语音模型批处理大小 (1-32)")
    parser.add_argument("--translation-batch-size", type=int, default=saved_params[13] if saved_params else DEFAULT_CONFIG["whisperx_params"].get("batch_size", 16), 
                        help="翻译模型批处理大小 (1-32)")
    parser.add_argument("--video-files", type=str, nargs="*", 
                        help="要处理的视频文件路径")
    parser.add_argument("--process", action="store_true", 
                        help="自动开始处理队列")
    
    args = parser.parse_args()
    
    # 打印命令行参数
    utils.timestamp_print("命令行参数:")
    utils.timestamp_print(f"  模型: {args.model}")
    utils.timestamp_print(f"  翻译模型: {args.translator}")
    utils.timestamp_print(f"  设备: {args.device}")
    utils.timestamp_print(f"  源语言: {args.source_language}")
    utils.timestamp_print(f"  目标语言: {args.target_language}")
    utils.timestamp_print(f"  语言模式: {args.language_mode}")
    utils.timestamp_print(f"  语音模型批处理大小: {args.speech_batch_size}")
    utils.timestamp_print(f"  翻译模型批处理大小: {args.translation_batch_size}")
    utils.timestamp_print(f"  视频文件: {args.video_files}")
    utils.timestamp_print(f"  自动处理: {args.process}")
    
    # 启动UI
    ui = VideoSubtitleUI()
    
    # 如果提供了视频文件，添加到队列
    if args.video_files:
        # 构建参数字典
        translator_model = config.translator_models.get(args.translator, args.translator)
        params = {
            'model': args.model,
            'translator': translator_model,
            'beam_size': DEFAULT_CONFIG["speech_recognition_params"]["beam_size"],
            'vad_filter': DEFAULT_CONFIG["speech_recognition_params"]["vad_filter"],
            'word_timestamps': DEFAULT_CONFIG["speech_recognition_params"]["word_timestamps"],
            'condition_on_previous_text': DEFAULT_CONFIG["speech_recognition_params"]["condition_on_previous_text"],
            'translation_beam_size': DEFAULT_CONFIG["translation_params"]["beam_size"],
            'translation_max_length': DEFAULT_CONFIG["translation_params"]["max_length"],
            'device': args.device,
            'source_language': args.source_language,
            'target_language': args.target_language,
            'language_mode': args.language_mode,
            'use_whisperx': True,  # 默认启用WhisperX
            'speech_batch_size': args.speech_batch_size,
            'translation_batch_size': args.translation_batch_size
        }
        
        # 添加文件到队列
        added_count = queue_manager.add_to_queue(args.video_files, params)
        utils.timestamp_print(f"成功添加 {added_count} 个视频到队列")
        
        # 如果指定了自动处理，开始处理队列
        if args.process:
            utils.timestamp_print("开始处理队列...")
            for result in queue_manager.process_queue():
                queue_data, message, logs, progress, status_text = result
                utils.timestamp_print(f"处理状态: {status_text}")
                utils.timestamp_print(f"进度: {progress}%")
            utils.timestamp_print("队列处理完成！")
    
    # 启动Gradio界面
    ui.launch()