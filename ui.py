# -*- coding: utf-8 -*-
"""
字幕翻译工具 - Gradio UI (极简版)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", ".cache", "huggingface"))

# 设置ffmpeg临时变量
FFMPEG_DIR = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg"))
os.environ["FFMPEG_DIR"] = FFMPEG_DIR
# 将ffmpeg目录添加到系统PATH
sys_path = os.environ.get("PATH", "")
os.environ["PATH"] = f"{FFMPEG_DIR};{sys_path}"
print(f"设置ffmpeg目录: {FFMPEG_DIR}")
print(f"更新后的PATH: {os.environ['PATH'][:200]}...")

# 设置日志重定向
from utils.logger import setup_print_redirect
setup_print_redirect()
print("应用程序启动")

import sys
import traceback
from pathlib import Path
import gradio as gr

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config, MODEL_OPTIONS
from utils.queue_manager import QueueManager

queue_manager = QueueManager()

# 从配置文件获取模型
MODEL = config.get('model', MODEL_OPTIONS[0] if MODEL_OPTIONS else "large-v3")
print(f"可用模型列表: {MODEL_OPTIONS}")
print(f"使用模型: {MODEL}")


def log(msg):
    print(msg)
    return msg


def add_to_queue(video):
    try:
        if not video:
            msg = "错误：请先上传视频文件"
            return log(msg)
        files = video if isinstance(video, list) else [video]
        if not files:
            msg = "错误：未选择任何文件"
            return log(msg)

        # 从配置文件获取所有参数
        params = {
            'model': config.get('model', MODEL_OPTIONS[0] if MODEL_OPTIONS else "large-v3"),
            'translator': config.get('translator', 'tencent/HY-MT1.5-7B-GGUF'),
            'source_language': config.get('source_language', 'ja'),
            'target_language': config.get('target_language', 'zh'),
            'device': config.get('device', 'auto'),
            'enable_whispercd': config.get('enable_whispercd', True),
            'whispercd_alpha': config.get('whispercd_alpha', 1.0),
            'whispercd_temperature': config.get('whispercd_temperature', 1.0),
            'whispercd_snr_db': config.get('whispercd_snr_db', 10.0),
            'whispercd_temporal_shift': config.get('whispercd_temporal_shift', 7.0),
            'enable_forced_alignment': config.get('enable_forced_alignment', True),
            'vad_threshold': config.get('vad_threshold', 0.4),
            'vad_min_speech_duration': config.get('vad_min_speech_duration', 1.0),
            'vad_max_speech_duration': config.get('vad_max_speech_duration', 30.0),
            'vad_min_silence_duration': config.get('vad_min_silence_duration', 1.0),
            'vad_speech_pad_ms': config.get('vad_speech_pad_ms', 300),
            'vad_prefix_padding_ms': config.get('vad_prefix_padding_ms', 50),
            'vad_neg_threshold': config.get('vad_neg_threshold', None),
            'use_max_poss_sil_at_max_speech': config.get('use_max_poss_sil_at_max_speech', True),
            'translation_batch_size': config.get('translation_batch_size', 3500),
            'translation_context_size': config.get('translation_context_size', 8192),
            'translation_temperature': config.get('translation_temperature', 0.0),
            'translation_top_k': config.get('translation_top_k', 20),
            'translation_top_p': config.get('translation_top_p', 0.6),
            'translation_repetition_penalty': config.get('translation_repetition_penalty', 1.05),
        }

        count = queue_manager.add_to_queue(files, params)
        msg = f"成功：已添加 {count} 个文件到队列\n使用模型: {config.get('model', MODEL_OPTIONS[0] if MODEL_OPTIONS else 'large-v3')}\n队列当前有 {len(queue_manager.video_queue)} 个文件"
        return log(msg)
    except Exception as e:
        msg = f"错误：添加文件失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return log(msg)


def clear_queue():
    try:
        count = queue_manager.clear_queue()
        msg = f"成功：队列已清空 (清除了 {count} 个文件)"
        return log(msg)
    except Exception as e:
        msg = f"错误：清空队列失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return log(msg)


def process_queue():
    try:
        if not queue_manager.video_queue:
            msg = "提示：队列为空，请先添加视频文件"
            return log(msg)

        total = len(queue_manager.video_queue)
        model = config.get('model', MODEL_OPTIONS[0] if MODEL_OPTIONS else "large-v3")
        logs = [f"开始处理，共 {total} 个文件...\n使用模型: {model}\n"]
        print("\n".join(logs))

        for i, item in enumerate(queue_manager.video_queue):
            try:
                status = f"[{i+1}/{total}] 正在处理: {item['filename']}"
                print(status)
                logs.append(status)
                success, msg, output, log = queue_manager.process_video(item['file_path'], item['params'])
                if success:
                    status = f"[{i+1}/{total}] 完成: {item['filename']}"
                else:
                    status = f"[{i+1}/{total}] 失败: {item['filename']} - {msg}"
                print(status)
                logs.append(status)
            except Exception as e:
                status = f"[{i+1}/{total}] 异常: {item['filename']} - {type(e).__name__}: {e}"
                print(status)
                logs.append(status)

        queue_manager.clear_queue()
        final_msg = "\n处理完成！"
        print(final_msg)
        logs.append(final_msg)
        return "\n".join(logs)
    except Exception as e:
        msg = f"错误：处理队列失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return log(msg)


def save_config(model, translator, translator_quantization, device, source_language, target_language,
                vad_threshold, vad_min_speech_duration, vad_max_speech_duration, vad_min_silence_duration,
                vad_speech_pad_ms, vad_prefix_padding_ms, vad_neg_threshold, use_max_poss_sil_at_max_speech,
                enable_whispercd, whispercd_alpha, whispercd_temperature, whispercd_snr_db, whispercd_temporal_shift,
                llama_server_host, llama_server_port, llama_server_context_size, llama_server_threads):
    try:
        success, msg = config.save(
            model=model,
            translator=translator,
            translator_quantization=translator_quantization,
            device=device,
            source_language=source_language,
            target_language=target_language,
            vad_threshold=vad_threshold,
            vad_min_speech_duration=vad_min_speech_duration,
            vad_max_speech_duration=vad_max_speech_duration,
            vad_min_silence_duration=vad_min_silence_duration,
            vad_speech_pad_ms=vad_speech_pad_ms,
            vad_prefix_padding_ms=vad_prefix_padding_ms,
            vad_neg_threshold=vad_neg_threshold,
            use_max_poss_sil_at_max_speech=use_max_poss_sil_at_max_speech,
            enable_whispercd=enable_whispercd,
            whispercd_alpha=whispercd_alpha,
            whispercd_temperature=whispercd_temperature,
            whispercd_snr_db=whispercd_snr_db,
            whispercd_temporal_shift=whispercd_temporal_shift,
            llama_server_host=llama_server_host,
            llama_server_port=llama_server_port,
            llama_server_context_size=llama_server_context_size,
            llama_server_threads=llama_server_threads
        )
        if success:
            return f"成功：{msg}"
        else:
            return f"错误：{msg}"
    except Exception as e:
        msg = f"错误：保存配置失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return msg


def reset_config():
    try:
        success, msg = config.reset()
        if success:
            return f"成功：{msg}"
        else:
            return f"错误：{msg}"
    except Exception as e:
        msg = f"错误：重置配置失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return msg


with gr.Blocks() as demo:
    gr.Markdown("# 字幕翻译工具")
    
    with gr.Tabs():
        # 直接使用标签页
        with gr.TabItem("直接使用"):
            gr.Markdown(f"上传视频文件添加到队列，或直接处理\n源语言: 日语 → 目标语言: 中文\n使用模型: {config.get('model', MODEL_OPTIONS[0] if MODEL_OPTIONS else 'large-v3')} (检测到 {len(MODEL_OPTIONS)} 个可用模型)")

            video = gr.File(label="选择视频文件", file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts"])

            with gr.Row():
                btn1 = gr.Button("添加到队列", variant="primary")
                btn2 = gr.Button("清空队列", variant="secondary")
                btn3 = gr.Button("处理队列", variant="primary")

            status = gr.Textbox(label="状态信息", lines=10, interactive=False)

            btn1.click(add_to_queue, inputs=[video], outputs=[status])
            btn2.click(clear_queue, outputs=[status])
            btn3.click(process_queue, outputs=[status])
        
        # 配置选项标签页
        with gr.TabItem("配置选项"):
            gr.Markdown("# 配置选项")
            gr.Markdown("修改配置参数并保存")
            
            # 基本设置
            with gr.Accordion("基本设置", open=True):
                model = gr.Dropdown(choices=MODEL_OPTIONS, value=config.get('model', MODEL_OPTIONS[0] if MODEL_OPTIONS else "large-v3"), label="语音识别模型")
                translator = gr.Dropdown(choices=["tencent/HY-MT1.5-7B-GGUF"], value=config.get('translator', 'tencent/HY-MT1.5-7B-GGUF'), label="翻译模型")
                translator_quantization = gr.Dropdown(choices=["auto", "Q4_K_M", "Q6_K", "Q8_0"], value=config.get('translator_quantization', 'Q4_K_M'), label="翻译模型量化版本")
                device = gr.Dropdown(choices=["auto", "cuda", "cpu"], value=config.get('device', 'auto'), label="计算设备")
                source_language = gr.Dropdown(choices=["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"], value=config.get('source_language', 'ja'), label="源语言")
                target_language = gr.Dropdown(choices=["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"], value=config.get('target_language', 'zh'), label="目标语言")
            
            # VAD设置
            with gr.Accordion("VAD设置", open=False):
                vad_threshold = gr.Slider(minimum=0.1, maximum=0.9, value=config.get('vad_threshold', 0.47), label="VAD阈值")
                vad_min_speech_duration = gr.Slider(minimum=0.1, maximum=10.0, value=config.get('vad_min_speech_duration', 0.2), label="最小语音持续时间(秒)")
                vad_max_speech_duration = gr.Slider(minimum=10.0, maximum=300.0, value=config.get('vad_max_speech_duration', 30.0), label="最大语音持续时间(秒)")
                vad_min_silence_duration = gr.Slider(minimum=0.1, maximum=10.0, value=config.get('vad_min_silence_duration', 2.0), label="最小静默持续时间(秒)")
                vad_speech_pad_ms = gr.Slider(minimum=0, maximum=5000, value=config.get('vad_speech_pad_ms', 400), label="语音填充时间(毫秒)")
                vad_prefix_padding_ms = gr.Slider(minimum=0, maximum=5000, value=config.get('vad_prefix_padding_ms', 0), label="前缀填充时间(毫秒)")
                vad_neg_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=config.get('vad_neg_threshold', 0.32), label="静音判定阈值")
                use_max_poss_sil_at_max_speech = gr.Checkbox(value=config.get('use_max_poss_sil_at_max_speech', True), label="优先使用最长静音分割点")
            
            # Whisper-CD设置
            with gr.Accordion("Whisper-CD设置", open=False):
                enable_whispercd = gr.Checkbox(value=config.get('enable_whispercd', True), label="启用Whisper-CD")
                whispercd_alpha = gr.Slider(minimum=0.0, maximum=2.0, value=config.get('whispercd_alpha', 1.0), label="对比强度参数")
                whispercd_temperature = gr.Slider(minimum=0.1, maximum=5.0, value=config.get('whispercd_temperature', 1.0), label="log-sum-exp温度参数")
                whispercd_snr_db = gr.Slider(minimum=0.0, maximum=30.0, value=config.get('whispercd_snr_db', 10.0), label="高斯噪声注入的SNR值")
                whispercd_temporal_shift = gr.Slider(minimum=0.0, maximum=15.0, value=config.get('whispercd_temporal_shift', 7.0), label="音频时间移位的秒数")
            
            # Llama Server设置
            with gr.Accordion("Llama Server设置", open=False):
                llama_server_host = gr.Textbox(value=config.get('llama_server_host', '127.0.0.1'), label="Llama Server地址")
                llama_server_port = gr.Slider(minimum=1, maximum=65535, value=config.get('llama_server_port', 8080), label="Llama Server端口")
                llama_server_context_size = gr.Slider(minimum=512, maximum=32768, value=config.get('llama_server_context_size', 8192), label="Llama Server上下文大小")
                llama_server_threads = gr.Slider(minimum=1, maximum=128, value=config.get('llama_server_threads', 8), label="Llama Server线程数")
            
            # 配置操作按钮
            with gr.Row():
                save_btn = gr.Button("保存配置", variant="primary")
                reset_btn = gr.Button("重置默认", variant="secondary")
            
            # 配置状态信息
            config_status = gr.Textbox(label="配置状态", lines=3, interactive=False)

            # 绑定按钮事件
            save_btn.click(
                save_config,
                inputs=[
                    model, translator, translator_quantization, device, source_language, target_language,
                    vad_threshold, vad_min_speech_duration, vad_max_speech_duration, vad_min_silence_duration,
                    vad_speech_pad_ms, vad_prefix_padding_ms, vad_neg_threshold, use_max_poss_sil_at_max_speech,
                    enable_whispercd, whispercd_alpha, whispercd_temperature, whispercd_snr_db, whispercd_temporal_shift,
                    llama_server_host, llama_server_port, llama_server_context_size, llama_server_threads
                ],
                outputs=[config_status]
            )
            
            reset_btn.click(
                reset_config,
                outputs=[config_status]
            )

demo.launch(server_name="127.0.0.1", server_port=None, share=False)
