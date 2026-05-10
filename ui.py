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
from utils.logger import setup_print_redirect, timestamp_print
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



def add_to_queue(video):
    try:
        if not video:
            msg = "错误：请先上传视频文件"
            timestamp_print(msg)
            return msg
        files = video if isinstance(video, list) else [video]
        if not files:
            msg = "错误：未选择任何文件"
            timestamp_print(msg)
            return msg

        # 从配置文件获取所有参数
        params = config.build_params()

        count = queue_manager.add_to_queue(files, params)
        msg = f"成功：已添加 {count} 个文件到队列\n使用模型: {config.get('model', MODEL_OPTIONS[0] if MODEL_OPTIONS else 'large-v3')}\n队列当前有 {len(queue_manager.video_queue)} 个文件"
        timestamp_print(msg)
        return msg
    except Exception as e:
        msg = f"错误：添加文件失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        timestamp_print(msg)
        return msg


def clear_queue():
    try:
        count = queue_manager.clear_queue()
        msg = f"成功：队列已清空 (清除了 {count} 个文件)"
        timestamp_print(msg)
        return msg
    except Exception as e:
        msg = f"错误：清空队列失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        timestamp_print(msg)
        return msg


def process_queue():
    try:
        if not queue_manager.video_queue:
            msg = "提示：队列为空，请先添加视频文件"
            timestamp_print(msg)
            return msg

        if queue_manager.processing:
            msg = "提示：正在处理中，请等待完成"
            timestamp_print(msg)
            return msg

        total = len(queue_manager.video_queue)
        model = config.get('model', MODEL_OPTIONS[0] if MODEL_OPTIONS else "large-v3")
        logs = [f"开始处理，共 {total} 个文件...\n使用模型: {model}\n"]

        for queue_state, _, log_text, progress, _ in queue_manager.process_queue():
            pass

        final_statuses = []
        with queue_manager._lock:
            for item in queue_manager.video_queue:
                final_statuses.append(f"{item['filename']}: {item['status']}")

        result_msg = "\n".join(logs)
        result_msg += "\n\n处理结果:\n" + "\n".join(final_statuses)
        return result_msg
    except Exception as e:
        msg = f"错误：处理队列失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        timestamp_print(msg)
        return msg


def save_config(model, translator, translator_quantization, device, source_language, target_language,
                vad_threshold, vad_min_speech_duration, vad_max_speech_duration, vad_min_silence_duration,
                vad_speech_pad_ms, vad_prefix_padding_ms, vad_neg_threshold, use_max_poss_sil_at_max_speech,
                vad_min_silence_at_max_speech, vad_time_resolution,
                whispercd_alpha, whispercd_temperature, whispercd_snr_db, whispercd_temporal_shift,
                whispercd_context_max_tokens,
                enable_forced_alignment,
                llama_server_host, llama_server_port, llama_server_context_size, llama_server_threads,
                llama_server_ngl, llama_server_batch_size,
                translation_temperature, translation_top_k, translation_top_p, translation_repetition_penalty,
                translation_batch_size, translation_context_size,
                translation_segment_context_window, translation_max_retries, translation_max_total_retries, translation_max_output_tokens):
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
            vad_min_silence_at_max_speech=vad_min_silence_at_max_speech,
            vad_time_resolution=vad_time_resolution,
            whispercd_alpha=whispercd_alpha,
            whispercd_temperature=whispercd_temperature,
            whispercd_snr_db=whispercd_snr_db,
            whispercd_temporal_shift=whispercd_temporal_shift,
            whispercd_context_max_tokens=whispercd_context_max_tokens,
            enable_forced_alignment=enable_forced_alignment,
            llama_server_host=llama_server_host,
            llama_server_port=llama_server_port,
            llama_server_context_size=llama_server_context_size,
            llama_server_threads=llama_server_threads,
            llama_server_ngl=llama_server_ngl,
            llama_server_batch_size=llama_server_batch_size,
            translation_temperature=translation_temperature,
            translation_top_k=translation_top_k,
            translation_top_p=translation_top_p,
            translation_repetition_penalty=translation_repetition_penalty,
            translation_batch_size=translation_batch_size,
            translation_context_size=translation_context_size,
            translation_segment_context_window=translation_segment_context_window,
            translation_max_retries=translation_max_retries,
            translation_max_total_retries=translation_max_total_retries,
            translation_max_output_tokens=translation_max_output_tokens
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


def cancel_processing():
    try:
        queue_manager.cancel_processing()
        msg = "成功：已发送取消请求"
        timestamp_print(msg)
        return msg
    except Exception as e:
        msg = f"错误：取消失败\n{type(e).__name__}: {e}"
        timestamp_print(msg)
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
                btn4 = gr.Button("取消处理", variant="stop")

            status = gr.Textbox(label="状态信息", lines=10, interactive=False)

            btn1.click(add_to_queue, inputs=[video], outputs=[status])
            btn2.click(clear_queue, outputs=[status])
            btn3.click(process_queue, outputs=[status])
            btn4.click(cancel_processing, outputs=[status])
        
        # 配置选项标签页
        with gr.TabItem("配置选项"):
            gr.Markdown("# 配置选项")
            gr.Markdown("修改配置参数并保存")
            
            # 基本设置
            with gr.Accordion("基本设置", open=True):
                model = gr.Dropdown(choices=MODEL_OPTIONS, value=config.get('model'), label="语音识别模型")
                translator = gr.Dropdown(choices=["tencent/HY-MT1.5-7B-GGUF"], value=config.get('translator'), label="翻译模型")
                translator_quantization = gr.Dropdown(choices=["auto", "Q4_K_M", "Q6_K", "Q8_0"], value=config.get('translator_quantization'), label="翻译模型量化版本")
                device = gr.Dropdown(choices=["auto", "cuda", "cpu"], value=config.get('device'), label="计算设备")
                source_language = gr.Dropdown(choices=["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"], value=config.get('source_language'), label="源语言")
                target_language = gr.Dropdown(choices=["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"], value=config.get('target_language'), label="目标语言")
            
            # VAD设置
            with gr.Accordion("VAD设置", open=False):
                vad_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=config.get('vad_threshold'), label="VAD阈值")
                vad_min_speech_duration = gr.Slider(minimum=0.01, maximum=10.0, value=config.get('vad_min_speech_duration'), label="最小语音持续时间(秒)")
                vad_max_speech_duration = gr.Slider(minimum=1.0, maximum=300.0, value=config.get('vad_max_speech_duration'), label="最大语音持续时间(秒)")
                vad_min_silence_duration = gr.Slider(minimum=0.01, maximum=10.0, value=config.get('vad_min_silence_duration'), label="最小静默持续时间(秒)")
                vad_speech_pad_ms = gr.Slider(minimum=0, maximum=5000, value=config.get('vad_speech_pad_ms'), label="语音填充时间(毫秒)")
                vad_prefix_padding_ms = gr.Slider(minimum=0, maximum=5000, value=config.get('vad_prefix_padding_ms'), label="前缀填充时间(毫秒)")
                vad_neg_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=config.get('vad_neg_threshold'), label="静音判定阈值")
                use_max_poss_sil_at_max_speech = gr.Checkbox(value=config.get('use_max_poss_sil_at_max_speech'), label="优先使用最长静音分割点")
                vad_min_silence_at_max_speech = gr.Slider(minimum=0, maximum=1000, value=config.get('vad_min_silence_at_max_speech'), step=1, label="长语音分割静音时长(毫秒)")
                vad_time_resolution = gr.Slider(minimum=1, maximum=1000, value=config.get('vad_time_resolution'), step=1, label="时间分辨率(毫秒)")
            
            # 强制对齐设置
            with gr.Accordion("强制对齐设置", open=False):
                enable_forced_alignment = gr.Checkbox(value=config.get('enable_forced_alignment'), label="启用强制对齐(Wav2Vec2)")
            
            # Whisper-CD设置
            with gr.Accordion("Whisper-CD设置", open=False):
                whispercd_alpha = gr.Slider(minimum=0.0, maximum=2.0, value=config.get('whispercd_alpha'), label="对比强度参数")
                whispercd_temperature = gr.Slider(minimum=0.1, maximum=5.0, value=config.get('whispercd_temperature'), label="log-sum-exp温度参数")
                whispercd_snr_db = gr.Slider(minimum=0.0, maximum=30.0, value=config.get('whispercd_snr_db'), label="高斯噪声注入的SNR值")
                whispercd_temporal_shift = gr.Slider(minimum=0.0, maximum=15.0, value=config.get('whispercd_temporal_shift'), label="音频时间移位的秒数")
                whispercd_context_max_tokens = gr.Slider(minimum=64, maximum=400, value=config.get('whispercd_context_max_tokens'), label="上下文最大 token 数")
            
            # Llama Server设置
            with gr.Accordion("Llama Server设置", open=False):
                llama_server_host = gr.Textbox(value=config.get('llama_server_host'), label="Llama Server地址")
                llama_server_port = gr.Slider(minimum=1, maximum=65535, value=config.get('llama_server_port'), label="Llama Server端口")
                llama_server_context_size = gr.Slider(minimum=512, maximum=32768, value=config.get('llama_server_context_size'), label="Llama Server上下文大小")
                llama_server_threads = gr.Slider(minimum=1, maximum=128, value=config.get('llama_server_threads'), label="Llama Server线程数")
                llama_server_ngl = gr.Slider(minimum=0, maximum=999, value=config.get('llama_server_ngl'), step=1, label="GPU卸载层数")
                llama_server_batch_size = gr.Slider(minimum=512, maximum=8192, value=config.get('llama_server_batch_size'), step=512, label="批处理大小")

            with gr.Accordion("翻译高级设置", open=False):
                translation_temperature = gr.Slider(minimum=0.0, maximum=2.0, value=config.get('translation_temperature'), label="翻译温度")
                translation_top_k = gr.Slider(minimum=1, maximum=100, value=config.get('translation_top_k'), step=1, label="Top-K采样")
                translation_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=config.get('translation_top_p'), label="Top-P核采样")
                translation_repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=config.get('translation_repetition_penalty'), step=0.01, label="重复惩罚")
                translation_batch_size = gr.Slider(minimum=1024, maximum=8192, value=config.get('translation_batch_size'), step=256, label="翻译批处理大小")
                translation_context_size = gr.Slider(minimum=1024, maximum=32768, value=config.get('translation_context_size'), step=256, label="翻译上下文大小")
                translation_segment_context_window = gr.Slider(minimum=0, maximum=20, value=config.get('translation_segment_context_window'), step=1, label="上下文片段数量")
                translation_max_retries = gr.Slider(minimum=1, maximum=10, value=config.get('translation_max_retries'), step=1, label="单条最大重试次数")
                translation_max_total_retries = gr.Slider(minimum=3, maximum=30, value=config.get('translation_max_total_retries'), step=1, label="验证失败最大总重试次数")
                translation_max_output_tokens = gr.Slider(minimum=1024, maximum=32768, value=config.get('translation_max_output_tokens'), step=1024, label="最大输出token数")

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
                    vad_min_silence_at_max_speech, vad_time_resolution,
                    whispercd_alpha, whispercd_temperature, whispercd_snr_db, whispercd_temporal_shift,
                    whispercd_context_max_tokens,
                    enable_forced_alignment,
                    llama_server_host, llama_server_port, llama_server_context_size, llama_server_threads,
                    llama_server_ngl, llama_server_batch_size,
                    translation_temperature, translation_top_k, translation_top_p, translation_repetition_penalty,
                    translation_batch_size, translation_context_size,
                    translation_segment_context_window, translation_max_retries, translation_max_total_retries, translation_max_output_tokens
                ],
                outputs=[config_status]
            )
            
            reset_btn.click(
                reset_config,
                outputs=[config_status]
            )

demo.launch(server_name="127.0.0.1", server_port=None, share=False)
