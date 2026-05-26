# -*- coding: utf-8 -*-
"""
字幕翻译工具 - Gradio UI (极简版)
"""

import os
import sys

from config import IS_PACKAGE_MODE, PROJECT_ROOT

if IS_PACKAGE_MODE:
    FFMPEG_DIR = os.path.join(PROJECT_ROOT, "ffmpeg", "ffmpeg-master-latest-win64-gpl", "bin")
else:
    FFMPEG_DIR = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg"))
os.environ["FFMPEG_DIR"] = FFMPEG_DIR
sys_path = os.environ.get("PATH", "")
os.environ["PATH"] = f"{FFMPEG_DIR};{sys_path}"

# 设置日志重定向
from utils.logger import setup_print_redirect
setup_print_redirect()
print("应用程序启动")

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

LANG_MAP = {
    "zh": "中文", "en": "英文", "ja": "日语", "ko": "韩语",
    "fr": "法语", "de": "德语", "es": "西班牙语", "ru": "俄语",
    "ar": "阿拉伯语", "hi": "印地语", "pt": "葡萄牙语", "it": "意大利语",
    "nl": "荷兰语", "pl": "波兰语"
}



def add_to_queue(video):
    try:
        if not video:
            msg = "错误：请先上传视频文件"
            print(msg)
            return msg
        files = video if isinstance(video, list) else [video]
        if not files:
            msg = "错误：未选择任何文件"
            print(msg)
            return msg

        # 从配置文件获取所有参数
        params = config.build_params()

        count = queue_manager.add_to_queue(files, params)
        msg = f"成功：已添加 {count} 个文件到队列\n使用模型: {config.get('model', MODEL_OPTIONS[0] if MODEL_OPTIONS else 'large-v3')}\n队列当前有 {len(queue_manager.video_queue)} 个文件"
        print(msg)
        return msg
    except Exception as e:
        msg = f"错误：添加文件失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(msg)
        return msg


def clear_queue():
    try:
        count = queue_manager.clear_queue()
        msg = f"成功：队列已清空 (清除了 {count} 个文件)"
        print(msg)
        return msg
    except Exception as e:
        msg = f"错误：清空队列失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(msg)
        return msg


def process_queue():
    try:
        if not queue_manager.video_queue:
            msg = "提示：队列为空，请先添加视频文件"
            print(msg)
            return msg

        if queue_manager.processing:
            msg = "提示：正在处理中，请等待完成"
            print(msg)
            return msg

        total = len(queue_manager.video_queue)
        model = config.get('model', MODEL_OPTIONS[0] if MODEL_OPTIONS else "large-v3")
        logs = [f"开始处理，共 {total} 个文件...\n使用模型: {model}\n"]

        for queue_state, _, log_text, progress, _ in queue_manager.process_queue():
            if log_text:
                logs.append(log_text)

        final_statuses = queue_manager.get_queue_statuses()

        result_msg = "\n".join(logs)
        result_msg += "\n\n处理结果:\n" + "\n".join(final_statuses)
        return result_msg
    except Exception as e:
        msg = f"错误：处理队列失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(msg)
        return msg


def save_config(
    model, device, source_language, target_language,
    enable_forced_alignment,
    whispercd_alpha, whispercd_temperature, whispercd_snr_db, whispercd_temporal_shift,
    whispercd_context_max_tokens, whispercd_max_duration, whispercd_min_duration, whispercd_gap_threshold,
    whispercd_particle_chars,
    translator, translator_quantization,
    llama_server_host, llama_server_port, llama_server_context_size, llama_server_threads,
    llama_server_ngl, llama_server_batch_size, llama_server_parallel_slots,
    translation_reset_session,
    translation_temperature, translation_top_k, translation_top_p, translation_repetition_penalty,
    translation_segment_context_window,
    translation_max_retries, translation_max_total_retries, translation_max_output_tokens,
):
    try:
        params = {
            'model': model,
            'translator': translator,
            'translator_quantization': translator_quantization,
            'device': device,
            'source_language': source_language,
            'target_language': target_language,
            'enable_forced_alignment': enable_forced_alignment,
            'whispercd_alpha': whispercd_alpha,
            'whispercd_temperature': whispercd_temperature,
            'whispercd_snr_db': whispercd_snr_db,
            'whispercd_temporal_shift': whispercd_temporal_shift,
            'whispercd_context_max_tokens': whispercd_context_max_tokens,
            'whispercd_max_duration': whispercd_max_duration,
            'whispercd_min_duration': whispercd_min_duration,
            'whispercd_gap_threshold': whispercd_gap_threshold,
            'whispercd_particle_chars': whispercd_particle_chars,
            'llama_server_host': llama_server_host,
            'llama_server_port': int(llama_server_port) if llama_server_port is not None else None,
            'llama_server_context_size': llama_server_context_size,
            'llama_server_threads': llama_server_threads,
            'llama_server_ngl': llama_server_ngl,
            'llama_server_batch_size': llama_server_batch_size,
            'llama_server_parallel_slots': llama_server_parallel_slots,
            'translation_temperature': translation_temperature,
            'translation_top_k': translation_top_k,
            'translation_top_p': translation_top_p,
            'translation_repetition_penalty': translation_repetition_penalty,
            'translation_segment_context_window': translation_segment_context_window,
            'translation_max_retries': translation_max_retries,
            'translation_max_total_retries': translation_max_total_retries,
            'translation_max_output_tokens': translation_max_output_tokens,
            'translation_reset_session': translation_reset_session,
        }
        success, msg = config.save(**params)
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
            return (
                f"成功：{msg}",
                config.get('model'), config.get('device'), config.get('source_language'), config.get('target_language'),
                config.get('enable_forced_alignment'),
                config.get('whispercd_alpha'), config.get('whispercd_temperature'), config.get('whispercd_snr_db'),
                config.get('whispercd_temporal_shift'), config.get('whispercd_context_max_tokens'),
                config.get('whispercd_max_duration'), config.get('whispercd_min_duration'),
                config.get('whispercd_gap_threshold'), config.get('whispercd_particle_chars'),
                config.get('translator'), config.get('translator_quantization'),
                config.get('llama_server_host'), config.get('llama_server_port'),
                config.get('llama_server_context_size'), config.get('llama_server_threads'),
                config.get('llama_server_ngl'), config.get('llama_server_batch_size'),
                config.get('llama_server_parallel_slots'),
                config.get('translation_reset_session'),
                config.get('translation_temperature'), config.get('translation_top_k'),
                config.get('translation_top_p'), config.get('translation_repetition_penalty'),
                config.get('translation_segment_context_window'), config.get('translation_max_retries'),
                config.get('translation_max_total_retries'), config.get('translation_max_output_tokens'),
            )
        else:
            return (f"错误：{msg}",) + (None,) * 32
    except Exception as e:
        msg = f"错误：重置配置失败\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return (msg,) + (None,) * 32


def cancel_processing():
    try:
        queue_manager.cancel_processing()
        msg = "成功：已发送取消请求"
        print(msg)
        return msg
    except Exception as e:
        msg = f"错误：取消失败\n{type(e).__name__}: {e}"
        print(msg)
        return msg


with gr.Blocks() as demo:
    gr.Markdown("# 字幕翻译工具")
    
    with gr.Tabs():
        # 直接使用标签页
        with gr.TabItem("直接使用"):
            src_lang = config.get('source_language')
            tgt_lang = config.get('target_language')
            src_lang_display = LANG_MAP.get(src_lang, src_lang)
            tgt_lang_display = LANG_MAP.get(tgt_lang, tgt_lang)
            if src_lang == tgt_lang:
                lang_hint = f"🔄 **提示**: 源语言与目标语言相同 ({src_lang_display})，将跳过翻译步骤，仅生成原文字幕"
            else:
                lang_hint = f"🔄 源语言: {src_lang_display} → 目标语言: {tgt_lang_display}"
            gr.Markdown(f"上传视频文件添加到队列，或直接处理\n{lang_hint}\n使用模型: {config.get('model', MODEL_OPTIONS[0] if MODEL_OPTIONS else 'large-v3')} (检测到 {len(MODEL_OPTIONS)} 个可用模型)")

            video = gr.File(label="选择视频文件", file_count="multiple", file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts"])

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
                device = gr.Dropdown(choices=["auto", "cuda", "cpu"], value=config.get('device'), label="计算设备")
                source_language = gr.Dropdown(choices=["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"], value=config.get('source_language'), label="源语言")
                target_language = gr.Dropdown(choices=["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"], value=config.get('target_language'), label="目标语言(当源语言和目标语言相同时跳过翻译)")

            # 语音识别设置
            with gr.Accordion("语音识别设置", open=False):
                enable_forced_alignment = gr.Checkbox(value=config.get('enable_forced_alignment'), label="启用强制对齐(Wav2Vec2)")
                whispercd_alpha = gr.Slider(minimum=0.0, maximum=2.0, value=config.get('whispercd_alpha'), step=0.1, label="对比强度参数")
                whispercd_temperature = gr.Slider(minimum=0.1, maximum=5.0, value=config.get('whispercd_temperature'), step=0.01, label="log-sum-exp温度参数")
                whispercd_snr_db = gr.Slider(minimum=0.0, maximum=30.0, value=config.get('whispercd_snr_db'), step=1.0, label="高斯噪声注入的SNR值")
                whispercd_temporal_shift = gr.Slider(minimum=0.0, maximum=15.0, value=config.get('whispercd_temporal_shift'), step=0.01, label="音频时间移位的秒数")
                whispercd_context_max_tokens = gr.Slider(minimum=64, maximum=400, value=config.get('whispercd_context_max_tokens'), label="上下文最大 token 数")
                whispercd_max_duration = gr.Slider(minimum=2.0, maximum=15.0, value=config.get('whispercd_max_duration'), step=0.5, label="字幕分段最大时长(秒)")
                whispercd_min_duration = gr.Slider(minimum=1.0, maximum=10.0, value=config.get('whispercd_min_duration'), step=0.5, label="字幕合并最小时长(秒)")
                whispercd_gap_threshold = gr.Slider(minimum=0.5, maximum=5.0, value=config.get('whispercd_gap_threshold'), step=0.5, label="跨边界合并间隔(秒)")
                whispercd_particle_chars = gr.Textbox(value=config.get('whispercd_particle_chars'), label="三级分割断点字符(助词等)")

            # 翻译服务设置
            with gr.Accordion("翻译服务设置", open=False):
                translator = gr.Dropdown(choices=["tencent/HY-MT1.5-1.8B-GGUF"], value=config.get('translator'), label="翻译模型")
                translator_quantization = gr.Dropdown(choices=["Q8_0"], value=config.get('translator_quantization'), label="翻译模型量化版本")
                llama_server_host = gr.Textbox(value=config.get('llama_server_host'), label="Llama Server地址")
                llama_server_port = gr.Number(
                    value=config.get('llama_server_port'),
                    label="Llama Server端口",
                    step=1
                )
                llama_server_context_size = gr.Slider(minimum=512, maximum=32768, value=config.get('llama_server_context_size'), label="Llama Server上下文大小")
                llama_server_threads = gr.Slider(minimum=1, maximum=128, value=config.get('llama_server_threads'), label="Llama Server线程数")
                llama_server_ngl = gr.Slider(minimum=0, maximum=999, value=config.get('llama_server_ngl'), step=1, label="GPU卸载层数")
                llama_server_batch_size = gr.Slider(minimum=512, maximum=8192, value=config.get('llama_server_batch_size'), step=512, label="批处理大小")
                llama_server_parallel_slots = gr.Slider(
                    minimum=1, maximum=8, value=1, step=1,
                    label="并行处理槽数",
                    info="单用户建议设为1，释放VRAM提升翻译速度"
                )
                translation_reset_session = gr.Checkbox(
                    value=config.get('translation_reset_session'),
                    label="翻译前重置会话"
                )

            # 翻译策略设置
            with gr.Accordion("翻译策略设置", open=False):
                translation_temperature = gr.Slider(minimum=0.0, maximum=2.0, value=config.get('translation_temperature'), step=0.01, label="翻译温度")
                translation_top_k = gr.Slider(minimum=1, maximum=100, value=config.get('translation_top_k'), step=1, label="Top-K采样")
                translation_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=config.get('translation_top_p'), step=0.01, label="Top-P核采样")
                translation_repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=config.get('translation_repetition_penalty'), step=0.01, label="重复惩罚")
                translation_segment_context_window = gr.Slider(minimum=0, maximum=20, value=config.get('translation_segment_context_window'), step=1, label="上下文片段数量")
                translation_max_retries = gr.Slider(minimum=1, maximum=10, value=config.get('translation_max_retries'), step=1, label="单条最大重试次数")
                translation_max_total_retries = gr.Slider(minimum=3, maximum=30, value=config.get('translation_max_total_retries'), step=1, label="验证失败最大总重试次数")
                translation_max_output_tokens = gr.Slider(minimum=64, maximum=4096, value=config.get('translation_max_output_tokens'), step=64, label="最大输出token数")

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
                    model, device, source_language, target_language,
                    enable_forced_alignment,
                    whispercd_alpha, whispercd_temperature, whispercd_snr_db, whispercd_temporal_shift,
                    whispercd_context_max_tokens, whispercd_max_duration, whispercd_min_duration, whispercd_gap_threshold,
                    whispercd_particle_chars,
                    translator, translator_quantization,
                    llama_server_host, llama_server_port, llama_server_context_size, llama_server_threads,
                    llama_server_ngl, llama_server_batch_size, llama_server_parallel_slots,
                    translation_reset_session,
                    translation_temperature, translation_top_k, translation_top_p, translation_repetition_penalty,
                    translation_segment_context_window, translation_max_retries, translation_max_total_retries,
                    translation_max_output_tokens,
                ],
                outputs=[config_status]
            )
            
            reset_btn.click(
                reset_config,
                outputs=[
                    config_status,
                    model, device, source_language, target_language,
                    enable_forced_alignment,
                    whispercd_alpha, whispercd_temperature, whispercd_snr_db, whispercd_temporal_shift,
                    whispercd_context_max_tokens, whispercd_max_duration, whispercd_min_duration, whispercd_gap_threshold,
                    whispercd_particle_chars,
                    translator, translator_quantization,
                    llama_server_host, llama_server_port, llama_server_context_size, llama_server_threads,
                    llama_server_ngl, llama_server_batch_size, llama_server_parallel_slots,
                    translation_reset_session,
                    translation_temperature, translation_top_k, translation_top_p, translation_repetition_penalty,
                    translation_segment_context_window, translation_max_retries, translation_max_total_retries,
                    translation_max_output_tokens,
                ]
            )

demo.launch(server_name="127.0.0.1", server_port=None, share=False)
