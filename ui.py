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

import logging
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any

import gradio as gr

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config, MODEL_OPTIONS
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

@dataclass
class SaveConfigParams:
    model: Any = None
    device: Any = None
    source_language: Any = None
    target_language: Any = None
    enable_forced_alignment: Any = None
    whispercd_alpha: Any = None
    whispercd_temperature: Any = None
    whispercd_snr_db: Any = None
    whispercd_temporal_shift: Any = None
    whispercd_context_max_tokens: Any = None
    whispercd_max_duration: Any = None
    whispercd_min_duration: Any = None
    whispercd_merge_max_duration: Any = None
    whispercd_gap_threshold: Any = None
    whispercd_particle_chars: Any = None
    whispercd_max_token_repeat: Any = None
    whispercd_long_seq_window: Any = None
    whispercd_long_seq_threshold: Any = None
    whispercd_target_token_count: Any = None
    whispercd_search_range: Any = None
    whispercd_low_confidence_threshold: Any = None
    whispercd_continuation_gap_multiplier: Any = None
    translator: Any = None
    llama_server_host: Any = None
    llama_server_port: Any = None
    llama_server_context_size: Any = None
    llama_server_threads: Any = None
    llama_server_ngl: Any = None
    llama_server_batch_size: Any = None
    llama_server_parallel_slots: Any = None
    translation_reset_session: Any = None
    translation_temperature: Any = None
    translation_top_k: Any = None
    translation_top_p: Any = None
    translation_repetition_penalty: Any = None
    translation_segment_context_window: Any = None
    translation_max_context_tokens: Any = None
    translation_max_retries: Any = None
    translation_max_total_retries: Any = None
    translation_max_output_tokens: Any = None
    translation_validation_threshold: Any = None
    translation_short_text_threshold: Any = None
    translation_kana_ratio_threshold: Any = None
    translation_request_timeout: Any = None

    @classmethod
    def from_config(cls, config_manager):
        return cls(**{f.name: config_manager.get(f.name) for f in fields(cls)})

    def to_tuple(self):
        return tuple(getattr(self, f.name) for f in fields(cls))


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
        logging.error("添加文件失败", exc_info=True)
        msg = "错误：添加文件失败，请查看日志获取详情"
        print(msg)
        return msg


def clear_queue():
    try:
        count = queue_manager.clear_queue()
        msg = f"成功：队列已清空 (清除了 {count} 个文件)"
        print(msg)
        return msg
    except Exception as e:
        logging.error("清空队列失败", exc_info=True)
        msg = "错误：清空队列失败，请查看日志获取详情"
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
        logging.error("处理队列失败", exc_info=True)
        msg = "错误：处理队列失败，请查看日志获取详情"
        print(msg)
        return msg


def save_config(params: SaveConfigParams):
    try:
        params_dict = asdict(params)
        params_dict['llama_server_port'] = int(params_dict['llama_server_port']) if params_dict['llama_server_port'] is not None else None
        success, msg = config.save(**params_dict)
        if success:
            return f"成功：{msg}"
        else:
            return f"错误：{msg}"
    except Exception as e:
        logging.error("保存配置失败", exc_info=True)
        msg = "错误：保存配置失败，请查看日志获取详情"
        print(msg)
        return msg


def reset_config():
    try:
        success, msg = config.reset()
        if success:
            params = SaveConfigParams.from_config(config)
            return (f"成功：{msg}",) + params.to_tuple()
        else:
            return (f"错误：{msg}",) + (None,) * len(fields(SaveConfigParams))
    except Exception as e:
        logging.error("重置配置失败", exc_info=True)
        msg = "错误：重置配置失败，请查看日志获取详情"
        return (msg,) + (None,) * len(fields(SaveConfigParams))


def cancel_processing():
    try:
        queue_manager.cancel_processing()
        msg = "成功：已发送取消请求"
        print(msg)
        return msg
    except Exception as e:
        logging.error("取消失败", exc_info=True)
        msg = "错误：取消失败，请查看日志获取详情"
        print(msg)
        return msg


_STR_FIELDS = {'model', 'device', 'source_language', 'target_language', 'translator', 'llama_server_host', 'whispercd_particle_chars'}
_INT_FIELDS = {'llama_server_port', 'whispercd_context_max_tokens', 'whispercd_max_token_repeat', 'whispercd_long_seq_window', 'whispercd_target_token_count', 'whispercd_search_range', 'llama_server_context_size', 'llama_server_threads', 'llama_server_ngl', 'llama_server_batch_size', 'llama_server_parallel_slots', 'translation_top_k', 'translation_segment_context_window', 'translation_max_context_tokens', 'translation_max_retries', 'translation_max_total_retries', 'translation_max_output_tokens', 'translation_short_text_threshold', 'translation_request_timeout'}
_FLOAT_FIELDS = {'whispercd_alpha', 'whispercd_temperature', 'whispercd_snr_db', 'whispercd_temporal_shift', 'whispercd_max_duration', 'whispercd_min_duration', 'whispercd_merge_max_duration', 'whispercd_gap_threshold', 'whispercd_long_seq_threshold', 'whispercd_low_confidence_threshold', 'whispercd_continuation_gap_multiplier', 'translation_temperature', 'translation_top_p', 'translation_repetition_penalty', 'translation_validation_threshold', 'translation_kana_ratio_threshold'}
_BOOL_FIELDS = {'enable_forced_alignment', 'translation_reset_session'}


def _gradio_save_config(*args):
    field_names = [f.name for f in fields(SaveConfigParams)]
    coerced = []
    for name, val in zip(field_names, args):
        if name in _STR_FIELDS:
            val = str(val) if val is not None else val
        elif name in _INT_FIELDS:
            val = int(val) if val is not None else val
        elif name in _FLOAT_FIELDS:
            val = float(val) if val is not None else val
        elif name in _BOOL_FIELDS:
            val = bool(val) if val is not None else val
        coerced.append(val)
    params = SaveConfigParams(**dict(zip(field_names, coerced)))
    return save_config(params)


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
                whispercd_alpha = gr.Slider(minimum=0.0, maximum=2.0, value=float(config.get('whispercd_alpha', 1.0)), step=0.1, label="对比强度参数")
                whispercd_temperature = gr.Slider(minimum=0.1, maximum=5.0, value=float(config.get('whispercd_temperature', 1.0)), step=0.01, label="log-sum-exp温度参数")
                whispercd_snr_db = gr.Slider(minimum=0.0, maximum=30.0, value=float(config.get('whispercd_snr_db', 10.0)), step=1.0, label="高斯噪声注入的SNR值")
                whispercd_temporal_shift = gr.Slider(minimum=0.0, maximum=15.0, value=float(config.get('whispercd_temporal_shift', 7.0)), step=0.01, label="音频时间移位的秒数")
                whispercd_context_max_tokens = gr.Slider(minimum=64, maximum=400, value=int(config.get('whispercd_context_max_tokens', 200)), label="上下文最大 token 数")
                whispercd_max_duration = gr.Slider(minimum=2.0, maximum=15.0, value=float(config.get('whispercd_max_duration', 6.0)), step=0.5, label="字幕最大时长(秒)", info="超过必须分割")
                whispercd_min_duration = gr.Slider(minimum=0.5, maximum=5.0, value=float(config.get('whispercd_min_duration', 1.5)), step=0.1, label="碎片阈值(秒)", info="低于此值视为碎片")
                whispercd_merge_max_duration = gr.Slider(minimum=1.0, maximum=15.0, value=float(config.get('whispercd_merge_max_duration', 8.0)), step=0.5, label="合并安全上限(秒)", info="合并后绝不超过此值")
                whispercd_gap_threshold = gr.Slider(minimum=0.5, maximum=5.0, value=float(config.get('whispercd_gap_threshold', 2.0)), step=0.5, label="跨边界合并间隔(秒)")
                whispercd_particle_chars = gr.Textbox(value=config.get('whispercd_particle_chars'), label="三级分割断点字符(助词等)")

                with gr.Accordion("重复抑制", open=False):
                    whispercd_max_token_repeat = gr.Slider(minimum=2, maximum=10, value=int(config.get('whispercd_max_token_repeat', 3)), step=1, label="单token重复抑制阈值", info="同一token出现次数≥此值时抑制")
                    whispercd_long_seq_window = gr.Slider(minimum=5, maximum=50, value=int(config.get('whispercd_long_seq_window', 20)), step=1, label="长序列检测窗口", info="滑动窗口大小")
                    whispercd_long_seq_threshold = gr.Slider(minimum=0.3, maximum=1.0, value=float(config.get('whispercd_long_seq_threshold', 0.7)), step=0.05, label="长序列相似度阈值", info="高于此值判定为重复")

                with gr.Accordion("分割策略", open=False):
                    whispercd_target_token_count = gr.Slider(minimum=10, maximum=50, value=int(config.get('whispercd_target_token_count', 25)), step=1, label="助词级分割目标token数", info="控制分割后每段的大致token数量")
                    whispercd_search_range = gr.Slider(minimum=2, maximum=15, value=int(config.get('whispercd_search_range', 6)), step=1, label="助词级分割搜索范围", info="在目标位置前后搜索的token数")
                    whispercd_low_confidence_threshold = gr.Slider(minimum=-2.0, maximum=0.0, value=float(config.get('whispercd_low_confidence_threshold', -0.5)), step=0.1, label="低置信度警告阈值", info="avg_logprob低于此值触发警告")
                    whispercd_continuation_gap_multiplier = gr.Slider(minimum=1.0, maximum=3.0, value=float(config.get('whispercd_continuation_gap_multiplier', 1.5)), step=0.1, label="续行间隔放宽倍数", info="检测到续行助词时gap_threshold乘以此值")

            # 翻译服务设置
            with gr.Accordion("翻译服务设置", open=False):
                translator = gr.Dropdown(choices=["tencent/HY-MT1.5-1.8B-GGUF-Q8_0", "SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF-Q6_K"], value=config.get('translator'), label="翻译模型")
                llama_server_host = gr.Textbox(value=config.get('llama_server_host'), label="Llama Server地址")
                llama_server_port = gr.Number(
                    value=int(config.get('llama_server_port', 8080)),
                    label="Llama Server端口",
                    step=1
                )
                llama_server_context_size = gr.Slider(minimum=512, maximum=32768, value=int(config.get('llama_server_context_size', 4096)), label="Llama Server上下文大小")
                llama_server_threads = gr.Slider(minimum=1, maximum=128, value=int(config.get('llama_server_threads', 8)), label="Llama Server线程数")
                llama_server_ngl = gr.Slider(minimum=0, maximum=999, value=int(config.get('llama_server_ngl', 99)), step=1, label="GPU卸载层数")
                llama_server_batch_size = gr.Slider(minimum=512, maximum=8192, value=int(config.get('llama_server_batch_size', 512)), step=512, label="批处理大小")
                llama_server_parallel_slots = gr.Slider(
                    minimum=1, maximum=8, value=int(config.get('llama_server_parallel_slots', 1)), step=1,
                    label="并行处理槽数",
                    info="单用户建议设为1，释放VRAM提升翻译速度"
                )
                translation_reset_session = gr.Checkbox(
                    value=config.get('translation_reset_session'),
                    label="翻译前重置会话"
                )

            # 翻译策略设置
            with gr.Accordion("翻译策略设置", open=False):
                translation_temperature = gr.Slider(minimum=0.0, maximum=2.0, value=float(config.get('translation_temperature', 0.1)), step=0.01, label="翻译温度")
                translation_top_k = gr.Slider(minimum=1, maximum=100, value=int(config.get('translation_top_k', 20)), step=1, label="Top-K采样")
                translation_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=float(config.get('translation_top_p', 0.6)), step=0.01, label="Top-P核采样")
                translation_repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=float(config.get('translation_repetition_penalty', 1.05)), step=0.01, label="重复惩罚")
                translation_segment_context_window = gr.Slider(minimum=0, maximum=20, value=int(config.get('translation_segment_context_window', 3)), step=1, label="上下文片段数量")
                translation_max_context_tokens = gr.Slider(minimum=0, maximum=4096, value=int(config.get('translation_max_context_tokens', 512)), step=64, label="上下文最大token数", info="限制上下文片段占用的最大token数，超出时自动截断，设为0则不使用上下文")
                translation_max_retries = gr.Slider(minimum=1, maximum=10, value=int(config.get('translation_max_retries', 3)), step=1, label="单条最大重试次数")
                translation_max_total_retries = gr.Slider(minimum=3, maximum=30, value=int(config.get('translation_max_total_retries', 3)), step=1, label="验证失败最大总重试次数")
                translation_max_output_tokens = gr.Slider(minimum=64, maximum=4096, value=int(config.get('translation_max_output_tokens', 512)), step=64, label="最大输出token数")

                with gr.Accordion("验证与超时", open=False):
                    translation_validation_threshold = gr.Slider(minimum=0.1, maximum=1.0, value=float(config.get('translation_validation_threshold', 0.5)), step=0.05, label="翻译验证阈值", info="目标语言占比低于此值判定为翻译失败")
                    translation_short_text_threshold = gr.Slider(minimum=1, maximum=20, value=int(config.get('translation_short_text_threshold', 5)), step=1, label="短文本判断阈值", info="低于此字符数视为短文本特殊处理")
                    translation_kana_ratio_threshold = gr.Slider(minimum=0.1, maximum=0.8, value=float(config.get('translation_kana_ratio_threshold', 0.3)), step=0.05, label="日语假名占比阈值", info="日译中时假名占比高于此值判定为未翻译")
                    translation_request_timeout = gr.Slider(minimum=30, maximum=600, value=int(config.get('translation_request_timeout', 300)), step=10, label="HTTP请求超时(秒)", info="翻译请求超时时间")

            # 配置操作按钮
            with gr.Row():
                save_btn = gr.Button("保存配置", variant="primary")
                reset_btn = gr.Button("重置默认", variant="secondary")
            
            # 配置状态信息
            config_status = gr.Textbox(label="配置状态", lines=3, interactive=False)

            # 绑定按钮事件
            save_btn.click(
                _gradio_save_config,
                inputs=[
                    model, device, source_language, target_language,
                    enable_forced_alignment,
                    whispercd_alpha, whispercd_temperature, whispercd_snr_db, whispercd_temporal_shift,
                    whispercd_context_max_tokens, whispercd_max_duration, whispercd_min_duration, whispercd_merge_max_duration, whispercd_gap_threshold,
                    whispercd_particle_chars,
                    whispercd_max_token_repeat, whispercd_long_seq_window, whispercd_long_seq_threshold,
                    whispercd_target_token_count, whispercd_search_range, whispercd_low_confidence_threshold, whispercd_continuation_gap_multiplier,
                    translator,
                    llama_server_host, llama_server_port, llama_server_context_size, llama_server_threads,
                    llama_server_ngl, llama_server_batch_size, llama_server_parallel_slots,
                    translation_reset_session,
                    translation_temperature, translation_top_k, translation_top_p, translation_repetition_penalty,
                    translation_segment_context_window, translation_max_context_tokens, translation_max_retries, translation_max_total_retries,
                    translation_max_output_tokens,
                    translation_validation_threshold, translation_short_text_threshold, translation_kana_ratio_threshold, translation_request_timeout,
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
                    whispercd_context_max_tokens, whispercd_max_duration, whispercd_min_duration, whispercd_merge_max_duration, whispercd_gap_threshold,
                    whispercd_particle_chars,
                    whispercd_max_token_repeat, whispercd_long_seq_window, whispercd_long_seq_threshold,
                    whispercd_target_token_count, whispercd_search_range, whispercd_low_confidence_threshold, whispercd_continuation_gap_multiplier,
                    translator,
                    llama_server_host, llama_server_port, llama_server_context_size, llama_server_threads,
                    llama_server_ngl, llama_server_batch_size, llama_server_parallel_slots,
                    translation_reset_session,
                    translation_temperature, translation_top_k, translation_top_p, translation_repetition_penalty,
                    translation_segment_context_window, translation_max_context_tokens, translation_max_retries, translation_max_total_retries,
                    translation_max_output_tokens,
                    translation_validation_threshold, translation_short_text_threshold, translation_kana_ratio_threshold, translation_request_timeout,
                ]
            )

demo.launch(server_name="127.0.0.1", server_port=None, share=False)
