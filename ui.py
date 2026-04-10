# -*- coding: utf-8 -*-
"""
字幕翻译工具 - Gradio UI
"""

import os
import sys
import gradio as gr
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 自动设置本地 ffmpeg 到 PATH
ffmpeg_local = project_root / "ffmpeg" / "ffmpeg-master-latest-win64-gpl" / "bin"
if os.path.exists(ffmpeg_local):
    os.environ["PATH"] = str(ffmpeg_local) + os.pathsep + os.environ.get("PATH", "")

from config import config, MODEL_OPTIONS, TRANSLATOR_MAP, LANGUAGE_OPTIONS, DEVICE_OPTIONS
from utils.queue_manager import QueueManager


def create_ui():
    """创建Gradio界面"""
    
    # 获取配置值
    cfg = config.ui_values()
    all_config = config.get_all()
    
    # 创建队列管理器实例
    queue_manager = QueueManager()
    
    with gr.Blocks(title="字幕翻译工具") as demo:
        
        gr.Markdown("# 🎬 字幕翻译工具")
        gr.Markdown("自动语音识别 + AI翻译，生成高质量字幕文件")
        
        with gr.Tabs():
            # 第一个标签页：视频处理
            with gr.TabItem("视频处理"):
                with gr.Row():
                    # 左侧：输入和配置
                    with gr.Column(scale=1):
                        video_input = gr.File(
                            label="上传视频文件",
                            file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                            type="filepath"
                        )
                        
                        # 处理按钮
                        process_btn = gr.Button("开始处理", variant="primary", size="lg")
                        
                        # 队列管理按钮
                        with gr.Row():
                            add_queue_btn = gr.Button("添加到队列", variant="secondary")
                            clear_queue_btn = gr.Button("清空队列", variant="secondary")
                        
                        # 状态显示
                        status_msg = gr.Textbox(label="状态", interactive=False, lines=3)
                    
                    # 右侧：队列和输出
                    with gr.Column(scale=1):
                        # 队列列表
                        gr.Markdown("### 处理队列")
                        queue_list = gr.Dataframe(
                            headers=["ID", "文件", "状态", "进度"],
                            label="队列",
                            interactive=False
                        )
                        
                        # 处理队列按钮
                        process_queue_btn = gr.Button("处理队列", variant="primary")
                        
                        # 输出文件
                        output_files = gr.File(label="输出文件", interactive=False)
                        
                        # 语言检测结果显示
                        lang_mode = gr.Textbox(
                            label="语言检测模式",
                            value="自动检测" if all_config.get('source_language') == 'auto' else "手动选择",
                            interactive=False
                        )
            
            # 第二个标签页：详细配置
            with gr.TabItem("详细配置"):
                with gr.Column():
                    gr.Markdown("### 详细参数配置")
                    
                    # =========================================================================
                    # 1. 模型与设备配置 (Model & Device Settings)
                    # =========================================================================
                    
                    with gr.Accordion("1. 模型与设备配置", open=True):
                        model_config = gr.Dropdown(
                            label="识别模型",
                            choices=MODEL_OPTIONS,
                            value=all_config.get('model', 'medium'),
                            interactive=True
                        )
                        translator_config = gr.Dropdown(
                            label="翻译模型",
                            choices=list(TRANSLATOR_MAP.keys()),
                            value=cfg[1],
                            interactive=True
                        )
                        translator_quantization = gr.Dropdown(
                            label="翻译模型量化版本",
                            choices=["auto", "Q4_K_M", "Q6_K", "Q8_0"],
                            value=all_config.get('translator_quantization', 'Q6_K'),
                            interactive=True
                        )
                        device_config = gr.Dropdown(
                            label="计算设备",
                            choices=DEVICE_OPTIONS,
                            value=all_config.get('device', 'auto'),
                            interactive=True
                        )
                        source_language_config = gr.Dropdown(
                            label="源语言",
                            choices=[l[0] for l in LANGUAGE_OPTIONS],
                            value=all_config.get('source_language', 'auto'),
                            interactive=True
                        )
                        target_language_config = gr.Dropdown(
                            label="目标语言",
                            choices=[l[0] for l in LANGUAGE_OPTIONS],
                            value=all_config.get('target_language', 'zh'),
                            interactive=True
                        )
                    
                    # =========================================================================
                    # 2. 语音识别配置 (Speech Recognition Settings)
                    # =========================================================================
                    
                    with gr.Accordion("2. 语音识别配置", open=False):
                        vad_filter_config = gr.Checkbox(
                            label="启用VAD语音活动检测",
                            value=all_config.get('vad_filter', True),
                            interactive=True
                        )
                        word_timestamps_config = gr.Checkbox(
                            label="生成单词级时间戳",
                            value=all_config.get('word_timestamps', True),
                            interactive=True
                        )
                        speech_batch_size_config = gr.Slider(
                            label="语音识别批处理大小",
                            minimum=1,
                            maximum=64,
                            value=all_config.get('speech_batch_size', 10),
                            step=1,
                            interactive=True
                        )
                        use_whisperx_config = gr.Checkbox(
                            label="使用WhisperX",
                            value=all_config.get('use_whisperx', True),
                            interactive=True
                        )
                    
                    with gr.Accordion("WhisperX VAD配置", open=False):
                        whisperx_chunk_size = gr.Slider(
                            label="VAD分段大小(秒)",
                            minimum=10,
                            maximum=60,
                            value=all_config.get('whisperx_chunk_size', 30),
                            step=5,
                            interactive=True
                        )
                        whisperx_vad_onset = gr.Slider(
                            label="VAD起始阈值",
                            minimum=0.0,
                            maximum=1.0,
                            value=all_config.get('whisperx_vad_onset', 0.3),
                            step=0.1,
                            interactive=True
                        )
                        whisperx_vad_offset = gr.Slider(
                            label="VAD结束阈值",
                            minimum=0.0,
                            maximum=1.0,
                            value=all_config.get('whisperx_vad_offset', 0.3),
                            step=0.1,
                            interactive=True
                        )
                        whisperx_compute_type = gr.Dropdown(
                            label="计算类型",
                            choices=["float16", "float32", "int8"],
                            value=all_config.get('whisperx_compute_type', 'float16'),
                            interactive=True
                        )
                        whisperx_condition_on_previous_text = gr.Checkbox(
                            label="基于前文条件预测",
                            value=all_config.get('whisperx_condition_on_previous_text', False),
                            interactive=True
                        )
                    
                    with gr.Accordion("WhisperX 文本处理配置", open=False):
                        whisperx_suppress_punctuation = gr.Checkbox(
                            label="抑制标点符号",
                            value=all_config.get('whisperx_suppress_punctuation', True),
                            interactive=True
                        )
                        whisperx_suppress_tokens = gr.Textbox(
                            label="抑制特定token",
                            value=all_config.get('whisperx_suppress_tokens', "-1"),
                            placeholder="-1表示不抑制，多个token用逗号分隔，如'1231,171'",
                            interactive=True
                        )
                        whisperx_initial_prompt = gr.Textbox(
                            label="初始提示文本",
                            value=all_config.get('whisperx_initial_prompt', ""),
                            placeholder="为模型提供上下文提示，可提高特定领域识别率",
                            interactive=True
                        )
                        whisperx_hotwords = gr.Textbox(
                            label="热词/提示词",
                            value=all_config.get('whisperx_hotwords', ""),
                            placeholder="专业术语，用逗号分隔，如: WhisperX, PyAnnote, GPU",
                            interactive=True
                        )
                    
                    # =========================================================================
                    # 3. 翻译配置 (Translation Settings)
                    # =========================================================================
                    
                    with gr.Accordion("3. 翻译配置", open=False):
                        translation_batch_size = gr.Slider(
                            label="翻译批处理大小",
                            minimum=1024,
                            maximum=8192,
                            value=all_config.get('translation_batch_size', 7096),
                            step=512,
                            interactive=True
                        )
                        translation_context_size = gr.Slider(
                            label="翻译模型上下文大小",
                            minimum=1024,
                            maximum=32768,
                            value=all_config.get('translation_context_size', 9000),
                            step=1024,
                            interactive=True
                        )
                        translation_temperature = gr.Slider(
                            label="温度参数",
                            minimum=0.0,
                            maximum=2.0,
                            value=all_config.get('translation_temperature', 0.3),
                            step=0.1,
                            interactive=True
                        )
                        translation_top_k = gr.Slider(
                            label="Top-K采样",
                            minimum=1,
                            maximum=100,
                            value=all_config.get('translation_top_k', 20),
                            step=5,
                            interactive=True
                        )
                        translation_top_p = gr.Slider(
                            label="Top-P采样",
                            minimum=0.0,
                            maximum=1.0,
                            value=all_config.get('translation_top_p', 0.6),
                            step=0.1,
                            interactive=True
                        )
                        translation_repetition_penalty = gr.Slider(
                            label="重复惩罚",
                            minimum=1.0,
                            maximum=2.0,
                            value=all_config.get('translation_repetition_penalty', 1.05),
                            step=0.05,
                            interactive=True
                        )
                    
                    # =========================================================================
                    # 4. 增强版语音识别配置 (Enhanced Speech Recognition Settings)
                    # =========================================================================
                    
                    with gr.Accordion("4. 增强版语音识别配置", open=False):
                        enable_forced_alignment_config = gr.Checkbox(
                            label="启用强制对齐",
                            value=all_config.get('enable_forced_alignment', True),
                            interactive=True
                        )
                        max_segment_duration = gr.Slider(
                            label="最大段落持续时间(秒)",
                            minimum=1.0,
                            maximum=30.0,
                            value=all_config.get('max_segment_duration', 10.0),
                            step=1.0,
                            interactive=True
                        )
                        sentence_pause_threshold = gr.Slider(
                            label="句子停顿阈值(秒)",
                            minimum=0.1,
                            maximum=2.0,
                            value=all_config.get('sentence_pause_threshold', 0.5),
                            step=0.1,
                            interactive=True
                        )
                        min_silence_for_split = gr.Slider(
                            label="断句所需最小静音时长(秒)",
                            minimum=0.1,
                            maximum=1.0,
                            value=all_config.get('min_silence_for_split', 0.3),
                            step=0.1,
                            interactive=True
                        )
                    
                    # 配置保存和重置
                    with gr.Row():
                        save_all_btn = gr.Button("保存所有参数", variant="primary")
                        reset_all_btn = gr.Button("恢复默认配置", variant="secondary")
                    
                    # 配置状态显示
                    with gr.Accordion("配置状态", open=False):
                        config_info = gr.JSON(label="当前配置", value=config.get_all())
                        refresh_btn = gr.Button("刷新状态")
                        refresh_btn.click(fn=lambda: config.get_all(), outputs=[config_info])
                    
                    gr.Markdown("**提示**: 只保存与默认值不同的参数")
            
            # 第三个标签页：参数说明
            with gr.TabItem("参数说明"):
                gr.Markdown("### 参数功能详细说明")
                
                # =========================================================================
                # 1. 模型与设备配置说明
                # =========================================================================
                with gr.Accordion("1. 模型与设备配置", open=True):
                    gr.Markdown("""
                    #### 识别模型
                    - **tiny**: 最小模型，速度最快，准确率最低
                    - **base**: 基础模型，平衡速度和准确率
                    - **small**: 小型模型，准确率较高
                    - **medium**: 中型模型，推荐使用
                    - **large-v2**: 大型模型，准确率高，速度慢
                    - **large-v3**: 最新大型模型，准确率最高
                    
                    #### 翻译模型
                    - 支持多种翻译模型
                    - 根据需求选择合适的模型
                    
                    #### 翻译模型量化版本
                    - **auto**: 自动选择最小模型
                    - **Q4_K_M**: 4位量化，平衡大小和性能
                    - **Q6_K**: 6位量化，默认选项，更好的质量
                    - **Q8_0**: 8位量化，最高质量，最大模型大小
                    
                    #### 计算设备
                    - **auto**: 自动检测GPU可用性，优先使用GPU
                    - **cuda**: 强制使用GPU
                    - **cpu**: 强制使用CPU
                    - 推荐使用GPU以获得最佳性能
                    
                    #### 源语言和目标语言
                    - 源语言：自动检测或手动指定
                    - 目标语言：选择翻译输出的语言
                    - 手动指定源语言可提高识别准确率
                    """)
                
                # =========================================================================
                # 2. 语音识别配置说明
                # =========================================================================
                with gr.Accordion("2. 语音识别配置", open=False):
                    gr.Markdown("""
                    #### 基础语音识别配置
                    - **启用VAD语音活动检测**: 启用后可以过滤掉静音部分，提高识别精度
                    - **生成单词级时间戳**: 启用后会为每个单词生成时间戳，提高字幕准确性
                    - **语音识别批处理大小**: 控制每次处理的音频批次大小，根据GPU显存调整
                    - **使用WhisperX**: 使用WhisperX替代原版Whisper，提供更好的时间戳精度
                    
                    #### WhisperX VAD配置
                    - **VAD分段大小**: 控制VAD处理的分段大小（秒），推荐值：30
                    - **VAD起始阈值**: 控制语音检测的灵敏度，推荐值：0.3
                    - **VAD结束阈值**: 控制语音结束的检测，推荐值：0.3
                    - **计算类型**: 控制计算精度，float16平衡速度和精度（推荐）
                    - **基于前文条件预测**: 使用前文信息辅助识别，提高连续性
                    
                    #### WhisperX 文本处理配置
                    - **抑制标点符号**: 启用后可减少标点符号的识别
                    - **抑制特定token**: 抑制指定的token，多个用逗号分隔
                    - **初始提示文本**: 为模型提供上下文提示，可提高特定领域识别率
                    - **热词/提示词**: 专业术语，用逗号分隔，提高专业术语识别率
                    """)
                
                # =========================================================================
                # 3. 翻译配置说明
                # =========================================================================
                with gr.Accordion("3. 翻译配置", open=False):
                    gr.Markdown("""
                    #### 翻译批处理与上下文配置
                    - **翻译批处理大小**: 控制每次翻译的批次大小，推荐值：4096-8192
                    - **翻译模型上下文大小**: 控制翻译模型的上下文窗口大小，推荐值：4096-9000
                    
                    #### 翻译采样配置
                    - **温度参数**: 控制输出的随机性，字幕翻译建议0.2-0.4
                    - **Top-K采样**: 限制采样范围为概率最高的K个token，推荐值：20
                    - **Top-P采样**: 核采样，累积概率超过P的token都被考虑，推荐值：0.6-0.9
                    - **重复惩罚**: 抑制重复生成，字幕翻译建议1.02-1.1
                    """)
                
                # =========================================================================
                # 4. 增强版语音识别配置说明
                # =========================================================================
                with gr.Accordion("4. 增强版语音识别配置", open=False):
                    gr.Markdown("""
                    #### 强制对齐与段落分割配置
                    - **启用强制对齐**: 使用Wav2Vec2/CTC进行强制对齐，提高时间戳精度
                    - **最大段落持续时间**: 超过此值的段落会被分割，推荐值：8-10秒
                    - **句子停顿阈值**: 用于检测句子边界的停顿时间
                    - **断句所需最小静音时长**: 低于此值的静音不会触发断句，推荐值：0.3秒
                    """)
        
        # 事件处理
        # 队列管理
        def add_to_queue(video):
            if not video:
                return queue_manager.get_queue(), "请先上传视频文件"
            params = {
                'model': all_config.get('model', 'medium'),
                'translator': all_config.get('translator', 'tencent/HY-MT1.5-7B-GGUF'),
                'source_language': all_config.get('source_language', 'auto'),
                'target_language': all_config.get('target_language', 'zh'),
                'device': all_config.get('device', 'auto'),
                'output_path': None,
                'word_timestamps': all_config.get('word_timestamps', True),
                'speech_batch_size': all_config.get('speech_batch_size', 16),
                'whisperx_vad_onset': all_config.get('whisperx_vad_onset', 0.3),
                'enable_forced_alignment': all_config.get('enable_forced_alignment', True),
                'translation_batch_size': all_config.get('translation_batch_size', 4096),
                'translation_context_size': all_config.get('translation_context_size', 4096)
            }
            count = queue_manager.add_to_queue(video, params)
            return queue_manager.get_queue(), f"已添加 {count} 个文件到队列"
        
        def clear_queue():
            queue_manager.clear_queue()
            return queue_manager.get_queue(), "队列已清空"
        
        def process_queue():
            if not queue_manager.video_queue:
                return queue_manager.get_queue(), "队列为空"
            # 处理队列中的第一个文件
            item = queue_manager.video_queue[0]
            item['status'] = '处理中'
            success, msg, output, logs = queue_manager.process_video(item['file_path'], item['params'])
            if success:
                item['status'] = '完成'
            else:
                item['status'] = f'失败: {msg}'
            return queue_manager.get_queue(), "\n".join(logs)
        
        add_queue_btn.click(
            fn=add_to_queue,
            inputs=[video_input],
            outputs=[queue_list, status_msg]
        )
        clear_queue_btn.click(fn=clear_queue, outputs=[queue_list, status_msg])
        process_btn.click(fn=process_queue, outputs=[queue_list, status_msg])
        
        # 保存所有参数
        def do_save_all(model, translator, translator_quantization, device, source_language, target_language,
                       vad_filter, word_timestamps, speech_batch_size, use_whisperx,
                       whisperx_chunk_size, whisperx_vad_onset, whisperx_vad_offset, whisperx_compute_type,
                       whisperx_condition_on_previous_text, whisperx_suppress_punctuation, whisperx_suppress_tokens,
                       whisperx_initial_prompt, whisperx_hotwords,
                       translation_batch_size, translation_context_size, translation_temperature,
                       translation_top_k, translation_top_p, translation_repetition_penalty,
                       enable_forced_alignment, max_segment_duration, sentence_pause_threshold, min_silence_for_split):
            
            # 构建所有参数
            all_params = {
                'model': model,
                'translator': translator,
                'translator_quantization': translator_quantization,
                'device': device,
                'source_language': source_language,
                'target_language': target_language,
                'vad_filter': vad_filter,
                'word_timestamps': word_timestamps,
                'speech_batch_size': speech_batch_size,
                'use_whisperx': use_whisperx,
                'whisperx_chunk_size': whisperx_chunk_size,
                'whisperx_vad_onset': whisperx_vad_onset,
                'whisperx_vad_offset': whisperx_vad_offset,
                'whisperx_compute_type': whisperx_compute_type,
                'whisperx_condition_on_previous_text': whisperx_condition_on_previous_text,
                'whisperx_suppress_punctuation': whisperx_suppress_punctuation,
                'whisperx_suppress_tokens': whisperx_suppress_tokens,
                'whisperx_initial_prompt': whisperx_initial_prompt,
                'whisperx_hotwords': whisperx_hotwords,
                'translation_batch_size': translation_batch_size,
                'translation_context_size': translation_context_size,
                'translation_temperature': translation_temperature,
                'translation_top_k': translation_top_k,
                'translation_top_p': translation_top_p,
                'translation_repetition_penalty': translation_repetition_penalty,
                'enable_forced_alignment': enable_forced_alignment,
                'max_segment_duration': max_segment_duration,
                'sentence_pause_threshold': sentence_pause_threshold,
                'min_silence_for_split': min_silence_for_split
            }
            
            # 保存参数
            success, msg = config.save(**all_params)
            
            # 刷新配置信息
            return msg, config.get_all()
        
        save_all_btn.click(
            fn=do_save_all,
            inputs=[
                model_config, translator_config, translator_quantization, device_config,
                source_language_config, target_language_config,
                vad_filter_config, word_timestamps_config, speech_batch_size_config, use_whisperx_config,
                whisperx_chunk_size, whisperx_vad_onset, whisperx_vad_offset, whisperx_compute_type,
                whisperx_condition_on_previous_text, whisperx_suppress_punctuation, whisperx_suppress_tokens,
                whisperx_initial_prompt, whisperx_hotwords,
                translation_batch_size, translation_context_size, translation_temperature,
                translation_top_k, translation_top_p, translation_repetition_penalty,
                enable_forced_alignment_config, max_segment_duration, sentence_pause_threshold, min_silence_for_split
            ],
            outputs=[status_msg, config_info]
        )
        
        # 恢复默认配置
        def do_reset_all():
            success, msg = config.reset()
            
            # 获取默认配置值
            default_config = config.get_all()
            
            return (
                msg,
                config.get_all(),
                default_config['model'],
                default_config['translator'].split('/')[-1] if '/' in default_config['translator'] else default_config['translator'],
                default_config['translator_quantization'],
                default_config['device'],
                default_config['source_language'],
                default_config['target_language'],
                default_config['vad_filter'],
                default_config['word_timestamps'],
                default_config['speech_batch_size'],
                default_config['use_whisperx'],
                default_config['whisperx_chunk_size'],
                default_config['whisperx_vad_onset'],
                default_config['whisperx_vad_offset'],
                default_config['whisperx_compute_type'],
                default_config['whisperx_condition_on_previous_text'],
                default_config['whisperx_suppress_punctuation'],
                default_config['whisperx_suppress_tokens'],
                default_config['whisperx_initial_prompt'],
                default_config['whisperx_hotwords'],
                default_config['translation_batch_size'],
                default_config['translation_context_size'],
                default_config['translation_temperature'],
                default_config['translation_top_k'],
                default_config['translation_top_p'],
                default_config['translation_repetition_penalty'],
                default_config['enable_forced_alignment'],
                default_config['max_segment_duration'],
                default_config['sentence_pause_threshold'],
                default_config['min_silence_for_split']
            )
        
        reset_all_btn.click(
            fn=do_reset_all,
            outputs=[
                status_msg, config_info,
                model_config, translator_config, translator_quantization, device_config,
                source_language_config, target_language_config,
                vad_filter_config, word_timestamps_config, speech_batch_size_config, use_whisperx_config,
                whisperx_chunk_size, whisperx_vad_onset, whisperx_vad_offset, whisperx_compute_type,
                whisperx_condition_on_previous_text, whisperx_suppress_punctuation, whisperx_suppress_tokens,
                whisperx_initial_prompt, whisperx_hotwords,
                translation_batch_size, translation_context_size, translation_temperature,
                translation_top_k, translation_top_p, translation_repetition_penalty,
                enable_forced_alignment_config, max_segment_duration, sentence_pause_threshold, min_silence_for_split
            ]
        )
        
        video_input.change(fn=lambda: "", outputs=[status_msg])
    
    return demo


def main():
    """主函数"""
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        css="""
        .container { max-width: 1200px; margin: auto; }
        .accordion { margin: 10px 0; }
        .slider { margin: 5px 0; }
        .dropdown { margin: 5px 0; }
        .checkbox { margin: 5px 0; }
        .textbox { margin: 5px 0; }
        
        /* 蓝色按钮样式 */
        .primary-btn, button.primary, .submit-btn, button[type="submit"] {
            background-color: #3b82f6 !important;
            border-color: #3b82f6 !important;
            color: white !important;
        }
        .primary-btn:hover, button.primary:hover, .submit-btn:hover, button[type="submit"]:hover {
            background-color: #2563eb !important;
            border-color: #2563eb !important;
        }
        
        /* Gradio 按钮样式覆盖 */
        .gr-button-primary {
            background-color: #3b82f6 !important;
            border-color: #3b82f6 !important;
        }
        .gr-button-primary:hover {
            background-color: #2563eb !important;
            border-color: #2563eb !important;
        }
        """
    )


if __name__ == "__main__":
    main()
