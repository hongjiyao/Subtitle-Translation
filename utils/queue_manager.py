#!/usr/bin/env python3
"""
队列管理器模块
负责管理视频处理队列，处理队列中的视频文件
"""
import os
import sys

# 确保在导入任何库之前设置HF-Mirror作为下载源
# 这些环境变量需要在导入其他库之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

import time
import subprocess
from config import DEFAULT_CONFIG, TEMP_DIR, OUTPUT_DIR
from utils.video_processor import extract_audio
from utils.speech_recognizer import recognize_speech
from utils.translator import translate_text
from utils.subtitle_generator import generate_subtitle

# 翻译模型映射
translator_models = {
    "m2m100_418M": "facebook/m2m100_418M",
    "m2m100_1.2B": "facebook/m2m100_1.2B"
}

# 模型选项
model_options = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
translator_options = list(translator_models.keys())

class QueueManager:
    """队列管理器类"""
    def __init__(self):
        self.video_queue = []
        self.queue_processing = False
        self.current_queue_index = -1
        self.max_queue_length = 50  # 队列最大长度限制
        self.logs = []
        self.progress = 0
    
    def log_message(self, message):
        """添加日志消息"""
        try:
            if message and isinstance(message, str):
                self.logs.append(message)
                
                # 直接打印日志到终端，不需要任何频率控制
                print(f"[处理日志] {message}")
                
                # 限制日志数量，避免内存占用过大
                if len(self.logs) > 500:
                    self.logs = self.logs[-500:]
            return "\n".join(self.logs)
        except Exception as e:
            # 忽略日志处理时的错误
            print(f"[错误信息] 日志处理出错: {str(e)}")
            return ""
    
    def update_progress(self, value):
        """更新进度"""
        try:
            if isinstance(value, (int, float)):
                # 确保进度值在0-100之间
                self.progress = max(0, min(100, value))
            return self.progress
        except Exception as e:
            # 忽略进度更新时的错误
            print(f"[错误信息] 进度更新出错: {str(e)}")
            return 0
    
    def get_queue(self):
        """获取队列状态
        
        Returns:
            list: 队列中的所有项
        """
        return self.video_queue
    
    def ensure_process_termination(self, process):
        """确保进程能够正确终止"""
        if not process or process.poll() is not None:
            return
        
        try:
            print(f"[进程管理] 正在终止进程 PID: {process.pid}")
        except:
            pass
        
        try:
            # 尝试优雅终止
            process.terminate()
            
            # 快速等待进程终止
            for i in range(5):  # 最多等待5秒
                if process.poll() is not None:
                    try:
                        print(f"[进程管理] 进程 PID: {process.pid} 已终止，返回码: {process.returncode}")
                    except:
                        pass
                    return
                if i == 0:
                    try:
                        print(f"[进程管理] 等待进程 PID: {process.pid} 终止...")
                    except:
                        pass
            
            # 如果进程仍在运行，尝试强制终止
            if process.poll() is None:
                try:
                    print(f"[进程管理] 强制终止进程 PID: {process.pid}")
                except:
                    pass
                process.kill()
                # 再次等待进程终止
                for i in range(3):
                    if process.poll() is not None:
                        try:
                            print(f"[进程管理] 进程 PID: {process.pid} 已强制终止，返回码: {process.returncode}")
                        except:
                            pass
                        return
            
            # 最终检查
            if process.poll() is None:
                try:
                    print(f"[进程管理] 警告: 进程 PID: {process.pid} 可能仍在运行")
                    # 再次尝试强制终止
                    process.kill()
                    if process.poll() is None:
                        print(f"[进程管理] 严重警告: 进程 PID: {process.pid} 无法终止")
                    else:
                        print(f"[进程管理] 进程 PID: {process.pid} 最终已终止")
                except Exception as e:
                    print(f"[进程管理] 最终终止尝试失败: {str(e)}")
        except Exception as e:
            try:
                print(f"[进程管理] 终止进程时出错：{str(e)}")
                # 尝试备选终止方法
                if process.poll() is None:
                    try:
                        process.kill()
                        print(f"[进程管理] 已尝试备选终止方法")
                    except:
                        pass
            except:
                pass
    
    def update_progress_based_on_output(self, line, current_progress):
        """基于输出更新进度"""
        # 更详细的阶段进度定义
        stage_progress = {
            '开始处理': 5,
            '提取音频': 25,
            '音频提取完成': 30,
            '语音识别': 50,
            '语音识别完成': 55,
            '翻译日文': 75,
            '翻译完成': 80,
            '生成字幕': 90,
            '字幕生成完成': 95
        }  # 各阶段的目标进度
        
        stage_updated = False
        if not line:
            return current_progress, stage_updated
        
        # 标准化行内容，处理可能的编码问题
        normalized_line = line.strip().lower()
        
        for stage, target_progress in stage_progress.items():
            stage_keywords = {
                '开始处理': ['开始', 'starting'],
                '提取音频': ['提取音频', 'ȡƵ', 'extract', 'audio'],
                '音频提取完成': ['音频提取完成', 'audio extracted'],
                '语音识别': ['语音识别', 'recogniz', 'transcrib'],
                '语音识别完成': ['识别完成', 'transcript completed'],
                '翻译日文': ['翻译', 'translate', '日文', 'japanese'],
                '翻译完成': ['翻译完成', 'translation completed'],
                '生成字幕': ['生成字幕', 'subtitle', 'srt'],
                '字幕生成完成': ['字幕完成', 'subtitle completed']
            }
            
            # 检查是否匹配当前阶段的任何关键词
            if any(keyword in normalized_line for keyword in stage_keywords.get(stage, [stage.lower()])):
                if target_progress > current_progress:
                    # 平滑过渡到目标进度
                    while current_progress < target_progress:
                        current_progress += 1
                        self.update_progress(current_progress)
                    stage_updated = True
                break
        
        return current_progress, stage_updated
    
    def add_to_queue(self, video_files, params):
        """添加文件到队列
        
        Args:
            video_files: 视频文件路径列表或单个文件路径
            params: 处理参数字典，包含以下键：
                - model: 语音识别模型
                - translator: 翻译模型
                - beam_size: 语音识别beam size
                - vad_filter: 是否启用VAD过滤
                - word_timestamps: 是否启用单词时间戳
                - condition_on_previous_text: 是否基于先前文本
                - translation_beam_size: 翻译beam size
                - translation_max_length: 翻译最大长度
                - translation_early_stopping: 是否启用翻译早停
                - device: 设备选择
        
        Returns:
            int: 添加成功的文件数量
        """
        # 打印所有信息到终端
        print("\n" + "="*80)
        print("[队列信息] 用户执行了'添加到队列'操作")
        print("[队列信息] 选择的视频文件:")
        if isinstance(video_files, list):
            for i, file in enumerate(video_files):
                print(f"  {i+1}. {file}")
        else:
            print(f"  {video_files}")
        print(f"[队列信息] 使用的模型: {params.get('model', '未知')}")
        print(f"[队列信息] 使用的翻译模型: {params.get('translator', '未知')}")
        print(f"[队列信息] 使用的设备: {params.get('device', '未知')}")
        print(f"[队列信息] Beam size: {params.get('beam_size', '未知')}")
        print(f"[队列信息] VAD filter: {params.get('vad_filter', '未知')}")
        print(f"[队列信息] Word timestamps: {params.get('word_timestamps', '未知')}")
        print(f"[队列信息] Condition on previous text: {params.get('condition_on_previous_text', '未知')}")
        print(f"[队列信息] Translation beam size: {params.get('translation_beam_size', '未知')}")
        print(f"[队列信息] Translation max length: {params.get('translation_max_length', '未知')}")
        print(f"[队列信息] Translation early stopping: {params.get('translation_early_stopping', '未知')}")
        print("="*80)
        
        try:
            if not video_files:
                error_msg = "未选择视频文件"
                print(f"[错误信息] {error_msg}")
                return 0
            
            # 确保video_files是列表
            if isinstance(video_files, str):
                video_files = [video_files]
            elif not isinstance(video_files, list):
                return 0
            
            # 处理文件路径
            processed_files = []
            for file_item in video_files:
                if isinstance(file_item, str):
                    # 直接是文件路径字符串
                    processed_files.append(file_item)
                elif hasattr(file_item, 'name'):
                    # 文件对象，使用name属性获取路径
                    processed_files.append(file_item.name)
                elif isinstance(file_item, dict) and 'path' in file_item:
                    # 包含path键的字典
                    processed_files.append(file_item['path'])
                elif isinstance(file_item, dict) and 'name' in file_item:
                    # 包含name键的字典
                    processed_files.append(file_item['name'])
            
            # 使用处理后的文件列表
            video_files = processed_files
            
            # 再次检查是否有有效文件
            if not video_files:
                return 0
            
            # 检查队列长度限制
            if len(self.video_queue) >= self.max_queue_length:
                return 0
            
            # 计算剩余可添加的文件数量
            remaining_slots = self.max_queue_length - len(self.video_queue)
            if len(video_files) > remaining_slots:
                video_files = video_files[:remaining_slots]  # 只取剩余可添加的数量
                print(f"[警告信息] 队列空间不足，只添加前 {remaining_slots} 个文件")
            
            added_count = 0
            for video_file in video_files:
                try:
                    # 检查文件是否存在
                    if not video_file or not isinstance(video_file, str):
                        continue
                    
                    if not os.path.exists(video_file):
                        print(f"[警告信息] 文件不存在: {video_file}")
                        continue
                    
                    # 检查文件是否是普通文件
                    if not os.path.isfile(video_file):
                        print(f"[警告信息] 不是普通文件: {video_file}")
                        continue
                    
                    # 检查文件扩展名
                    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpg', '.mpeg']
                    ext = os.path.splitext(video_file)[1].lower()
                    if ext not in video_extensions:
                        print(f"[警告信息] 不支持的文件格式: {ext}")
                        continue
                    
                    # 检查文件大小
                    try:
                        file_size = os.path.getsize(video_file)
                        # 限制文件大小为10GB
                        if file_size > 10 * 1024 * 1024 * 1024:
                            print(f"[警告信息] 文件过大: {video_file}")
                            continue
                    except Exception as e:
                        # 忽略文件大小检查错误
                        print(f"[警告信息] 检查文件大小时出错: {str(e)}")
                        pass
                    
                    # 添加到队列
                    queue_item = {
                        'file_path': video_file,
                        'filename': os.path.basename(video_file),
                        'status': '等待中',
                        'params': params
                    }
                    self.video_queue.append(queue_item)
                    added_count += 1
                    print(f"[成功信息] 添加文件到队列: {video_file}")
                except Exception as e:
                    # 处理单个文件添加时的异常
                    print(f"[错误信息] 添加文件到队列时出错: {str(e)}")
                    continue
            
            # 只打印到终端，不显示在UI中
            print(f"[队列状态] 已添加 {added_count} 个文件到队列")
            
            return added_count
        except Exception as e:
            # 处理整体异常，只打印到终端，不显示在UI中
            error_msg = f"添加文件到队列时出错: {str(e)}"
            print(f"[错误信息] {error_msg}")
            return 0
    
    def remove_from_queue(self, index):
        """从队列删除文件
        
        Args:
            index: 要删除的队列项索引
        
        Returns:
            None
        """
        # 打印所有信息到终端
        print("\n" + "="*80)
        print("[队列信息] 用户执行了'从队列删除'操作")
        print(f"[队列信息] 删除索引: {index}")
        print(f"[队列信息] 当前队列长度: {len(self.video_queue)}")
        if self.video_queue:
            print("[队列信息] 当前队列内容:")
            for i, item in enumerate(self.video_queue):
                try:
                    filename = item.get('filename', '未知文件名')
                    status = item.get('status', '未知状态')
                    print(f"  {i}. {filename} - {status}")
                except:
                    print(f"  {i}. {item}")
        print("="*80)
        
        try:
            if not self.video_queue:
                error_msg = "队列为空，无法删除文件"
                print(f"[错误信息] {error_msg}")
                return
            
            try:
                # 处理不同类型的索引输入
                if isinstance(index, str):
                    index = index.strip()
                    if not index.isdigit():
                        print(f"[错误信息] 请输入有效的数字索引")
                        return
                    index = int(index)
                elif not isinstance(index, (int, float)):
                    print(f"[错误信息] 索引必须是数字")
                    return
                else:
                    index = int(index)
                
                if 0 <= index < len(self.video_queue):
                    # 记录被删除的文件名
                    removed_file = self.video_queue[index]['filename']
                    self.video_queue.pop(index)
                    print(f"[成功信息] 已从队列删除文件: {removed_file}")
                else:
                    error_msg = f"索引超出范围，队列长度为 {len(self.video_queue)}"
                    print(f"[错误信息] {error_msg}")
            except ValueError as e:
                print(f"[错误信息] 请输入有效的索引: {str(e)}")
            except Exception as e:
                print(f"[错误信息] 删除文件时出错: {str(e)}")
        except Exception as e:
            # 处理整体异常，只打印到终端，不显示在UI中
            error_msg = f"从队列删除文件时出错: {str(e)}"
            print(f"[错误信息] {error_msg}")
    
    def clear_queue(self):
        """清空队列
        
        Returns:
            int: 清除的文件数量
        """
        # 打印所有信息到终端
        print("\n" + "="*80)
        print("[队列信息] 用户执行了'清空队列'操作")
        print(f"[队列信息] 清空前队列长度: {len(self.video_queue)}")
        if self.video_queue:
            print("[队列信息] 清空前队列内容:")
            for i, item in enumerate(self.video_queue):
                try:
                    filename = item.get('filename', '未知文件名')
                    status = item.get('status', '未知状态')
                    print(f"  {i}. {filename} - {status}")
                except:
                    print(f"  {i}. {item}")
        print("="*80)
        
        try:
            # 记录清空前的队列长度
            cleared_count = len(self.video_queue)
            self.video_queue = []
            
            if cleared_count > 0:
                print(f"[成功信息] 队列已清空，共删除 {cleared_count} 个文件")
            else:
                print(f"[信息] 队列为空")
            
            return cleared_count
        except Exception as e:
            # 处理异常，只打印到终端，不显示在UI中
            error_msg = f"清空队列时出错: {str(e)}"
            print(f"[错误信息] {error_msg}")
            return 0
    
    def process_video(self, video_file, params):
        """处理单个视频文件
        
        Args:
            video_file: 视频文件路径
            params: 处理参数字典
        
        Returns:
            tuple: (成功标志, 消息, 输出路径, 日志)
        """
        # 定义一个内部进度回调函数，用于实时更新进度
        def progress_callback(progress):
            """进度回调函数"""
            self.update_progress(progress)
            print(f"[进度更新] 当前进度: {progress}%")
        # 在函数开始处导入必要的模块
        import os
        import sys
        import time
        from tqdm import tqdm
        from utils.video_processor import extract_audio
        from utils.speech_recognizer import recognize_speech
        from utils.translator import translate_text
        from utils.subtitle_generator import generate_subtitle
        from config import DEFAULT_CONFIG, TEMP_DIR, OUTPUT_DIR
        
        try:
            # 验证输入
            if not video_file:
                error_msg = "未选择视频文件"
                print(f"[错误信息] {error_msg}")
                return False, "未选择视频文件", None, []
            
            if not os.path.exists(video_file):
                error_msg = f"文件不存在：{video_file}"
                print(f"[错误信息] {error_msg}")
                return False, "文件不存在", None, []
            
            # 检查文件扩展名
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpg', '.mpeg']
            ext = os.path.splitext(video_file)[1].lower()
            if ext not in video_extensions:
                error_msg = f"不支持的文件格式：{ext}，请选择视频文件"
                print(f"[错误信息] {error_msg}")
                return False, "不支持的文件格式，请选择视频文件", None, []
            
            # 参数验证
            try:
                beam_size = int(params['beam_size'])
                translation_beam_size = int(params['translation_beam_size'])
                translation_max_length = int(params['translation_max_length'])
                
                # 验证beam_size范围
                if beam_size < 1 or beam_size > 10:
                    return False, "语音识别beam size必须在1-10之间", None, []
                
                # 验证translation_beam_size范围
                if translation_beam_size < 1 or translation_beam_size > 10:
                    return False, "翻译beam size必须在1-10之间", None, []
                
                # 验证translation_max_length范围
                if translation_max_length < 50 or translation_max_length > 1000:
                    return False, "翻译最大长度必须在50-1000之间", None, []
                
                # 验证device值
                if params['device'] not in ["auto", "cpu", "cuda"]:
                    return False, "设备选择必须是auto、cpu或cuda", None, []
                    
                # 验证模型值
                if params['model'] not in model_options:
                    return False, "语音识别模型选择错误", None, []
                    
                # 验证翻译模型值
                # 允许使用完整的模型名称或简称
                if params['translator'] not in translator_options and params['translator'] not in translator_models.values():
                    return False, "翻译模型选择错误", None, []
            except ValueError as e:
                error_msg = f"参数错误：{str(e)}"
                print(f"[错误信息] {error_msg}")
                return False, "参数错误，请检查输入", None, []
            except Exception as e:
                error_msg = f"参数验证出错：{str(e)}"
                print(f"[错误信息] {error_msg}")
                return False, "参数验证出错", None, []
            
            # 获取实际的翻译模型路径
            try:
                # 检查是否是完整的模型名称
                if params['translator'] in translator_models.values():
                    # 已经是完整的模型名称
                    translator_model = params['translator']
                else:
                    # 尝试使用简称查找
                    translator_model = translator_models.get(params['translator'], params['translator'])
            except Exception as e:
                error_msg = f"获取翻译模型路径时出错：{str(e)}"
                print(f"[错误信息] {error_msg}")
                return False, "获取翻译模型路径时出错", None, []
            
            # 生成输出路径
            try:
                # 检查是否有用户自定义的输出路径
                if 'output_path' in params and params['output_path']:
                    # 使用用户自定义的输出路径
                    output_path = params['output_path']
                    # 确保输出目录存在
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                else:
                    # 使用默认输出路径
                    base_name = os.path.splitext(os.path.basename(video_file))[0]
                    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_subtitles.srt")
            except Exception as e:
                error_msg = f"生成输出路径时出错：{str(e)}"
                print(f"[错误信息] {error_msg}")
                return False, "生成输出路径时出错", None, []
            
            self.log_message(f"开始处理视频: {video_file}")
            self.log_message(f"输出路径: {output_path}")
            self.log_message(f"使用模型: {params['model']}")
            self.log_message(f"使用翻译模型: {translator_model}")
            self.log_message(f"设备选择: {params['device']}")
            
            # 直接调用处理逻辑，不使用 subprocess
            try:
                # 确保临时目录存在
                os.makedirs(TEMP_DIR, exist_ok=True)
                
                # 1. 提取音频
                print("[处理阶段] 1. 提取音频...")
                self.log_message("1. 提取音频...")
                start_time = time.time()
                
                try:
                    audio_path = extract_audio(video_file)
                except Exception as e:
                    error_msg = f"音频提取失败 - {str(e)}"
                    print(f"[错误信息] {error_msg}")
                    self.log_message(f"错误: {error_msg}")
                    return False, f"音频提取失败: {str(e)}", None, self.logs
                
                extract_time = time.time() - start_time
                extract_msg = f"音频提取完成，耗时: {extract_time:.2f} 秒"
                print(f"[处理结果] {extract_msg}")
                self.log_message(extract_msg)
                
                # 更新进度
                self.update_progress(30)
                
                # 2. 语音识别
                print("[处理阶段] 2. 语音识别...")
                self.log_message("\n2. 语音识别...")
                start_time = time.time()
                
                try:
                    recognized_text = recognize_speech(
                        audio_path, 
                        params['model'], 
                        device_choice=params['device'],
                        progress_callback=lambda progress: progress_callback(30 + progress * 0.25),
                        beam_size=params['beam_size'],
                        vad_filter=params['vad_filter'],
                        word_timestamps=params['word_timestamps'],
                        condition_on_previous_text=params['condition_on_previous_text']
                    )
                except Exception as e:
                    error_msg = f"语音识别失败 - {str(e)}"
                    print(f"[错误信息] {error_msg}")
                    self.log_message(f"错误: {error_msg}")
                    return False, f"语音识别失败: {str(e)}", None, self.logs
                
                recognize_time = time.time() - start_time
                recognize_msg = f"语音识别完成，耗时: {recognize_time:.2f} 秒"
                language_msg = f"检测到的语言: {recognized_text['language']}"
                print(f"[处理结果] {recognize_msg}")
                print(f"[处理结果] {language_msg}")
                self.log_message(recognize_msg)
                self.log_message(language_msg)
                
                # 更新进度
                self.update_progress(55)
                
                # 3. 翻译日文（如果检测到的不是中文）
                translated_text = recognized_text
                translate_time = 0
                if recognized_text['language'] != 'zh':
                    print("[处理阶段] 3. 翻译日文...")
                    self.log_message("\n3. 翻译日文...")
                    start_time = time.time()
                    
                    try:
                        translated_text = translate_text(
                            recognized_text, 
                            translator_model, 
                            device_choice=params['device'],
                            progress_callback=lambda progress: progress_callback(55 + progress * 0.25),
                            beam_size=params['translation_beam_size'],
                            max_length=params['translation_max_length'],
                            early_stopping=params['translation_early_stopping']
                        )
                    except Exception as e:
                        error_msg = f"翻译失败 - {str(e)}"
                        print(f"[错误信息] {error_msg}")
                        self.log_message(f"错误: {error_msg}")
                        return False, f"翻译失败: {str(e)}", None, self.logs
                    
                    translate_time = time.time() - start_time
                    translate_msg = f"翻译完成，耗时: {translate_time:.2f} 秒"
                    print(f"[处理结果] {translate_msg}")
                    self.log_message(translate_msg)
                else:
                    skip_msg = "跳过翻译步骤，因为检测到的语言是中文"
                    print(f"[处理结果] {skip_msg}")
                    self.log_message(skip_msg)
                    # 更新进度
                    self.update_progress(80)
                
                # 更新进度
                self.update_progress(80)
                
                # 4. 生成字幕
                print("[处理阶段] 4. 生成字幕...")
                self.log_message("\n4. 生成字幕...")
                start_time = time.time()
                
                try:
                    generate_subtitle(
                        translated_text, 
                        output_path, 
                        progress_callback=lambda progress: progress_callback(80 + progress * 0.2)
                    )
                except Exception as e:
                    error_msg = f"字幕生成失败 - {str(e)}"
                    print(f"[错误信息] {error_msg}")
                    self.log_message(f"错误: {error_msg}")
                    return False, f"字幕生成失败: {str(e)}", None, self.logs
                
                generate_time = time.time() - start_time
                generate_msg = f"字幕生成完成，耗时: {generate_time:.2f} 秒"
                print(f"[处理结果] {generate_msg}")
                self.log_message(generate_msg)
                
                # 显示总耗时
                total_time = extract_time + recognize_time + translate_time + generate_time
                complete_msg = f"字幕生成完成：{output_path}"
                total_time_msg = f"总耗时: {total_time:.2f} 秒"
                print(f"[最终结果] {complete_msg}")
                print(f"[最终结果] {total_time_msg}")
                self.log_message(complete_msg)
                self.log_message(total_time_msg)
                
                # 更新进度到100%
                self.update_progress(100)
                
                # 清理临时文件
                print("[资源管理] 清理临时文件...")
                if os.path.exists(TEMP_DIR):
                    try:
                        for file in os.listdir(TEMP_DIR):
                            file_path = os.path.join(TEMP_DIR, file)
                            try:
                                os.remove(file_path)
                                print(f"[资源管理] 已删除临时文件: {file}")
                                self.log_message(f"[资源管理] 已删除临时文件: {file}")
                            except Exception as e:
                                # 如果文件正在使用，跳过删除
                                print(f"[错误信息] 清理临时文件时出错: {str(e)}")
                                self.log_message(f"[错误信息] 清理临时文件时出错: {str(e)}")
                                continue
                    except Exception as e:
                        # 忽略清理错误，不影响主流程
                        print(f"[错误信息] 清理临时文件时出错: {str(e)}")
                        self.log_message(f"[错误信息] 清理临时文件时出错: {str(e)}")
                        pass
                else:
                    print("[资源管理] 临时目录不存在，跳过清理")
                
                # 执行垃圾回收
                print("[资源管理] 执行垃圾回收...")
                try:
                    import gc
                    gc.collect()
                    print("[资源管理] 已执行垃圾回收")
                    self.log_message("[资源管理] 已执行垃圾回收")
                except Exception as e:
                    print(f"[错误信息] 执行垃圾回收时出错: {str(e)}")
                    self.log_message(f"[错误信息] 执行垃圾回收时出错: {str(e)}")
                
                # 检查输出文件是否存在
                print("[结果验证] 检查输出文件是否存在...")
                if os.path.exists(output_path):
                    success_msg = f"字幕生成完成: {output_path}"
                    print(f"[最终结果] {success_msg}")
                    print("[最终结果] ✅ 处理成功！")
                    self.log_message(success_msg)
                    self.log_message("✅ 处理成功！")
                    message = "处理成功！"
                    print(f"[返回信息] 成功: {message}")
                    return True, message, output_path, self.logs
                else:
                    error_msg = f"字幕文件不存在: {output_path}"
                    print(f"[错误信息] {error_msg}")
                    print("[最终结果] ❌ 处理失败！")
                    self.log_message(error_msg)
                    self.log_message("❌ 处理失败！")
                    message = "处理失败，字幕文件未生成"
                    print(f"[返回信息] 失败: {message}")
                    return False, message, None, self.logs
                    
            except Exception as e:
                # 处理整体异常
                import traceback
                error_message = f"处理视频时出错：{str(e)}"
                error_detail = f"详细错误：{traceback.format_exc()}"
                print(f"[错误信息] {error_message}")
                print(f"[错误信息] {error_detail}")
                self.log_message(f"[错误信息] {error_message}")
                self.log_message(f"[错误信息] {error_detail}")
                return False, f"处理过程中出现错误: {str(e)}", None, self.logs
        except Exception as e:
            # 处理整体异常，只打印到终端，不显示在UI中
            import traceback
            error_message = f"处理视频时出错：{str(e)}"
            error_detail = f"详细错误：{traceback.format_exc()}"
            print(f"[错误信息] {error_message}")
            print(f"[错误信息] {error_detail}")
            self.log_message(f"[错误信息] {error_message}")
            self.log_message(f"[错误信息] {error_detail}")
            return False, "处理过程中出现错误", None, self.logs
    
    def process_queue(self):
        """处理队列中的所有视频文件
        
        Yields:
            tuple: (队列数据, 结果消息, 日志输出, 进度值, 队列状态文本)
        """
        # 打印所有信息到终端
        print("\n" + "="*80)
        print("[队列信息] 用户执行了'开始处理队列'操作")
        print(f"[队列信息] 当前队列长度: {len(self.video_queue)}")
        if self.video_queue:
            print("[队列信息] 队列内容:")
            for i, item in enumerate(self.video_queue):
                try:
                    filename = item.get('filename', '未知文件名')
                    status = item.get('status', '未知状态')
                    print(f"  {i}. {filename} - {status}")
                except:
                    print(f"  {i}. {item}")
        print("="*80)
        
        try:
            if not self.video_queue:
                error_msg = "队列为空，没有文件可处理"
                print(f"[错误信息] {error_msg}")
                yield [], "", "", 0, ""
                return
            
            if self.queue_processing:
                yield [[item['filename'], item['status']] for item in self.video_queue], "", "", 0, ""
                return
            
            self.queue_processing = True
            self.current_queue_index = 0
            
            try:
                # 动态遍历队列，使用索引跟踪当前处理位置
                i = 0
                while i < len(self.video_queue):
                    # 获取当前队列项
                    queue_item = self.video_queue[i]
                    # 记录当前处理的文件名，用于日志和错误信息
                    current_filename = queue_item.get('filename', '未知文件')
                    
                    # 更新队列状态
                    try:
                        self.video_queue[i]['status'] = '处理中'
                        queue_data = [[item['filename'], item['status']] for item in self.video_queue]
                        # 使用当前队列的实际长度来计算进度
                        current_total = len(self.video_queue)
                        queue_status_text = f"正在处理队列中的第 {i+1}/{current_total} 个文件：{current_filename}"
                        # 打印到终端
                        print(f"[队列状态] {queue_status_text}")
                    except Exception as e:
                        error_message = f"更新队列状态时出错：{str(e)}"
                        print(f"[错误信息] {error_message}")
                        yield [], "", "", 0, ""
                        i += 1
                        continue
                    
                    # 实时更新队列状态，UI消息为空
                    yield queue_data, "", "", 0, ""
                    
                    # 获取参数
                    try:
                        params = queue_item['params']
                        video_file = queue_item['file_path']
                    except Exception as e:
                        error_message = f"获取队列项参数时出错：{str(e)}"
                        print(f"[错误信息] {error_message}")
                        self.video_queue[i]['status'] = '失败'
                        queue_data = [[item['filename'], item['status']] for item in self.video_queue]
                        yield queue_data, "", "", 0, ""
                        i += 1
                        continue
                    
                    # 清空日志和进度状态
                    try:
                        self.logs = []
                        self.progress = 0
                    except:
                        pass
                    
                    # 调用通用处理函数
                    print("[队列处理] 开始处理视频文件...")
                    success, message, output_path, logs = self.process_video(video_file, params)
                    
                    # 打印消息到终端
                    print(f"[队列处理] 处理结果: {message}")
                    print(f"[队列处理] 日志数量: {len(logs)}")
                    print(f"[队列处理] 成功状态: {success}")
                    
                    # 更新队列状态
                    if success:
                        self.video_queue[i]['status'] = '已完成'
                    else:
                        self.video_queue[i]['status'] = '失败'
                    
                    # 更新队列数据
                    try:
                        queue_data = []
                        for item in self.video_queue:
                            try:
                                # 检查item是否有必要的键
                                if isinstance(item, dict) and 'filename' in item and 'status' in item:
                                    filename = item['filename'] if item['filename'] else '未知文件名'
                                    status = item['status'] if item['status'] else '未知状态'
                                    queue_data.append([filename, status])
                                else:
                                    print(f"[错误信息] 队列项格式错误: {item}")
                                    queue_data.append(['格式错误', '错误'])
                            except Exception as item_error:
                                print(f"[错误信息] 处理队列项时出错: {item_error}")
                                queue_data.append(['处理错误', '错误'])
                        current_total = len(self.video_queue)
                        queue_status_text = f"已完成队列中的第 {i+1}/{current_total} 个文件：{current_filename}"
                        # 打印到终端
                        print(f"[队列状态] {queue_status_text}")
                    except Exception as e:
                        error_msg = f"更新队列数据时出错：{str(e)}"
                        print(f"[错误信息] {error_msg}")
                        queue_data = []
                        queue_status_text = error_msg
                    
                    # 实时更新UI，消息为空
                    yield queue_data, "", '\n'.join(logs), 100, ""
                    
                    # 清理资源，准备处理下一个视频
                    try:
                        # 确保所有进程都已终止
                        print("[资源管理] 清理资源，准备处理下一个视频...")
                        # 执行垃圾回收
                        import gc
                        gc.collect()
                        gc.collect()  # 再次执行，确保完全释放
                        print("[资源管理] 已执行垃圾回收")
                    except Exception as e:
                        print(f"[错误信息] 清理资源时出错：{str(e)}")
                    
                    # 减少延迟时间，同时确保资源释放
                    time.sleep(0.5)  # 减少到0.5秒，平衡资源释放和处理速度
                    
                    # 处理完当前视频，增加索引
                    i += 1
                
                # 处理完成
                try:
                    actual_processed = len(self.video_queue)
                    queue_status_text = f"队列处理完成，共处理了 {actual_processed} 个文件"
                    print(f"[队列状态] {queue_status_text}")
                    yield [[item['filename'], item['status']] for item in self.video_queue], "", '\n'.join(self.logs), 100, ""
                except Exception as e:
                    error_msg = f"队列处理完成时出错：{str(e)}"
                    print(f"[错误信息] {error_msg}")
                    yield [], "", "", 0, ""
            finally:
                self.queue_processing = False
                self.current_queue_index = -1
                # 清空日志和进度状态
                try:
                    self.logs = []
                    self.progress = 0
                except:
                    pass
        except Exception as e:
            # 处理整体异常，只打印到终端，不显示在UI中
            import traceback
            error_message = f"处理队列时出错：{str(e)}"
            error_detail = f"详细错误：{traceback.format_exc()}"
            print(f"[错误信息] {error_message}")
            print(f"[错误信息] {error_detail}")
            
            try:
                self.log_message(error_message)
                self.log_message(error_detail)
            except:
                pass
            finally:
                self.queue_processing = False
                self.current_queue_index = -1
                # 清空日志和进度状态
                try:
                    self.logs = []
                    self.progress = 0
                except:
                    pass
            # UI消息为空
            yield [], "", "", 0, ""

# 创建队列管理器实例
queue_manager = QueueManager()
