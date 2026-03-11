import os
import datetime

# 确保在导入任何库之前设置HF-Mirror作为下载源
# 这些环境变量需要在导入transformers之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.environ["HUGGINGFACE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# 获取带时间戳的打印函数
def timestamp_print(message):
    """带时间戳的打印函数"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import time
from config import MODEL_CACHE_DIR

def clear_translator_cache():
    """清空翻译模型缓存以释放内存"""
    # 强制垃圾回收
    import gc
    gc.collect()
    # 清空CUDA缓存
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        timestamp_print(f"[内存管理] 清空CUDA缓存时出错: {str(e)}")
    # 再次垃圾回收，确保完全释放
    gc.collect()
    timestamp_print("[内存管理] 已执行垃圾回收")

def translate_text(recognized_result, model_path, device_choice="auto", progress_callback=None, beam_size=1, max_length=256, target_language="zh", batch_size=8):
    """翻译识别结果"""
    # 确保torch可用
    try:
        import torch
    except ImportError:
        timestamp_print("[错误信息] 无法导入torch")
        raise
    
    # 确定目标语言
    if not target_language:
        target_language = "zh"
    # 确定设备
    if device_choice == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_choice)
    
    # 使用用户选择的翻译模型
    model_name = model_path
    
    # 检查是否是本地模型路径或需要使用本地路径
    local_model_path = None
    
    # 优先检查本地models目录
    if model_name == "facebook/m2m100_418M":
        # 尝试多种可能的本地路径
        local_paths = [
            os.path.join(MODEL_CACHE_DIR, "m2m100_418M"),
            os.path.join(MODEL_CACHE_DIR, "models--facebook--m2m100_418M", "snapshots", "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"),
            os.path.join(MODEL_CACHE_DIR, "facebook", "m2m100_418M"),
            os.path.join(MODEL_CACHE_DIR, "facebook--m2m100_418M")
        ]
        
        for path in local_paths:
            if os.path.exists(path):
                local_model_path = path
                break
                
        if local_model_path:
            model_name = local_model_path
            timestamp_print(f"[翻译] 使用本地模型: {local_model_path}")
            
    elif model_name == "facebook/m2m100_1.2B":
        # 尝试多种可能的本地路径
        local_paths = [
            os.path.join(MODEL_CACHE_DIR, "m2m100_1.2B"),
            os.path.join(MODEL_CACHE_DIR, "models--facebook--m2m100_1.2B", "snapshots", "d12e0448187684a8274a11959f5b394a8c1edf7d"),
            os.path.join(MODEL_CACHE_DIR, "facebook", "m2m100_1.2B"),
            os.path.join(MODEL_CACHE_DIR, "facebook--m2m100_1.2B")
        ]
        
        for path in local_paths:
            if os.path.exists(path):
                local_model_path = path
                break
                
        if local_model_path:
            model_name = local_model_path
            timestamp_print(f"[翻译] 使用本地模型: {local_model_path}")
            
    # 通用本地模型检查
    elif os.path.exists(model_name):
        # 如果用户直接提供了本地路径，使用它
        timestamp_print(f"[翻译] 使用用户提供的本地模型: {model_name}")
    else:
        # 尝试直接在models目录中查找
        direct_model_path = os.path.join(MODEL_CACHE_DIR, model_name)
        if os.path.exists(direct_model_path):
            model_name = direct_model_path
            timestamp_print(f"[翻译] 使用本地模型: {model_name}")
        else:
            # 只使用本地模型，不下载
            error_msg = f"本地模型不存在: {model_path}，请确保模型已在models目录中"
            timestamp_print(f"[错误信息] {error_msg}")
            raise FileNotFoundError(error_msg)
    
    # 每次都重新加载模型
    # 加载分词器和模型
    timestamp_print(f"[模型加载] 正在将翻译模型 {model_name} 加载到内存中...")
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    # 加载模型时指定tie_word_embeddings=False以消除警告
    model = M2M100ForConditionalGeneration.from_pretrained(
        model_name,
        tie_word_embeddings=False
    )
    model.to(device)
    timestamp_print(f"[模型加载] 翻译模型 {model_name} 已成功加载到内存，设备: {device.type}")
    
    # 检测模型加载后的显存使用情况
    if device.type == "cuda":
        try:
            import subprocess
            import re
            # 执行nvidia-smi命令
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            # 解析输出
            output = result.stdout.strip()
            if output:
                total, free = map(int, output.split(','))
                used = total - free
                timestamp_print(f"[内存管理] 模型加载后GPU显存: 总内存={total/1024:.2f}GB, 可用内存={free/1024:.2f}GB, 已使用={used/1024:.2f}GB")
        except Exception as e:
            timestamp_print(f"[错误信息] 获取模型加载后GPU内存时出错: {str(e)}")
    
    # 处理每个片段
    translated_segments = []
    total_segments = len(recognized_result["segments"])
    translated_count = 0
    
    timestamp_print(f"[翻译] 开始处理 {total_segments} 个片段...")
    
    # 根据模型大小和设备自动调整批量大小
    if device.type == "cuda":
        # 使用nvidia-smi检查GPU内存
        try:
            import subprocess
            import re
            
            # 执行nvidia-smi命令
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # 解析输出
            output = result.stdout.strip()
            if output:
                total, free = map(int, output.split(','))
                total_memory_gb = total / 1024
                free_memory_gb = free / 1024
                
                timestamp_print(f"[内存管理] GPU总内存: {total_memory_gb:.2f} GB")
                timestamp_print(f"[内存管理] 可用内存: {free_memory_gb:.2f} GB")
                
                # 根据可用内存调整批量大小
                if free_memory_gb < 2:
                    # 内存不足，使用最小批量大小
                    adjusted_batch_size = 1
                    timestamp_print(f"[内存管理] 内存不足，调整批量大小为: {adjusted_batch_size}")
                elif free_memory_gb < 4:
                    # 内存较少，使用小批量
                    adjusted_batch_size = 2
                    timestamp_print(f"[内存管理] 内存较少，调整批量大小为: {adjusted_batch_size}")
                elif free_memory_gb < 6:
                    # 内存适中，使用中等批量
                    adjusted_batch_size = 4
                    timestamp_print(f"[内存管理] 内存适中，调整批量大小为: {adjusted_batch_size}")
                else:
                    # 内存充足，使用用户指定的批量大小
                    adjusted_batch_size = batch_size
                    timestamp_print(f"[内存管理] 内存充足，使用批量大小: {adjusted_batch_size}")
            else:
                timestamp_print("[内存管理] 无法获取GPU内存信息，使用默认批量大小")
                adjusted_batch_size = batch_size
        except Exception as e:
            timestamp_print(f"[内存管理] 无法检测GPU内存: {str(e)}，使用默认批量大小")
            adjusted_batch_size = batch_size
    else:
        # CPU模式，使用较小的批量大小
        adjusted_batch_size = min(batch_size, 4)
        timestamp_print(f"[内存管理] CPU模式，使用批量大小: {adjusted_batch_size}")
    
    # 准备批量数据
    batches = []
    for i in range(0, total_segments, adjusted_batch_size):
        batch = recognized_result["segments"][i:i+adjusted_batch_size]
        batches.append(batch)
    
    current_segment = 0
    
    # 处理每个批次
    for batch_idx, batch in enumerate(batches):
        batch_texts = [segment["text"] for segment in batch]
        batch_segments = batch
        
        try:
            # 对于M2M100模型，设置目标语言
            tokenizer.tgt_lang = target_language
            
            # 限制输入文本长度，避免超时
            batch_texts = [text[:max_length] for text in batch_texts]
            
            # 批量编码输入文本
            encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            
            # 设置生成参数，优化速度
            start_time = time.time()
            
            # 批量生成翻译，显式设置 early_stopping=False 避免警告
            generated_tokens = model.generate(
                **encoded,
                max_length=max_length,
                num_beams=beam_size,
                forced_bos_token_id=tokenizer.get_lang_id(target_language),
                no_repeat_ngram_size=2,  # 避免重复
                early_stopping=False  # 显式设置避免警告
            )
            
            # 解码翻译结果
            translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # 更新批量中的每个片段
            for j, (segment, translation) in enumerate(zip(batch_segments, translations)):
                segment["translated"] = translation
                translated_count += 1
                current_segment += 1
            
            process_time = time.time() - start_time
            timestamp_print(f"[翻译] 已翻译批次 {batch_idx+1}/{len(batches)}，处理了 {current_segment}/{total_segments} 个片段 (耗时: {process_time:.2f}秒)")
        except RuntimeError as e:
            if "out of memory" in str(e):
                # 内存不足，尝试减小批量大小
                timestamp_print(f"[翻译] CUDA内存不足: {str(e)}，尝试减小批量大小")
                # 回退到逐段处理
                for j, segment in enumerate(batch_segments):
                    try:
                        # 清理内存
                        import gc
                        gc.collect()
                        
                        text = segment["text"][:max_length]
                        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
                        # 逐段生成翻译，显式设置 early_stopping=False 避免警告
                        generated_tokens = model.generate(
                            **encoded,
                            max_length=max_length,
                            num_beams=beam_size,
                            forced_bos_token_id=tokenizer.get_lang_id(target_language),
                            no_repeat_ngram_size=2,
                            early_stopping=False  # 显式设置避免警告
                        )
                        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                        segment["translated"] = translation
                        translated_count += 1
                    except Exception as e:
                        timestamp_print(f"[翻译] 翻译片段失败: {str(e)}")
                        segment["translated"] = segment["text"]
                    current_segment += 1
            else:
                # 其他运行时错误，回退到逐段处理
                timestamp_print(f"[翻译] 批量处理失败: {str(e)}，回退到逐段处理")
                for j, segment in enumerate(batch_segments):
                    try:
                        text = segment["text"][:max_length]
                        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
                        # 逐段生成翻译，显式设置 early_stopping=False 避免警告
                        generated_tokens = model.generate(
                            **encoded,
                            max_length=max_length,
                            num_beams=beam_size,
                            forced_bos_token_id=tokenizer.get_lang_id(target_language),
                            no_repeat_ngram_size=2,
                            early_stopping=False  # 显式设置避免警告
                        )
                        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                        segment["translated"] = translation
                        translated_count += 1
                    except Exception as e:
                        timestamp_print(f"[翻译] 翻译片段失败: {str(e)}")
                        segment["translated"] = segment["text"]
                    current_segment += 1
        except Exception as e:
            # 其他异常，回退到逐段处理
            timestamp_print(f"[翻译] 批量处理失败: {str(e)}，回退到逐段处理")
            for j, segment in enumerate(batch_segments):
                try:
                    text = segment["text"][:max_length]
                    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
                    # 逐段生成翻译，显式设置 early_stopping=False 避免警告
                    generated_tokens = model.generate(
                        **encoded,
                        max_length=max_length,
                        num_beams=beam_size,
                        forced_bos_token_id=tokenizer.get_lang_id(target_language),
                        no_repeat_ngram_size=2,
                        early_stopping=False  # 显式设置避免警告
                    )
                    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    segment["translated"] = translation
                    translated_count += 1
                except Exception as e:
                    timestamp_print(f"[翻译] 翻译片段失败: {str(e)}")
                    segment["translated"] = segment["text"]
                current_segment += 1
        
        # 清理内存
        import gc
        gc.collect()
        
        # 将批次处理的片段添加到结果中
        translated_segments.extend(batch_segments)
        
        # 更新进度条
        if progress_callback and total_segments > 0:
            progress = int((current_segment / total_segments) * 100)
            progress_callback(progress)
        
        # 移除暂停，提高翻译速度
    
    timestamp_print(f"[翻译] 翻译完成，共翻译 {translated_count}/{total_segments} 个片段")
    
    # 直接使用处理后的片段，因为已经按照原始顺序处理
    recognized_result["segments"] = translated_segments
    
    # 清理内存
    try:
        # 删除模型和分词器对象
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        # 强制垃圾回收
        import gc
        gc.collect()
        # 清空CUDA缓存
        if device.type == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception as e:
                timestamp_print(f"[内存管理] 清空CUDA缓存时出错: {str(e)}")
        # 再次垃圾回收
        gc.collect()
    except Exception as e:
        timestamp_print(f"[内存管理] 清理内存时出错: {str(e)}")
    
    return recognized_result
