import os

# 确保在导入任何库之前设置HF-Mirror作为下载源
# 这些环境变量需要在导入transformers之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.environ["HUGGINGFACE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import time
from config import MODEL_CACHE_DIR

# 翻译模型缓存
TRANSLATOR_MODEL_CACHE = {}

def translate_text(recognized_result, model_path, device_choice="auto", progress_callback=None, beam_size=1, max_length=256, early_stopping=True):
    """翻译识别结果中的日文部分"""
    # 确定设备
    if device_choice == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_choice)
    
    # 构建模型缓存键
    cache_key = f"translator-m2m100-{device.type}"
    
    # 检查模型是否已在缓存中
    if cache_key not in TRANSLATOR_MODEL_CACHE:
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
                print(f"[翻译] 使用本地模型: {local_model_path}")
                
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
                print(f"[翻译] 使用本地模型: {local_model_path}")
                
        # 通用本地模型检查
        elif os.path.exists(model_name):
            # 如果用户直接提供了本地路径，使用它
            print(f"[翻译] 使用用户提供的本地模型: {model_name}")
        else:
            # 尝试直接在models目录中查找
            direct_model_path = os.path.join(MODEL_CACHE_DIR, model_name)
            if os.path.exists(direct_model_path):
                model_name = direct_model_path
                print(f"[翻译] 使用本地模型: {model_name}")
            else:
                # 只使用本地模型，不下载
                error_msg = f"本地模型不存在: {model_path}，请确保模型已在models目录中"
                print(f"[错误信息] {error_msg}")
                raise FileNotFoundError(error_msg)
        
        # 加载分词器和模型
        print(f"[翻译] 加载翻译模型: {model_name}")
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        # 加载模型时指定tie_word_embeddings=False以消除警告
        model = M2M100ForConditionalGeneration.from_pretrained(
            model_name,
            tie_word_embeddings=False
        )
        model.to(device)
        print(f"[翻译] 模型加载完成，设备: {device.type}")
        
        # 将模型添加到缓存
        TRANSLATOR_MODEL_CACHE[cache_key] = (tokenizer, model)
    
    tokenizer, model = TRANSLATOR_MODEL_CACHE[cache_key]
    
    # 处理每个片段
    translated_segments = []
    total_segments = len(recognized_result["segments"])
    translated_count = 0
    
    print(f"[翻译] 开始处理 {total_segments} 个片段...")
    
    for i, segment in enumerate(recognized_result["segments"]):
        text = segment["text"]
        
        # 检测语言（简单实现）
        is_japanese = any(0x3040 <= ord(c) <= 0x30FF for c in text)
        
        if is_japanese:
            try:
                # 对于M2M100模型，设置目标语言为中文
                tokenizer.tgt_lang = "zh"
                
                # 限制输入文本长度，避免超时
                text = text[:max_length]  # 直接截断过长文本
                
                # 编码输入文本
                encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
                
                # 设置生成参数，优化速度
                start_time = time.time()
                
                # 生成翻译
                generated_tokens = model.generate(
                    **encoded,
                    max_length=max_length,
                    num_beams=beam_size,
                    early_stopping=early_stopping,
                    forced_bos_token_id=tokenizer.get_lang_id("zh"),
                    length_penalty=0.8,  # 长度惩罚，加速生成
                    no_repeat_ngram_size=2  # 避免重复
                )
                
                # 解码翻译结果
                chinese_translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                
                segment["translated"] = chinese_translation
                translated_count += 1
                
                process_time = time.time() - start_time
                print(f"[翻译] 已翻译第 {i+1}/{total_segments} 个片段 (耗时: {process_time:.2f}秒)")
            except Exception as e:
                # 回退到使用原始文本
                print(f"[翻译] 翻译第 {i+1} 个片段失败: {str(e)}")
                segment["translated"] = text
        else:
            # 非日语不需要翻译
            segment["translated"] = text
        
        translated_segments.append(segment)
        
        # 更新进度条
        if progress_callback and total_segments > 0:
            progress = int(((i + 1) / total_segments) * 100)
            progress_callback(progress)
        
        # 短暂暂停，避免资源占用过高
        time.sleep(0.1)
    
    print(f"[翻译] 翻译完成，共翻译 {translated_count}/{total_segments} 个片段")
    
    # 按原始顺序排序
    translated_segments.sort(key=lambda x: recognized_result["segments"].index(x) if x in recognized_result["segments"] else 0)
    
    recognized_result["segments"] = translated_segments
    return recognized_result
