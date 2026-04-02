# -*- coding: utf-8 -*-
def generate_subtitle(translated_result, output_path, progress_callback=None):
    """生成字幕文件"""
    print(f"正在生成字幕: {output_path}")
    
    try:
        total_segments = len(translated_result["segments"])
        
        # 获取语言信息
        language = translated_result.get("language", "en")
        
        # 定义不需要空格的语言列表
        no_space_languages = ["zh", "zh-Hant", "ja", "ko", "th", "vi", "my", "km", "bo", "ug"]
        
        # 判断当前语言是否需要空格
        needs_spaces = language not in no_space_languages
        
        print(f"检测到语言: {language}, {'需要空格' if needs_spaces else '不需要空格'}")
        
        # 使用缓冲区减少磁盘操作
        subtitle_content = []
        
        for i, segment in enumerate(translated_result["segments"]):
            # 构建字幕内容
            subtitle_content.append(f"{i+1}")
            
            # 写入时间轴
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            subtitle_content.append(f"{start_time} --> {end_time}")
            
            # 只写入中文译文，不显示原文
            if "translated" in segment:
                text = segment['translated']
            else:
                # 如果没有译文，则使用原文
                text = segment['text']
            
            # 根据语言类型处理空格
            if needs_spaces:
                # 需要空格的语言：移除首尾空格，保留单词间的单个空格
                text = text.strip()
                text = ' '.join(text.split())
            else:
                # 不需要空格的语言：移除所有空格
                text = text.replace(' ', '')
            
            subtitle_content.append(f"{text}")
            
            # 空行分隔
            subtitle_content.append("")
            
            # 更新进度条（如果有）
            if progress_callback and total_segments > 0:
                progress = int(((i + 1) / total_segments) * 100)
                progress_callback(progress)
        
        # 一次性写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(subtitle_content))
        
        print(f"字幕生成完成: {output_path}")
        return output_path
    except Exception as e:
        print(f"字幕生成失败: {str(e)}")
        raise

def format_time(seconds):
    """格式化时间为SRT格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
