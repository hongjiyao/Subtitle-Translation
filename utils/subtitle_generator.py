# -*- coding: utf-8 -*-
import os

def generate_subtitle(translated_result, output_path, progress_callback=None):
    """生成字幕文件"""
    base_name = os.path.basename(output_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_dir = "outputs"
    output_path = os.path.join(output_dir, f"{name_without_ext}.srt")

    print(f"正在生成原文字幕: {output_path}")

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(translated_result.get('segments', []), 1):
                f.write(f"{i}\n")

                start = seg.get('start', 0)
                end = seg.get('end', 0)
                start_time = format_time(start)
                end_time = format_time(end)
                f.write(f"{start_time} --> {end_time}\n")

                text = seg.get('text', '')
                f.write(f"{text}\n\n")

        print(f"原文字幕生成完成: {output_path}")
        return output_path
    except Exception as e:
        print(f"原文字幕生成失败: {str(e)}")
        raise

def generate_translated_subtitle(translated_result, output_path, progress_callback=None):
    """生成译文字幕文件

    Args:
        translated_result: 包含翻译结果的字典
        output_path: 输出文件路径（不应该包含 _translated 后缀）
        progress_callback: 进度回调
    """
    import os as os_module

    print(f"正在生成译文字幕: {output_path}")

    try:
        output_dir = os_module.path.dirname(output_path) or "outputs"
        base_name = os_module.path.basename(output_path)
        name_without_ext = os_module.path.splitext(base_name)[0]

        final_output_path = os_module.path.join(output_dir, f"{name_without_ext}.srt")

        os_module.makedirs(output_dir, exist_ok=True)

        with open(final_output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(translated_result.get('segments', []), 1):
                f.write(f"{i}\n")

                start = seg.get('start', 0)
                end = seg.get('end', 0)
                start_time = format_time(start)
                end_time = format_time(end)
                f.write(f"{start_time} --> {end_time}\n")

                if 'translated' in seg:
                    text = seg['translated']
                else:
                    text = seg.get('text', '')
                f.write(f"{text}\n\n")

        print(f"译文字幕生成完成: {final_output_path}")
        return final_output_path
    except Exception as e:
        print(f"译文字幕生成失败: {str(e)}")
        raise

def generate_bilingual_subtitle(translated_result, output_path, progress_callback=None):
    """生成双语字幕文件

    Args:
        translated_result: 包含翻译结果的字典
        output_path: 输出文件路径
        progress_callback: 进度回调
    """
    import os as os_module

    print(f"正在生成双语字幕: {output_path}")

    try:
        output_dir = os_module.path.dirname(output_path) or "outputs"
        base_name = os_module.path.basename(output_path)
        name_without_ext = os_module.path.splitext(base_name)[0]

        final_output_path = os_module.path.join(output_dir, f"{name_without_ext}.srt")

        os_module.makedirs(output_dir, exist_ok=True)

        with open(final_output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(translated_result.get('segments', []), 1):
                f.write(f"{i}\n")

                start = seg.get('start', 0)
                end = seg.get('end', 0)
                start_time = format_time(start)
                end_time = format_time(end)
                f.write(f"{start_time} --> {end_time}\n")

                original_text = seg.get('original_text', seg.get('text', ''))
                translated_text = seg.get('translated', '')
                f.write(f"{original_text}\n{translated_text}\n\n")

        print(f"双语字幕生成完成: {final_output_path}")
        return final_output_path
    except Exception as e:
        print(f"双语字幕生成失败: {str(e)}")
        raise

def format_time(seconds):
    """格式化时间为SRT格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
