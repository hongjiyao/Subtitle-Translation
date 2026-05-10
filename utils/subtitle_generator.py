# -*- coding: utf-8 -*-
import os


def _write_srt(segments, output_path, text_extractor, label):
    print(f"正在生成{label}字幕: {output_path}")
    try:
        output_dir = os.path.dirname(output_path) or "outputs"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        final_path = os.path.join(output_dir, f"{base_name}.srt")

        with open(final_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{format_time(seg.get('start', 0))} --> {format_time(seg.get('end', 0))}\n")
                f.write(f"{text_extractor(seg)}\n\n")

        print(f"{label}字幕生成完成: {final_path}")
        return final_path
    except Exception as e:
        print(f"{label}字幕生成失败: {str(e)}")
        raise


def generate_subtitle(translated_result, output_path, progress_callback=None):
    return _write_srt(
        translated_result.get('segments', []),
        output_path,
        lambda seg: seg.get('text', ''),
        "原文"
    )


def generate_translated_subtitle(translated_result, output_path, progress_callback=None):
    return _write_srt(
        translated_result.get('segments', []),
        output_path,
        lambda seg: seg['translated'] if 'translated' in seg else seg.get('text', ''),
        "译文"
    )


def generate_bilingual_subtitle(translated_result, output_path, progress_callback=None):
    return _write_srt(
        translated_result.get('segments', []),
        output_path,
        lambda seg: f"{seg.get('original_text', seg.get('text', ''))}\n{seg.get('translated', '')}",
        "双语"
    )


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
