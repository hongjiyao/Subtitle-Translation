# -*- coding: utf-8 -*-
"""
语言占比检测模块
用于检测文本中各语言的字符占比，判断翻译是否成功
"""

import re
from typing import Dict, Tuple


# 语言字符范围定义
LANGUAGE_RANGES = {
    'chinese': (0x4E00, 0x9FFF),      # 中文：CJK统一表意文字
    'japanese_hiragana': (0x3040, 0x309F),  # 平假名
    'japanese_katakana': (0x30A0, 0x30FF),  # 片假名
    'japanese': (0x3040, 0x30FF),      # 日语（平假名+片假名）
    'korean': (0xAC00, 0xD7AF),       # 韩文
    'latin': (0x0041, 0x007A),        # 拉丁字母（ASCII）
    'latin_extended': (0x00C0, 0x024F),  # 扩展拉丁字母
    'cyrillic': (0x0400, 0x04FF),     # 西里尔字母
    'arabic': (0x0600, 0x06FF),       # 阿拉伯字母
    'thai': (0x0E00, 0x0E7F),         # 泰文
    'devanagari': (0x0900, 0x097F),   # 天城文
}


def is_other_chinese_char(char: str) -> bool:
    """检测字符是否为中文标点符号

    Args:
        char: 单个字符

    Returns:
        bool: 是否为中文标点符号
    """
    if not char:
        return False
    code = ord(char)
    # 中文标点符号范围（CJK标点符号）
    # U+3000-U+303F: CJK Symbols and Punctuation
    # U+FF00-U+FFEF: Halfwidth and Fullwidth Forms
    # U+2000-U+206F: General Punctuation (包括省略号 U+2026)
    return (0x3000 <= code <= 0x303F) or (0xFF00 <= code <= 0xFFEF) or (0x2000 <= code <= 0x206F)


def is_chinese_char(char: str) -> bool:
    """检测字符是否为中文

    Args:
        char: 单个字符

    Returns:
        bool: 是否为中文字符
    """
    if not char:
        return False
    code = ord(char)
    return 0x4E00 <= code <= 0x9FFF


def is_japanese_char(char: str) -> bool:
    """检测字符是否为日文字符（平假名或片假名）

    Args:
        char: 单个字符

    Returns:
        bool: 是否为日文字符
    """
    if not char:
        return False
    code = ord(char)
    return (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF)


def is_latin_char(char: str) -> bool:
    """检测字符是否为拉丁字母

    Args:
        char: 单个字符

    Returns:
        bool: 是否为拉丁字母
    """
    if not char:
        return False
    code = ord(char)
    return (0x0041 <= code <= 0x007A) or (0x00C0 <= code <= 0x024F)


def is_korean_char(char: str) -> bool:
    """检测字符是否为韩文字符

    Args:
        char: 单个字符

    Returns:
        bool: 是否为韩文字符
    """
    if not char:
        return False
    code = ord(char)
    return 0xAC00 <= code <= 0xD7AF


def detect_language_chars(text: str) -> Dict[str, int]:
    """检测文本中各语言的字符数量

    Args:
        text: 待检测文本

    Returns:
        Dict[str, int]: 各语言的字符数量，键包括：
        - 'chinese': 中文字符数量
        - 'japanese': 日文字符数量
        - 'korean': 韩文字符数量
        - 'latin': 拉丁字母数量
        - 'other': 其他字符数量
    """
    if not text:
        return {
            'chinese': 0,
            'japanese': 0,
            'korean': 0,
            'latin': 0,
            'other': 0
        }

    counts = {
        'chinese': 0,
        'japanese': 0,
        'korean': 0,
        'latin': 0,
        'other': 0
    }

    for char in text:
        if is_chinese_char(char):
            counts['chinese'] += 1
        elif is_japanese_char(char):
            counts['japanese'] += 1
        elif is_korean_char(char):
            counts['korean'] += 1
        elif is_latin_char(char):
            counts['latin'] += 1
        elif is_other_chinese_char(char):
            # 中文标点符号也算作中文字符
            counts['chinese'] += 1
        else:
            counts['other'] += 1

    return counts


def calculate_language_ratio(text: str, language: str = 'chinese') -> float:
    """计算指定语言的字符占比

    Args:
        text: 待检测文本
        language: 目标语言 ('chinese', 'japanese', 'korean', 'latin')

    Returns:
        float: 指定语言的字符占比（0.0 - 1.0）
    """
    if not text:
        return 0.0

    counts = detect_language_chars(text)
    total = len(text)

    # 计算目标语言的占比
    if language == 'chinese':
        target_count = counts['chinese']
    elif language == 'japanese':
        target_count = counts['japanese']
    elif language == 'korean':
        target_count = counts['korean']
    elif language == 'latin':
        target_count = counts['latin']
    else:
        # 默认计算中文字符占比
        target_count = counts['chinese']

    return target_count / total if total > 0 else 0.0


def check_translation_success(
    original_text: str,
    translated_text: str,
    source_lang: str = 'ja',
    target_lang: str = 'zh',
    threshold: float = 0.5
) -> Tuple[bool, float, Dict[str, int]]:
    """检查翻译是否成功

    基于目标语言的字符占比来判断翻译是否成功。
    例如：日译中时，如果译文中的中文字符占比超过阈值，则认为翻译成功。

    Args:
        original_text: 原文
        translated_text: 译文
        source_lang: 源语言代码 ('ja'=日语, 'en'=英语等)
        target_lang: 目标语言代码 ('zh'=中文, 'en'=英语等)
        threshold: 目标语言字符占比阈值（默认0.5，即50%）

    Returns:
        Tuple[bool, float, Dict[str, int]]: 
        - 是否翻译成功
        - 目标语言字符占比
        - 各语言字符数量统计
    """
    if not translated_text:
        return False, 0.0, {}

    # 统计译文的语言字符
    lang_counts = detect_language_chars(translated_text)

    # 计算目标语言占比
    if target_lang == 'zh':
        # 目标是中文，检测中文字符占比
        target_ratio = lang_counts['chinese'] / len(translated_text)
    elif target_lang == 'ja':
        # 目标是日语，检测日文字符占比
        target_ratio = lang_counts['japanese'] / len(translated_text)
    elif target_lang == 'ko':
        # 目标是韩语，检测韩文字符占比
        target_ratio = lang_counts['korean'] / len(translated_text)
    elif target_lang == 'en':
        # 目标是英语，检测拉丁字母占比
        target_ratio = lang_counts['latin'] / len(translated_text)
    else:
        # 默认检测中文字符占比
        target_ratio = lang_counts['chinese'] / len(translated_text)

    # 判断翻译是否成功
    success = target_ratio >= threshold

    return success, target_ratio, lang_counts


def get_translation_quality_info(
    original_text: str,
    translated_text: str,
    source_lang: str = 'ja',
    target_lang: str = 'zh'
) -> Dict[str, any]:
    """获取翻译质量详细信息

    Args:
        original_text: 原文
        translated_text: 译文
        source_lang: 源语言代码
        target_lang: 目标语言代码

    Returns:
        Dict: 包含翻译质量各项指标
    """
    if not translated_text:
        return {
            'success': False,
            'target_ratio': 0.0,
            'lang_counts': {},
            'total_chars': 0,
            'message': '翻译结果为空'
        }

    lang_counts = detect_language_chars(translated_text)
    total_chars = len(translated_text)

    # 计算目标语言占比
    if target_lang == 'zh':
        target_ratio = lang_counts['chinese'] / total_chars if total_chars > 0 else 0.0
        target_lang_name = '中文'
    elif target_lang == 'ja':
        target_ratio = lang_counts['japanese'] / total_chars if total_chars > 0 else 0.0
        target_lang_name = '日文'
    elif target_lang == 'ko':
        target_ratio = lang_counts['korean'] / total_chars if total_chars > 0 else 0.0
        target_lang_name = '韩文'
    elif target_lang == 'en':
        target_ratio = lang_counts['latin'] / total_chars if total_chars > 0 else 0.0
        target_lang_name = '英文'
    else:
        target_ratio = lang_counts['chinese'] / total_chars if total_chars > 0 else 0.0
        target_lang_name = '目标语言'

    success = target_ratio >= 0.5

    return {
        'success': success,
        'target_ratio': target_ratio,
        'target_lang_name': target_lang_name,
        'lang_counts': lang_counts,
        'total_chars': total_chars,
        'message': f'{target_lang_name}占比: {target_ratio:.2%}' + (' ✓' if success else ' ✗')
    }


# 测试代码
if __name__ == '__main__':
    # 测试用例
    test_cases = [
        # 日译中成功案例
        ("くそ!", "该死", 'ja', 'zh'),
        # 日译中失败案例（包含日文）
        ("くそ!", "くそ!", 'ja', 'zh'),
        # 混合语言
        ("うわあぁああああ!", "哇啊啊啊啊啊!", 'ja', 'zh'),
        # 短文本
        ("がっ", "嘎", 'ja', 'zh'),
    ]

    print("=" * 60)
    print("语言占比检测测试")
    print("=" * 60)

    for original, translated, src, tgt in test_cases:
        result = get_translation_quality_info(original, translated, src, tgt)
        print(f"\n原文: {original}")
        print(f"译文: {translated}")
        print(f"源语言: {src} -> 目标语言: {tgt}")
        print(f"字符统计: {result['lang_counts']}")
        print(f"总字符数: {result['total_chars']}")
        print(f"结果: {result['message']}")
        print("-" * 60)