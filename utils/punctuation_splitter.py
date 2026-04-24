# -*- coding: utf-8 -*-
"""
标点断句模块
基于标点符号进行断句，支持中英文标点
"""

from typing import List, Dict
import re


# 句末标点（遇到这些标点必定断句）
UNIVERSAL_SENTENCE_ENDS = {"。", "？", "！", "…", ".", "?", "!"}

# 分隔标点（遇到这些标点也断句）
UNIVERSAL_SPLIT_PUNCTUATION = {"、", "，", "；", ":", ",", ";"}

# 所有断句标点
ALL_SPLIT_PUNCTUATION = UNIVERSAL_SENTENCE_ENDS.union(UNIVERSAL_SPLIT_PUNCTUATION)


def split_by_punctuation(text: str, return_normalized: bool = False) -> List[dict]:
    """基于标点符号断句，返回每个句子的文本和位置信息

    遇到句末标点（。？！… .?!）时断句
    遇到分隔标点（、，；: ,;）时断句
    两个连续的省略号（……）视为一个标点
    单独的省略号不独立成句，会与前后内容合并

    注意：此函数保留原始文本的位置信息，不断句文本不进行规范化

    Args:
        text: 原始文本
        return_normalized: 是否返回规范化后的文本（默认False，不规范化）

    Returns:
        List[dict]: 每个元素包含 'text'、'start' 和 'end' 位置信息（原始文本中的位置）
    """
    if not text:
        return []

    # 不再预处理规范化，直接在原始文本上断句
    results = []
    current = []
    start_pos = 0
    i = 0
    n = len(text)

    while i < n:
        char = text[i]

        if char == '…':
            # 优先检查两个连续的省略号
            if i + 1 < n and text[i + 1] == '…':
                # 两个或更多省略号作为一个标点
                current.append('…' * 2)
                i += 2
                # 跳过剩余的连续省略号
                while i < n and text[i] == '…':
                    i += 1
                if current:
                    sentence = "".join(current).strip()
                    if sentence:
                        results.append({
                            'text': sentence,
                            'start': start_pos,
                            'end': i
                        })
                    current = []
                    start_pos = i
            else:
                # 单个省略号也视为句末标点
                current.append(char)
                i += 1
                if current:
                    sentence = "".join(current).strip()
                    if sentence:
                        results.append({
                            'text': sentence,
                            'start': start_pos,
                            'end': i
                        })
                    current = []
                    start_pos = i

        elif char in UNIVERSAL_SENTENCE_ENDS:
            # 句末标点：当前字符属于当前句子
            current.append(char)
            i += 1
            if current:
                sentence = "".join(current).strip()
                if sentence:
                    results.append({
                        'text': sentence,
                        'start': start_pos,
                        'end': i
                    })
                current = []
                start_pos = i

        elif char in UNIVERSAL_SPLIT_PUNCTUATION:
            # 分隔标点：当前字符不属于当前句子，句子到此为止
            if current:
                sentence = "".join(current).strip()
                if sentence:
                    results.append({
                        'text': sentence,
                        'start': start_pos,
                        'end': i
                    })
                current = []
            # start_pos 设置为下一个字符的位置
            start_pos = i + 1
            i += 1
        elif char == ' ':
            # 空格断句：当空格在两个文字中间时断句
            if current and i + 1 < n and text[i + 1].strip():
                # 确保空格前后都有文字
                sentence = "".join(current).strip()
                if sentence:
                    results.append({
                        'text': sentence,
                        'start': start_pos,
                        'end': i
                    })
                current = []
                # start_pos 设置为下一个字符的位置
                start_pos = i + 1
                i += 1
            else:
                # 空格在开头或结尾，或者连续空格，不进行断句
                current.append(char)
                i += 1
        else:
            current.append(char)
            i += 1

    if current:
        sentence = "".join(current).strip()
        if sentence:
            results.append({
                'text': sentence,
                'start': start_pos,
                'end': i
            })

    # 如果需要，返回规范化后的文本
    if return_normalized:
        for result in results:
            result['text'] = normalize_punctuation(result['text'])

    return results
