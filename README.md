# 字幕翻译工具 (Subtitle-TranslationA)

一个功能强大、界面友好的视频字幕翻译解决方案，基于先进的AI模型提供高质量的语音识别和翻译功能。

## 项目概述

Subtitle-TranslationA 是一款专业的视频字幕翻译工具，利用最新的Whisper-CD实现高精度的语音识别。

### 核心功能

- 🎯 **多语言支持**：支持15种语言的自动检测和手动选择
- 📹 **视频处理**：支持多种视频格式的处理和音频提取
- 🔍 **高精度识别**：使用Whisper模型和Whisper-CD技术进行语音识别
- 🌐 **智能翻译**：使用HY-MT1.5-7B-GGUF模型进行高质量翻译
- 📝 **字幕生成**：支持多种字幕格式输出
- 🖥️ **友好的Web界面**：基于Gradio构建的现代Web界面
- 🚀 **批处理能力**：支持多文件上传和队列管理
- 🔧 **丰富的配置选项**：可调整模型参数、VAD设置等
- 💾 **参数保存**：支持保存和加载参数配置
- 📊 **实时进度**：处理过程中实时显示进度信息

### 技术栈

- **语音识别**：OpenAI Whisper + Whisper-CD
- **翻译**：腾讯HY-MT1.5-7B-GGUF
- **UI框架**：Gradio
- **视频处理**：FFmpeg
- **模型推理**：llama.cpp
- **队列管理**：队列系统


## 快速开始

### 系统要求

- **操作系统**：Windows 11
- **Python**：python<=3.12.0 python=>3.9.0
- **内存**：推荐8GB以上
- **GPU**：支持CUDA的GPU（可选，但推荐用于加速处理）
- **磁盘空间**：至少20GB（用于模型和临时文件）

### 安装步骤

**无需配置任何环境，也无需手动下载任何程序，只需按顺序运行两个脚本即可完成所有安装和配置。**

1. **获取项目**
   ```bash
   # 克隆项目或下载压缩包
   git clone https://github.com/yourusername/Subtitle-TranslationA.git
   cd Subtitle-TranslationA
   ```

2. **第一步：运行安装脚本**
   ```bash
   venv_setup.bat（用管理员身份运行）
   ```

   这个脚本会自动：
   - 清理旧的虚拟环境
   - 检查并安装Python 3.11.7（如果未安装）
   - 创建新的虚拟环境 `.venv_final`
   - 升级pip
   - 安装所有必要的依赖包（包括PyTorch、Gradio、Whisper等）
   - 验证安装

3. **第二步：运行启动脚本**
   ```bash
   start.bat
   ```

   这个脚本会自动：
   - 检查并安装Visual C++ Redistributable（如果未安装）
   - 检查Python环境
   - 提供菜单选项：
     1. 下载所有模型
     2. 运行UI
     3. 先下载模型再运行UI
     4. 退出

### 快速启动指南

1. **首次使用**：运行 `start.bat` 并选择选项 `3`（先下载模型再运行UI）
2. **后续使用**：运行 `start.bat` 并选择选项 `2`（直接运行UI）
3. **浏览器访问**：应用会自动打开 http://localhost:7860
4. **上传视频**：点击"选择视频文件"按钮上传视频
5. **添加到队列**：点击"添加到队列"按钮
6. **开始处理**：点击"处理队列"按钮
7. **查看结果**：处理完成后，字幕文件会保存在 `outputs` 目录中

## 功能特性

### Whisper-CD 技术

实现了论文《Whisper-CD: Accurate Long-Form Speech Recognition using Multi-Negative Contrastive Decoding》中的对比解码算法，该论文由 Hoseong Ahn、Jeongyun Chae、Yoonji Park 和 Kyuhong Shim 发表。

**论文摘要**：长形式语音识别使用大型编码器-解码器模型（如 Whisper）通常会出现幻觉、重复循环和内容遗漏。当使用前一个片段的转录作为解码上下文时，这些错误会累积并进一步放大。本研究提出了 Whisper-CD，一种无训练的对比解码框架，通过三种声学激励扰动（高斯噪声注入、静音信号和音频时间移位）计算负logits，并与干净音频的logits进行对比。通过 log-sum-exp 算子聚合这些负样本，构建统一的多负目标用于逐token解码。在五个英语长形式基准测试中，Whisper-CD 在 CORAAL 上将 WER 降低了高达 24.3pp，并显示出比 beam search 快 48% 的token生成吞吐量。由于 Whisper-CD 纯在推理时操作，它可以作为已部署 Whisper 系统的即插即用替代品，无需重新训练。


### 模型下载

- **Whisper模型**：large-v3-turbo（用于语音识别）
- **翻译模型**：HY-MT1.5-7B-GGUF（用于智能翻译）
- **下载加速**：使用HF-Mirror和aria2多线程下载


### 模块关系

| 模块 | 功能 | 依赖 |
|------|------|------|
| ui.py | Web界面 | Gradio |
| queue_manager.py | 队列管理 | video_processor |
| video_processor.py | 视频处理 | ffmpeg |
| speech_recognizer.py | 语音识别 | Whisper |
| whisper_cd_original.py | Whisper-CD处理 | Whisper |
| translator.py | 翻译处理 | llama_server_manager |
| llama_server_manager.py | LLM服务器管理 | llama.cpp |
| subtitle_generator.py | 字幕生成 | - |
| config.py | 配置管理 | - |


## 项目结构

```
Subtitle-TranslationA/
├── .venv_final/           # 虚拟环境目录
├── aria2-1.37.0-win-64bit-build1/  # Aria2下载工具
├── ffmpeg/                # FFmpeg视频处理工具
├── llama_cpp/             # llama.cpp模型量化库
├── models/                # 模型存储目录
├── outputs/               # 输出字幕文件目录
├── packages/              # 依赖包存储
├── temp/                  # 临时文件目录
├── utils/                 # 工具函数目录
│   ├── language_ratio_detector.py  # 语言占比检测
│   ├── llama_server_manager.py     # LLM服务器管理
│   ├── logger.py                  # 日志管理
│   ├── punctuation_splitter.py     # 标点断句
│   ├── queue_manager.py            # 队列管理
│   ├── speech_recognizer.py        # 语音识别
│   ├── subtitle_generator.py       # 字幕生成
│   ├── translator.py               # 翻译功能
│   ├── video_processor.py          # 视频处理
│   └── whisper_cd_original.py      # Whisper-CD实现
├── aria2_downloader.py    # Aria2下载器
├── config.py              # 配置管理模块
├── download_all_models.py # 模型下载脚本
├── download_ffmpeg.py     # FFmpeg下载脚本
├── download_llama_cpp.py  # llama.cpp下载脚本
├── venv_setup.bat         # 环境安装脚本
├── start.bat              # 启动脚本
├── ui.py                  # Gradio UI界面
└── README.md              # 项目说明文档
```



## 许可证

本项目采用 **GNU Affero General Public License v3 (AGPLv3)** 开源许可证。


## 致谢


- **OpenAI**：Whisper模型
- **腾讯**：HY-MT1.5-7B-GGUF模型
- **Gradio**：Web界面框架
- **FFmpeg**：视频处理工具
- **llama.cpp**：模型量化库



**Subtitle-TranslationA** - 让视频字幕翻译变得简单高效！
