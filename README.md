
---

# 字幕翻译工具 (Subtitle-Translation)

一个功能强大、界面友好的视频字幕翻译解决方案，基于先进的 AI 模型提供高质量的语音识别和翻译功能。

## 项目概述

Subtitle-Translation 是一款专业的视频字幕翻译工具，利用最新的 Whisper-CD 技术实现高精度的语音识别，并采用腾讯 HY-MT1.5 模型进行智能翻译。

### 核心功能

- 🎯 **多语言支持** – 支持 15 种语言的自动检测和手动选择
- 📹 **视频处理** – 支持多种视频格式的处理和音频提取
- 🔍 **高精度识别** – 使用 Whisper 模型和 Whisper-CD 技术进行语音识别
- 🌐 **智能翻译** – 使用 HY-MT1.5-1.8B-GGUF 模型进行高质量翻译
- 📝 **字幕生成** – 支持多种字幕格式输出
- 🖥️ **友好的 Web 界面** – 基于 Gradio 构建的现代 Web 界面
- 🚀 **批处理能力** – 支持多文件上传和队列管理
- 🔧 **丰富的配置选项** – 可调整模型参数、Whisper-CD、翻译策略等
- 💾 **参数保存** – 支持保存和加载参数配置
- 📊 **实时进度** – 处理过程中实时显示进度信息

### 技术栈

| 组件           | 技术                                                         |
| -------------- | ------------------------------------------------------------ |
| 语音识别       | OpenAI Whisper + Whisper-CD                                  |
| 翻译           | 腾讯 HY-MT1.5-1.8B-GGUF (Q8_0)                               |
| UI 框架        | Gradio                                                       |
| 视频处理       | FFmpeg                                                       |
| 模型推理       | llama.cpp                                                    |
| 队列管理       | 自定义队列系统                                               |

## 快速开始

### 系统要求

- **操作系统**：Windows 11
- **Python**：3.9 ~ 3.12
- **内存**：推荐 8 GB 以上
- **GPU**：支持 CUDA 的 GPU（可选，但推荐用于加速处理）
- **磁盘空间**：至少 20 GB（用于模型和临时文件）

### 安装步骤

> **无需配置任何环境，也无需手动下载任何程序，只需按顺序运行两个脚本即可完成所有安装和配置。**

1. **获取项目**

   ```bash
   git clone https://github.com/yourusername/Subtitle-Translation.git
   cd Subtitle-Translation
   ```

2. **第一步：运行安装脚本**

   以**管理员身份**运行 `venv_setup.bat`。

   该脚本会自动完成以下操作：
   - 清理旧的虚拟环境
   - 检查并安装 Python 3.11.7（如果未安装）
   - 创建新的虚拟环境 `.venv_final`
   - 升级 pip
   - 安装所有必要的依赖包（包括 PyTorch、Gradio、Whisper 等）
   - 验证安装

3. **第二步：运行启动脚本**

   运行 `start.bat`。该脚本会自动：
   - 检查并安装 Visual C++ Redistributable（如果未安装）
   - 检查 Python 环境
   - 提供菜单选项：
     1. 下载所有模型
     2. 运行 UI
     3. 先下载模型再运行 UI
     4. 退出

### 快速启动指南

1. **首次使用**：运行 `start.bat` 并选择选项 `3`（先下载模型再运行 UI）
2. **后续使用**：运行 `start.bat` 并选择选项 `2`（直接运行 UI）
3. **浏览器访问**：应用会自动打开 `http://localhost:7860`
4. **上传视频**：点击“选择视频文件”按钮上传视频
5. **添加到队列**：点击“添加到队列”按钮
6. **开始处理**：点击“处理队列”按钮
7. **查看结果**：处理完成后，字幕文件会保存在 `outputs` 目录中

## 功能特性

### Whisper-CD 技术

> Ahn, Hoseong, et al. "Whisper-CD: Accurate Long-Form Speech Recognition using Multi-Negative Contrastive Decoding." *arXiv preprint arXiv:2603.06193* (2026).  
> [https://arxiv.org/abs/2603.06193](https://arxiv.org/abs/2603.06193)

Whisper-CD 是一种针对长语音的高精度识别方法，通过多负样本对比解码显著降低重复和漏句错误，特别适用于视频字幕生成场景。

### 模型下载

- **Whisper 模型**：`large-v3-turbo`（语音识别）
- **翻译模型**：`HY-MT1.5-1.8B-GGUF Q8_0`（智能翻译）
- **下载加速**：使用 HF-Mirror 镜像站和 aria2 多线程下载

### 模块关系

| 模块                     | 功能               | 依赖                       |
| ------------------------ | ------------------ | -------------------------- |
| `ui.py`                  | Web 界面           | Gradio                     |
| `queue_manager.py`       | 队列管理           | `video_processor`          |
| `video_processor.py`     | 视频处理           | FFmpeg                     |
| `speech_recognizer.py`   | 语音识别           | Whisper                    |
| `whisper_cd_original.py` | Whisper-CD 处理    | Whisper                    |
| `translator.py`          | 翻译处理           | `llama_server_manager`     |
| `llama_server_manager.py`| LLM 服务器管理     | llama.cpp                  |
| `subtitle_generator.py`  | 字幕生成           | -                          |
| `config.py`              | 配置管理           | -                          |

## 项目结构

```
Subtitle-Translation/
├── .venv_final/                     # 虚拟环境目录
├── aria2-1.37.0-win-64bit-build1/   # Aria2 下载工具
├── ffmpeg/                          # FFmpeg 视频处理工具
├── llama_cpp/                       # llama.cpp 模型量化库
├── models/                          # 模型存储目录
│   ├── openai--whisper-large-v3-turbo/           # Whisper 语音识别模型
│   │   └── model.safetensors
│   ├── tencent--HY-MT1.5-1.8B-GGUF/              # HY-MT1.5 翻译模型
│   │   └── HY-MT1.5-1.8B-Q8_0.gguf
│   └── jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn/  # wav2vec2 模型（备用）
│       └── pytorch_model.bin
├── outputs/                         # 输出字幕文件目录
├── packages/                        # 依赖包存储
├── temp/                            # 临时文件目录
├── utils/                           # 工具函数目录
│   ├── language_ratio_detector.py
│   ├── llama_server_manager.py
│   ├── logger.py
│   ├── punctuation_splitter.py
│   ├── queue_manager.py
│   ├── speech_recognizer.py
│   ├── subtitle_generator.py
│   ├── translator.py
│   ├── video_processor.py
│   └── whisper_cd_original.py
├── aria2_downloader.py              # Aria2 下载器
├── config.py                        # 配置管理模块
├── download_all_models.py           # 模型下载脚本
├── download_ffmpeg.py               # FFmpeg 下载脚本
├── download_llama_cpp.py            # llama.cpp 下载脚本
├── venv_setup.bat                   # 环境安装脚本
├── start.bat                        # 启动脚本
├── ui.py                            # Gradio UI 界面
└── README.md                        # 项目说明文档
```

## 许可证

本项目采用 **GNU Affero General Public License v3 (AGPLv3)** 开源许可证。  
详见 [LICENSE](https://www.gnu.org/licenses/agpl-3.0.html) 文件。

## 致谢

- **OpenAI** – Whisper 模型
- **腾讯** – HY-MT1.5-1.8B-GGUF 模型
- **Gradio** – Web 界面框架
- **FFmpeg** – 视频处理工具
- **llama.cpp** – 模型量化库

---

**Subtitle-Translation** – 让视频字幕翻译变得简单高效！