# 视频转字幕工具 (Subtitle Translation)

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI--Whisper-20250625-orange.svg)](https://github.com/openai/whisper)

简洁高效的视频转字幕与翻译解决方案，基于先进的AI模型提供高质量的语音识别和多语言翻译服务。

## 功能特性

- **多模型语音识别**：支持 Whisper 系列模型 (tiny, base, small, medium, large-v2, large-v3)
- **智能翻译**：集成 M2M100 翻译模型，支持多种语言互译
- **批量处理**：支持队列管理，可批量处理多个视频文件
- **可视化界面**：基于 Gradio 构建的现代化 Web UI
- **GPU 加速**：支持 CUDA 加速，大幅提升处理速度
- **参数配置**：灵活的参数调整，满足不同场景需求
- **VAD 过滤**：内置语音活动检测，提高识别准确率
- **进度监控**：实时显示处理进度和队列状态

## 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 或更高 |
| 内存 | 8GB RAM | 16GB+ RAM |
| 存储空间 | 10GB 可用空间 | 20GB+ 可用空间 |
| GPU (可选) | - | NVIDIA GPU with CUDA 12.6+ |

### 软件要求

- **操作系统**：Windows 10/11
- **Python**：3.10 或更高版本
- **FFmpeg**：用于视频和音频处理

## 快速开始

### 1. 环境准备

```batch
# 进入项目目录
cd Subtitle-Translation

# 运行自动设置脚本（一键安装虚拟环境和依赖）
venv_setup.bat
```

### 2. 下载模型

```batch
# 运行模型下载脚本
python download_all_models.py
```

### 3. 启动应用

```batch
# 双击运行或在命令行中执行
start.bat
```

**或手动启动：**
```batch
# 激活虚拟环境
venv\Scripts\activate

# 启动应用
python ui.py
```

启动后，在浏览器中打开：`http://localhost:7868`

## 项目结构

```
Subtitle-Translation/
├── utils/                      # 工具模块
│   ├── model_cache_manager.py  # 模型缓存管理
│   ├── queue_manager.py        # 队列管理
│   ├── speech_recognizer.py    # 语音识别
│   ├── subtitle_generator.py   # 字幕生成
│   ├── translator.py           # 翻译模块
│   └── video_processor.py      # 视频处理
├── models/                      # 模型缓存目录 (自动创建)
├── temp/                        # 临时文件目录 (自动创建)
├── outputs/                     # 输出文件目录 (自动创建)
├── config.py                    # 配置文件
├── ui.py                        # 主界面
├── requirements.txt             # 依赖列表
├── download_all_models.py       # 模型下载脚本
├── setup_all.py                 # 完整设置脚本
├── start.bat                    # Windows 启动脚本
└── venv_setup.bat               # Windows 环境设置脚本
```

## 使用说明

### 处理单个文件

1. 在 **处理单个文件** 标签页中，点击「选择视频文件」上传视频
2. 调整参数设置（可选）：
   - 选择语音识别模型
   - 选择翻译模型
   - 选择运行设备 (auto/cpu/cuda)
3. 点击「处理单个文件」按钮
4. 等待处理完成后，下载生成的字幕文件

### 批量处理

1. 在 **批量处理** 标签页中，选择多个视频文件
2. 点击「添加到队列」
3. 根据需要调整队列（删除文件、清空队列）
4. 点击「开始处理队列」
5. 查看队列状态和统计信息

### 参数配置

#### 语音识别参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| Beam Size | 搜索宽度，值越大准确率越高但速度越慢 | 1 |
| VAD 过滤 | 过滤非语音部分，提高识别准确率 | 开启 |
| 单词时间戳 | 为每个单词添加时间戳 | 关闭 |
| 基于先前文本 | 利用上下文提高识别准确率 | 关闭 |

#### 翻译参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| Beam Size | 搜索宽度，值越大翻译质量越高但速度越慢 | 1 |
| 最大长度 | 翻译结果的最大长度 | 256 |
| 早停 | 在找到合适结果后提前停止 | 开启 |

## 模型选择指南

### 语音识别模型 (Whisper)

| 模型 | 大小 | 速度 | 准确率 | 适用场景 |
|------|------|------|--------|----------|
| tiny | ~39MB | ⚡⚡⚡ | ⭐⭐ | 快速预览 |
| base | ~74MB | ⚡⚡ | ⭐⭐⭐ | 日常使用 |
| small | ~244MB | ⚡ | ⭐⭐⭐⭐ | 平衡之选 |
| medium | ~769MB | 🐢 | ⭐⭐⭐⭐⭐ | 高质量需求 |
| large-v2 | ~1.5GB | 🐢🐢 | ⭐⭐⭐⭐⭐ | 专业级 |
| large-v3 | ~1.5GB | 🐢🐢 | ⭐⭐⭐⭐⭐+ | 最新最佳 |

### 翻译模型 (M2M100)

| 模型 | 大小 | 特点 |
|------|------|------|
| m2m100_418M | ~418MB | 速度快，适合日常使用 |
| m2m100_1.2B | ~1.2GB | 准确率高，适合专业场景 |

## 常见问题

### Q: 模型下载失败怎么办？
A: 检查网络连接，或手动从 Hugging Face 下载模型到 `models/` 目录。

### Q: 如何启用 GPU 加速？
A: 确保已安装 CUDA 12.6+ 和对应的 PyTorch 版本，在参数设置中选择 `cuda` 设备。

### Q: 支持哪些视频格式？
A: 支持 MP4, AVI, MOV, MKV, WMV 等常见视频格式。

### Q: 生成的字幕文件是什么格式？
A: 默认生成 SRT 格式字幕，可在大多数视频播放器中使用。

## 技术栈

- **语音识别**：[OpenAI Whisper](https://github.com/openai/whisper) / [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- **机器翻译**：[Facebook M2M100](https://huggingface.co/facebook/m2m100_418M)
- **Web UI**：[Gradio](https://gradio.app/)
- **视频处理**：[FFmpeg](https://ffmpeg.org/), [MoviePy](https://zulko.github.io/moviepy/)
- **深度学习**：[PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers/index)

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

- OpenAI 团队开发的 Whisper 模型
- Hugging Face 提供的预训练模型和工具
- Gradio 团队开发的优秀 UI 框架

---

**如有问题，请提交 Issue 或联系开发者。**
