# PaddleOCR PDF 解析与翻译 Web 服务

这是一个基于 PaddleOCR-VL 和 OpenAI 兼容 API (如硅基流动) 的本地 Web 服务，专为深度学习论文解析和翻译设计。

## 功能特点

- **美观简洁的界面**：基于 Tailwind CSS 设计。
- **PDF/图片解析**：使用 PaddleOCR-VL 进行文档解析，支持版面分析。
- **智能翻译**：集成 OpenAI 兼容接口（如 SiliconFlow），支持长文档 Markdown 翻译。
- **参数可配置**：
  - 支持开启/关闭文档方向分类 (`use_doc_orientation_classify`)
  - 支持开启/关闭文档去弯曲 (`use_doc_unwarping`)
  - 自定义布局检测阈值 (`layout_threshold`)
  - 自定义翻译 API Key、Base URL、模型和提示词
- **深度学习论文优化**：默认参数已针对论文阅读进行优化。

## 环境要求

- Windows 11
- Conda 环境: `paddle`
- Python 3.x

## 快速开始

1. **启动服务**

   在终端中运行以下命令：
   ```powershell
   conda run -n paddle python 1.py
   ```

2. **访问网页**

   服务启动后，打开浏览器访问：
   [http://localhost:8000](http://localhost:8000)

3. **使用说明**

   - **上传文件**：点击上传区域选择 PDF 文件或图片。
   - **参数设置**：
     - **OCR 设置**：默认已勾选适合论文的配置（布局检测开启，阈值 0.5）。如果不涉及弯曲文档，建议关闭“方向分类”和“去弯曲”以提高速度。
     - **翻译设置**：
       - **API Key**：默认已预填您的 Key。
       - **Base URL**：默认为 `https://api.siliconflow.cn/v1`。
       - **Model**：默认为 `tencent/Hunyuan-MT-7B`。
       - **高级设置**：点击“高级参数设置”可调整 Temperature 和 Top P。
   - **开始处理**：点击“开始处理”按钮。
   - **结果下载**：处理完成后，页面会显示 Markdown 结果和翻译结果，并提供下载链接。

## 输出目录

所有处理结果将保存在 `output` 目录下，按任务 ID 分类。

## 注意事项

- 首次运行可能需要下载 PaddleOCR 模型文件。
- 翻译功能依赖外部 API，请确保网络通畅且 Key 有效。
- 仅供本地使用。
