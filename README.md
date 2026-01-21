
# DocReconstruct - Intelligent Document Reconstruction & Translation

## Project Overview

**DocReconstruct** is an AI-native application designed to solve the complex problem of **layout-preserving document translation**. Built as a portfolio project for an AI Product Internship, it demonstrates the ability to integrate state-of-the-art Computer Vision (OCR) with Large Language Models (LLM) into a seamless, user-friendly product.

Unlike traditional translation tools that lose formatting, DocReconstruct understands the document structure (headers, paragraphs, lists) and reconstructs a translated PDF that visually mirrors the original.

## Key Features

*   **📄 Full PDF Support**: Handles multi-page PDF documents, converting them page-by-page for analysis.
*   **🖼️ Intelligent Layout Analysis**: Powered by **PaddleOCR v3 (PP-Structure)**, it accurately detects and classifies document regions (Text, Title, Header, Footer).
*   **🤖 Context-Aware Translation**: Integrates with OpenAI-compatible APIs (e.g., DeepSeek, Tencent Hunyuan) to provide high-quality, context-aware translations.
*   **✨ Layout Reconstruction**: Generates a new PDF where translated text is overlaid on the original layout, preserving the visual structure.
*   **🚀 Modern UI/UX**: A clean, responsive web interface built with **FastAPI** and **Tailwind CSS**, featuring real-time progress tracking and side-by-side previews.

## Technical Architecture

*   **Backend**: Python, FastAPI, Uvicorn
*   **OCR Engine**: PaddleOCR v3 (PP-Structure) with Layout Analysis
*   **PDF Processing**: PyMuPDF (fitz) for rendering, ReportLab for generation
*   **LLM Integration**: OpenAI SDK (Compatible with SiliconFlow/DeepSeek)
*   **Frontend**: HTML5, Tailwind CSS, JavaScript (Vanilla)

## Why This Project?

This project showcases the "AI Product Manager" mindset:
1.  **Identifying a User Need**: Reading foreign language papers/docs while keeping context is hard.
2.  **Rapid Prototyping**: Leveraged AI Agents (Trae/Claude) to build a functional MVP in record time.
3.  **End-to-End Delivery**: From backend logic to frontend polish, delivering a complete "product" not just a script.
4.  **Technological Integration**: Successfully combining CV and NLP models in a practical workflow.

## Getting Started

1.  **Prerequisites**:
    *   Python 3.8+
    *   Conda environment with `paddlepaddle` and `paddleocr` installed.
    *   `pip install fastapi uvicorn pymupdf reportlab openai`

2.  **Run the Application**:
    ```bash
    python app.py
    ```

3.  **Usage**:
    *   Open `http://127.0.0.1:8000`.
    *   Upload a PDF or Image.
    *   Enter your API Key (or use the default).
    *   Download the translated Markdown or Reconstructed PDF.

