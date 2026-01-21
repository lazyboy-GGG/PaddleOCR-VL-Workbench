
import os
import shutil
import time
import uuid
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

# Third-party Imports
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# PDF Processing
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("Warning: PyMuPDF (fitz) not found. PDF support will be limited.")

# PaddleOCR
try:
    # Try importing the multimodal pipeline
    from paddleocr import PaddleOCRVL
    print("Using PaddleOCR-VL Pipeline")
except ImportError:
    try:
        from paddleocr import PPStructureV3 as PaddleOCRVL # Fallback or similar interface
        print("Using PPStructureV3 as fallback")
    except ImportError:
        print("Warning: paddleocr not found. Please install it in 'paddle' env.")
        PaddleOCRVL = None

# OpenAI
from openai import OpenAI

# PDF Generation
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY

# -------------------------------------------------------------------------
# 1. Configuration & Logging
# -------------------------------------------------------------------------

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("OCR-App")

# Paths
BASE_DIR = Path(__file__).parent.absolute()
OUTPUT_BASE_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp_workspace"

# Ensure Dirs
OUTPUT_BASE_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Fonts
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
FONT_NAME = "SimHei"
if os.path.exists(FONT_PATH):
    try:
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
        logger.info(f"Loaded font: {FONT_NAME}")
    except Exception as e:
        logger.error(f"Failed to load font: {e}")
        FONT_NAME = "Helvetica"
else:
    logger.warning("SimHei font not found. Chinese support may be limited.")
    FONT_NAME = "Helvetica" # Fallback

# -------------------------------------------------------------------------
# 2. Core Modules
# -------------------------------------------------------------------------

class FileManager:
    """Manages temporary workspace and final output promotion."""
    
    @staticmethod
    def create_session(filename: str) -> str:
        """Creates a unique session directory."""
        session_id = str(uuid.uuid4())
        session_dir = TEMP_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        # Store metadata
        (session_dir / "meta.json").write_text(json.dumps({
            "original_filename": filename,
            "created_at": time.time()
        }), encoding="utf-8")
        return session_id

    @staticmethod
    def get_session_dir(session_id: str) -> Path:
        return TEMP_DIR / session_id

    @staticmethod
    def promote_to_output(session_id: str) -> Optional[Path]:
        """Moves/Copies session files to final output directory."""
        session_dir = FileManager.get_session_dir(session_id)
        if not session_dir.exists():
            return None
            
        meta_path = session_dir / "meta.json"
        if not meta_path.exists():
            return None
            
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            orig_name = Path(meta["original_filename"]).stem
            
            # Target Folder: output/{filename}_dual
            target_folder_name = f"{orig_name}_dual"
            target_dir = OUTPUT_BASE_DIR / target_folder_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all files from temp to target
            shutil.copytree(session_dir, target_dir, dirs_exist_ok=True)
            logger.info(f"Promoted session {session_id} to {target_dir}")
            return target_dir
        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            return None

class PDFProcessor:
    """Handles PDF to Image conversion and processing."""
    
    @staticmethod
    def pdf_to_images(pdf_path: str, output_dir: Path) -> List[str]:
        """
        Converts PDF pages to images.
        Returns a list of image paths.
        """
        image_paths = []
        if not fitz:
            logger.warning("PyMuPDF not installed. Cannot convert PDF.")
            return []
            
        try:
            doc = fitz.open(pdf_path)
            for i in range(len(doc)):
                page = doc.load_page(i)
                # Zoom = 2 for better OCR quality
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                
                img_name = f"page_{i+1}.png"
                img_path = output_dir / img_name
                pix.save(str(img_path))
                image_paths.append(str(img_path))
                
            logger.info(f"Converted PDF to {len(image_paths)} images.")
            return image_paths
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return []

class OCREngine:
    """Encapsulates PaddleOCR-VL logic."""
    
    def __init__(self):
        self.engine = None
        if PaddleOCRVL:
            try:
                # Initialize PaddleOCR-VL
                # Note: Parameters here are initial configs. 
                # Specific behaviors like 'use_formula' might be controlled via predict or model selection.
                # Assuming standard initialization for now.
                self.engine = PaddleOCRVL(
                    use_layout_detection=True, 
                    use_doc_orientation_classify=True
                )
                logger.info("Initialized PaddleOCR-VL successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR-VL: {e}")
                self.engine = None
            
    def analyze(self, image_path: str, prompt: str = "") -> List[Dict]:
        """
        Performs analysis on the given image.
        """
        if not self.engine:
            raise RuntimeError("PaddleOCR-VL not initialized")
        
        logger.info(f"Starting analysis: {image_path}")
        
        try:
            # PaddleOCR-VL predict method
            # Support prompt_label for specific tasks if needed, though 'ocr' is default for full page
            # If the user wants formula protection, we might rely on the model detecting formulas 
            # and us filtering them during translation.
            
            res_list = self.engine.predict(input=image_path)
            
            standardized_result = []
            
            # The result structure from PaddleOCRVL usually contains 'markdown', 'json', etc.
            # We need to extract layout info for reconstruction.
            # 'json' property usually has the layout info including bboxes.
            
            for res in res_list:
                # Each 'res' is a page result
                page_json = {}
                if hasattr(res, 'json'):
                    page_json = res.json
                    if callable(page_json):
                        page_json = page_json()
                elif isinstance(res, dict):
                    page_json = res
                else:
                    try:
                        page_json = dict(res)
                    except:
                        logger.warning(f"Unknown result format: {type(res)}")
                        continue
                
                # Unwrap 'res' key if present (PaddleOCR-VL structure)
                if 'res' in page_json and isinstance(page_json['res'], dict):
                    page_json = page_json['res']

                logger.info(f"Page keys found: {list(page_json.keys())}")
                
                # Priority 1: Check for parsing_res_list (Rich text/layout info)
                if 'parsing_res_list' in page_json:
                     for item in page_json['parsing_res_list']:
                        # Support both VL (block_*) and Standard (label/bbox) keys
                        bbox = item.get('block_bbox', item.get('bbox', []))
                        label = item.get('block_label', item.get('label', 'text'))
                        text_val = item.get('block_content', item.get('text', ''))
                        
                        # Fallback for text content
                        if not text_val and 'res' in item:
                             content_obj = item['res']
                             if isinstance(content_obj, list):
                                text_val = "".join([c.get('text', '') for c in content_obj])
                             elif isinstance(content_obj, str):
                                text_val = content_obj
                                
                        standardized_result.append({
                            'type': label,
                            'bbox': bbox,
                            'res': [{'text': text_val}]
                        })

                # Priority 2: Check for layout_det_res (Basic layout info)
                elif 'layout_det_res' in page_json:
                    layout_data = page_json['layout_det_res']
                    # layout_det_res might be a dict with 'boxes' or a list
                    items = []
                    if isinstance(layout_data, dict) and 'boxes' in layout_data:
                        items = layout_data['boxes']
                    elif isinstance(layout_data, list):
                        items = layout_data
                    
                    for item in items:
                        coord = item.get('coordinate', item.get('block_bbox', []))
                        label = item.get('label', item.get('block_label', 'text'))
                        # Try multiple keys for text content
                        text_content = item.get('text', item.get('ocr_text', item.get('block_content', '')))
                        
                        standardized_result.append({
                            'type': label,
                            'bbox': coord,
                            'res': [{'text': text_content}]
                        })
                
                # Fallback: Check if there is a global 'markdown' field if layout details failed
                if not standardized_result and 'markdown' in page_json:
                    standardized_result.append({
                        'type': 'text',
                        'bbox': [0, 0, 100, 100],
                        'res': [{'text': page_json['markdown']}]
                    })

            return standardized_result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

class Translator:
    """Handles text translation using OpenAI-compatible APIs."""
    
    def __init__(self, api_key: str, base_url: str, model: str, system_prompt: str = "", temperature: float = 0.3, top_p: float = 0.7):
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        # Default prompt with LaTeX protection instruction
        default_prompt = "You are a professional translator. Translate the text to Chinese directly. Maintain the original tone and format. Do not translate LaTeX formulas or code blocks. Do not add explanations."
        self.sys_prompt = system_prompt if system_prompt.strip() else default_prompt

    def translate_batch(self, texts: List[str]) -> List[str]:
        if not self.client or not texts:
            return texts
            
        translated = []
        
        def _trans(text):
            if not text.strip(): return ""
            # Simple heuristic: If text looks like a formula (starts with $ or $$), skip
            if text.strip().startswith('$') or text.strip().startswith('\\'):
                return text

            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.sys_prompt},
                        {"role": "user", "content": text}
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Translation error: {e}")
                return text

        # Use ThreadPool for concurrent API calls
        with ThreadPoolExecutor(max_workers=5) as executor:
            translated = list(executor.map(_trans, texts))
            
        return translated

class PDFRenderer:
    """Generates layout-preserved PDFs using ReportLab."""
    
    @staticmethod
    def render_pages(
        pages_data: List[Dict], # [{'image_path': str, 'layout': list, 'trans_map': dict}]
        output_path: str
    ):
        try:
            c = canvas.Canvas(output_path)
            
            for page_idx, data in enumerate(pages_data):
                image_path = data['image_path']
                structure_res = data['layout']
                translated_map = data['trans_map']
                
                try:
                    image = Image.open(image_path)
                    width, height = image.size
                    c.setPageSize((width, height))
                    c.drawImage(image_path, 0, 0, width=width, height=height)
                except Exception as e:
                    logger.error(f"Failed to load image for PDF: {e}")
                    continue
                
                style_sheet = getSampleStyleSheet()
                normal_style = ParagraphStyle(
                    'Normal_CN',
                    parent=style_sheet['Normal'],
                    fontName=FONT_NAME,
                    fontSize=10,
                    leading=12,
                    textColor=colors.black,
                    alignment=TA_JUSTIFY
                )
                
                for idx, region in enumerate(structure_res):
                    # Skip non-text regions or formulas if we want to keep original
                    r_type = region['type']
                    if r_type in ['figure', 'table']: 
                        continue # Keep original image for charts/tables
                    
                    if r_type == 'formula':
                        continue # Keep original formula image
                        
                    text = translated_map.get(idx, "")
                    if not text:
                        continue
                        
                    bbox = region['bbox']
                    if not bbox or len(bbox) != 4: continue
                    
                    x1, y1, x2, y2 = bbox
                    rect_x = x1
                    rect_y = height - y2
                    rect_w = x2 - x1
                    rect_h = y2 - y1
                    
                    # Draw masking rectangle (White)
                    c.saveState()
                    c.setFillColor(colors.white)
                    c.rect(rect_x, rect_y, rect_w, rect_h, fill=1, stroke=0)
                    c.restoreState()
                    
                    # Dynamic font size
                    current_font_size = 10
                    if rect_h < 20: current_font_size = 8
                    if rect_h < 12: current_font_size = 6
                    
                    p_style = ParagraphStyle(
                        'Dynamic',
                        parent=normal_style,
                        fontSize=current_font_size,
                        leading=current_font_size * 1.2
                    )
                    
                    p = Paragraph(text, p_style)
                    
                    frame = Frame(
                        rect_x, rect_y, rect_w, rect_h, 
                        showBoundary=0, 
                        leftPadding=2, bottomPadding=2, rightPadding=2, topPadding=2
                    )
                    frame.addFromList([p], c)
                
                c.showPage()
                
            c.save()
            logger.info(f"PDF generated: {output_path}")
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")

# -------------------------------------------------------------------------
# 3. FastAPI Application
# -------------------------------------------------------------------------

app = FastAPI(title="DocReconstruct Pro")

app.mount("/output", StaticFiles(directory=str(OUTPUT_BASE_DIR)), name="output")
app.mount("/temp", StaticFiles(directory=str(TEMP_DIR)), name="temp")

# Global Engines
ocr_engine = OCREngine()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PaddleOCR-VL 文档重构工作台</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; background: #f0f4f8; }
        .glass-panel { background: rgba(255, 255, 255, 0.98); border: 1px solid rgba(226, 232, 240, 1); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
        .gradient-primary { background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    </style>
</head>
<body class="h-screen flex flex-col text-slate-800">
    
    <!-- Header -->
    <header class="z-20 px-6 py-3 flex justify-between items-center bg-white border-b border-slate-200 shadow-sm">
        <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded bg-blue-600 flex items-center justify-center text-white">
                <i class="fa-solid fa-cube"></i>
            </div>
            <h1 class="text-lg font-bold text-slate-800">PaddleOCR-VL <span class="font-normal text-slate-500">Workbench</span></h1>
        </div>
        <div class="text-xs text-slate-500 font-mono">Model: PaddleOCR-VL-0.9B (Local)</div>
    </header>

    <div class="flex-1 flex overflow-hidden">
        
        <!-- Sidebar -->
        <aside class="w-[360px] bg-white border-r border-slate-200 flex flex-col z-10">
            <div class="p-5 border-b border-slate-100 bg-slate-50">
                <h2 class="font-semibold text-slate-800 flex items-center gap-2 text-sm">
                    <i class="fa-solid fa-gears text-blue-500"></i> 参数配置
                </h2>
            </div>
            
            <form id="processForm" class="flex-1 overflow-y-auto p-5 space-y-6">
                <!-- File Upload -->
                <div class="space-y-2">
                    <label class="text-xs font-bold text-slate-500 uppercase">源文档</label>
                    <input type="file" name="file" id="fileInput" accept=".jpg,.jpeg,.png,.pdf" class="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer border border-slate-200 rounded-lg p-1" required>
                </div>

                <!-- LLM Config -->
                <div class="space-y-4 pt-4 border-t border-slate-100">
                    <label class="text-xs font-bold text-slate-500 uppercase">大模型翻译设置</label>
                    
                    <div>
                        <label class="block text-xs font-medium text-slate-700 mb-1">API Base URL</label>
                        <input type="text" name="base_url" value="https://api.siliconflow.cn/v1" class="w-full bg-white border border-slate-200 rounded px-3 py-2 text-xs focus:ring-1 focus:ring-blue-500 outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-700 mb-1">API Key</label>
                        <input type="password" name="api_key" placeholder="sk-..." class="w-full bg-white border border-slate-200 rounded px-3 py-2 text-xs focus:ring-1 focus:ring-blue-500 outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-700 mb-1">Model Name (支持自定义)</label>
                        <input type="text" name="model" list="model_list" value="tencent/Hunyuan-MT-7B" class="w-full bg-white border border-slate-200 rounded px-3 py-2 text-xs focus:ring-1 focus:ring-blue-500 outline-none" placeholder="输入或选择模型...">
                        <datalist id="model_list">
                            <option value="tencent/Hunyuan-MT-7B">
                            <option value="deepseek-ai/DeepSeek-V2.5">
                            <option value="gpt-3.5-turbo">
                            <option value="gpt-4o">
                        </datalist>
                    </div>
                </div>

                <!-- Advanced Params -->
                <div class="space-y-4 pt-4 border-t border-slate-100">
                    <label class="text-xs font-bold text-slate-500 uppercase">高级参数</label>
                    
                    <div>
                        <label class="block text-xs font-medium text-slate-700 mb-1">System Prompt (提示词)</label>
                        <textarea name="system_prompt" rows="3" class="w-full bg-white border border-slate-200 rounded px-3 py-2 text-xs focus:ring-1 focus:ring-blue-500 outline-none" placeholder="自定义翻译指令...">You are a professional translator. Translate the text to Chinese directly. Do not translate LaTeX formulas.</textarea>
                    </div>
                    
                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="block text-xs font-medium text-slate-700 mb-1">Temperature</label>
                            <input type="number" name="temperature" value="0.3" step="0.1" min="0" max="1" class="w-full bg-white border border-slate-200 rounded px-3 py-2 text-xs">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-700 mb-1">Top P</label>
                            <input type="number" name="top_p" value="0.7" step="0.1" min="0" max="1" class="w-full bg-white border border-slate-200 rounded px-3 py-2 text-xs">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-700 mb-1">公式保护</label>
                            <div class="flex items-center h-[34px]">
                                <label class="inline-flex items-center cursor-pointer">
                                    <input type="checkbox" name="protect_formulas" checked class="sr-only peer">
                                    <div class="relative w-9 h-5 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
                                    <span class="ms-2 text-xs font-medium text-slate-600">启用</span>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>
            </form>

            <div class="p-5 border-t border-slate-100 bg-slate-50">
                <button type="submit" form="processForm" id="submitBtn" class="w-full gradient-primary text-white font-medium py-2.5 rounded shadow-sm hover:shadow-md active:scale-[0.98] transition-all flex items-center justify-center gap-2 text-sm">
                    <i class="fa-solid fa-play"></i> 运行任务
                </button>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="flex-1 bg-slate-100 relative p-6 flex flex-col gap-4">
            <!-- Status Bar -->
            <div id="loadingState" class="hidden bg-white p-4 rounded border border-slate-200 shadow-sm flex items-center gap-4 animate-pulse">
                <i class="fa-solid fa-circle-notch animate-spin text-blue-600"></i>
                <span class="text-sm font-medium text-slate-700" id="loadingText">正在初始化任务...</span>
            </div>

            <!-- Results -->
            <div id="resultState" class="hidden flex-1 flex flex-col gap-4 overflow-hidden">
                <div class="flex gap-4 h-full">
                    <!-- Markdown View -->
                    <div class="flex-1 bg-white rounded border border-slate-200 flex flex-col">
                        <div class="px-4 py-2 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
                            <span class="text-xs font-bold text-slate-500">MARKDOWN</span>
                            <button onclick="downloadFile('md')" class="text-blue-600 hover:text-blue-700 text-xs font-medium"><i class="fa-solid fa-download"></i> 下载</button>
                        </div>
                        <div id="mdPreview" class="flex-1 overflow-y-auto p-6 prose prose-sm max-w-none"></div>
                    </div>
                    
                    <!-- PDF View -->
                    <div class="flex-1 bg-white rounded border border-slate-200 flex flex-col">
                        <div class="px-4 py-2 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
                            <span class="text-xs font-bold text-slate-500">PDF PREVIEW</span>
                            <button onclick="downloadFile('pdf')" class="text-blue-600 hover:text-blue-700 text-xs font-medium"><i class="fa-solid fa-download"></i> 下载 PDF</button>
                        </div>
                        <div class="flex-1 bg-slate-100 flex items-center justify-center">
                            <div class="text-center text-slate-400">
                                <i class="fa-solid fa-file-pdf text-4xl mb-2"></i>
                                <p class="text-sm">生成完毕</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        let currentSessionId = null;

        document.getElementById('processForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const loading = document.getElementById('loadingState');
            const result = document.getElementById('resultState');
            const btn = document.getElementById('submitBtn');
            const loadingText = document.getElementById('loadingText');
            
            loading.classList.remove('hidden');
            result.classList.add('hidden');
            btn.disabled = true;
            btn.classList.add('opacity-50');

            const steps = ["正在进行版面分析 (PaddleOCR-VL)...", "正在提取文本与公式...", "正在调用大模型翻译...", "正在重构 PDF 文档..."];
            let stepIdx = 0;
            const interval = setInterval(() => {
                loadingText.textContent = steps[stepIdx++ % steps.length];
            }, 4000);

            try {
                const formData = new FormData(e.target);
                const res = await fetch('/api/process', { method: 'POST', body: formData });
                const data = await res.json();

                if(!res.ok) throw new Error(data.detail || 'Failed');

                currentSessionId = data.session_id;
                document.getElementById('mdPreview').innerHTML = marked.parse(data.preview_md);
                
                loading.classList.add('hidden');
                result.classList.remove('hidden');
            } catch(err) {
                alert('Error: ' + err.message);
                loading.classList.add('hidden');
            } finally {
                clearInterval(interval);
                btn.disabled = false;
                btn.classList.remove('opacity-50');
            }
        });

        function downloadFile(type) {
            if(currentSessionId) window.open(`/api/download/${currentSessionId}/${type}`, '_blank');
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(HTML_TEMPLATE)

@app.post("/api/process")
async def process_document(
    file: UploadFile = File(...),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
    system_prompt: str = Form(""),
    temperature: float = Form(0.3),
    top_p: float = Form(0.7),
    protect_formulas: str = Form("on") # Checkbox sends "on" if checked
):
    try:
        # 1. Session Setup
        session_id = FileManager.create_session(file.filename)
        session_dir = FileManager.get_session_dir(session_id)
        input_path = session_dir / file.filename
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # 2. Convert to Images
        input_str = str(input_path)
        pages_to_process = []
        if input_str.lower().endswith('.pdf'):
            pages_to_process = PDFProcessor.pdf_to_images(input_str, session_dir)
            if not pages_to_process:
                raise HTTPException(400, "PDF processing failed. Please check file or PyMuPDF.")
        else:
            pages_to_process = [input_str]

        # 3. Analyze & Translate
        if not ocr_engine.engine:
            raise HTTPException(500, "PaddleOCR Engine not initialized.")

        full_md_text = ""
        all_pages_data = []

        # Setup Translator
        translator = Translator(api_key, base_url, model, system_prompt, temperature, top_p)

        for p_idx, p_img_path in enumerate(pages_to_process):
            # Analyze
            layout_res = ocr_engine.analyze(p_img_path)
            
            # Extract Text
            texts = []
            indices = []
            for idx, region in enumerate(layout_res):
                # Filter if protecting formulas? 
                # Actually, we filter at translation step or here.
                # If region type is 'formula', we skip adding it to translation list
                r_type = region['type']
                
                if protect_formulas == "on" and r_type == 'formula':
                    continue
                
                # Extract text
                res_content = region.get('res', [])
                full_text = "".join([c.get('text', '') for c in res_content]) if isinstance(res_content, list) else ""
                
                if r_type in ['text', 'title', 'header', 'footer', 'list', 'reference', 'figure_caption', 'table_caption', 'paragraph']:
                    texts.append(full_text)
                    indices.append(idx)

            # Translate
            translated_texts = translator.translate_batch(texts)
            translated_map = dict(zip(indices, translated_texts))
            
            all_pages_data.append({
                'image_path': p_img_path,
                'layout': layout_res,
                'trans_map': translated_map
            })
            
            # Markdown Build
            full_md_text += f"\n\n## Page {p_idx+1}\n\n"
            for i, txt in enumerate(texts):
                trans = translated_texts[i] if i < len(translated_texts) else ""
                full_md_text += f"**原文**: {txt}\n\n**译文**: {trans}\n\n---\n"

        # 4. Generate Output
        (session_dir / f"{Path(file.filename).stem}.md").write_text(full_md_text, encoding="utf-8")
        
        pdf_name = f"{Path(file.filename).stem}_translated.pdf"
        PDFRenderer.render_pages(all_pages_data, str(session_dir / pdf_name))

        return {
            "session_id": session_id,
            "preview_md": full_md_text[:5000]
        }

    except Exception as e:
        logger.error(f"Processing Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/download/{session_id}/{file_type}")
async def download_artifact(session_id: str, file_type: str):
    target_dir = FileManager.promote_to_output(session_id)
    if not target_dir: raise HTTPException(404, "File not found")
    
    # Simple lookup based on file type
    # ... (Similar to previous implementation)
    # Re-implementing specifically for completeness
    stem = target_dir.name.replace("_dual", "")
    if file_type == 'md':
        path = target_dir / f"{stem}.md"
        media = "text/markdown"
    else:
        path = target_dir / f"{stem}_translated.pdf"
        media = "application/pdf"
        
    if not path.exists(): raise HTTPException(404, "File missing")
    return FileResponse(path, media_type=media, filename=path.name)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
