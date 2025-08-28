# img2digits-api

A lightweight **Digits OCR API** that extracts numeric text from images you own or have rights to process.  
Built with **Flask**, **EasyOCR (PyTorch)**, and **OpenCV**.

> ⚠️ **Acceptable Use**  
> This service is for processing images you are authorized to handle (e.g., scanned forms, receipts, internal datasets).  
> **Do not** use it to defeat or bypass third-party access controls, rate limits, or CAPTCHAs.

---

## Features

- Accepts images via **file upload**, **base64**, **data URLs**, or **direct image URLs**
- Returns both the **raw OCR** and a **digits-only** version (e.g., “1 2,3” → `123`)
- Simple **health** and **info** endpoints
- CORS enabled for local dev integrations

---

## Quickstart

### 1) Requirements

- Python 3.9+ recommended
- pip packages:
  - `flask`, `flask-cors`, `easyocr`, `opencv-python` (or `opencv-python-headless`), `pillow`, `numpy`, `requests`, `touchtouch`, `tolerant_isinstance`
- **PyTorch** (required by EasyOCR)

Install PyTorch per your environment from the official site. Examples:

```bash
# CPU-only (generic):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or CUDA 12.1 build (example; adjust for your GPU/driver):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


pip install flask flask-cors easyocr opencv-python pillow numpy requests touchtouch tolerant_isinstance
```

```bash 
.
├─ server.py            # Server
├─ solver.py            # OCR & preprocessing
├─ requirements.txt     # optional
└─ README.md
```
# Captcha Example:

<img width="150" height="50" alt="image" src="https://github.com/user-attachments/assets/bca61727-bf84-4770-9265-4d564cf7a874" />

# Python Request Example:
```Bash
import requests

base64_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAAAyCAYAAA|etc..."
url = "http://localhost:5000"
payload = {'url': base64_url}

response = requests.get(url, params=payload)
print(response.text)
```

```Bash
Successful Solved Response: 
{
    "success": true,
    "result": "1967",
    "raw_result": "1967"
}

```
