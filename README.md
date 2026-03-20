# 🌲 Wood Species Identifier — Fast Cloud Edition

**ONNX Runtime version** — deploys in ~1 minute instead of ~8 minutes.

| | PyTorch version | ONNX version (this) |
|---|---|---|
| **Cold start** | ~8 min | ~1-2 min |
| **Install size** | ~800 MB | ~50 MB |
| **Inference speed** | ~200 ms/img | ~150 ms/img |
| **Grad-CAM** | Yes | No (use local v5 app) |
| **All other features** | Yes | Yes |

## Deploy in 5 minutes

### 1. Convert model to ONNX (run once in Colab)
Run `convert_to_onnx.py` in Google Colab. This creates:
- `fsd_resnet50.onnx` (~95 MB)
- `fsd_classes.json` (~5 KB)

### 2. Share both files on Google Drive
- Right-click each → Share → "Anyone with the link"
- Extract file IDs from the share URLs

### 3. Update app.py (lines 25-26)
```python
ONNX_FILE_ID = "your_onnx_file_id"
CLASSES_FILE_ID = "your_classes_file_id"
```

### 4. Push to GitHub & deploy
```bash
git init && git add . && git commit -m "Fast wood identifier"
git branch -M main
git remote add origin https://github.com/YOUR_USER/wood-species-identifier.git
git push -u origin main
```
Then deploy at [share.streamlit.io](https://share.streamlit.io) → select repo → main file: `app.py`

---
*TÜBİTAK 1002 Project*
