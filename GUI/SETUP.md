# Breast Cancer Detection App - Setup Guide

## Python Version Requirements

### ✅ **Tested and Working:**
- **Python 3.13.0** (currently running and fully functional)

### 🔧 **Recommended Python Versions:**
- **Python 3.9** - 3.13 (all should work)
- **Python 3.10** or **3.11** (most stable for TensorFlow)

### ⚠️ **Important Notes:**
- TensorFlow officially supports Python 3.9-3.12
- Python 3.13 works in practice but may show compatibility warnings
- Avoid Python 3.8 or older (not supported by newer TensorFlow versions)
- Avoid Python 3.14+ (too new, likely incompatible)

## Installation Steps

### 1. **Check Python Version**
```bash
python --version
# Should show Python 3.9.x - 3.13.x
```

### 2. **Create Virtual Environment (Recommended)**
```bash
python -m venv breast_cancer_env
# Windows:
breast_cancer_env\Scripts\activate
# Linux/Mac:
source breast_cancer_env/bin/activate
```

### 3. **Install Dependencies**

**Option A: Minimal Installation (Recommended)**
```bash
pip install -r requirements-minimal.txt
```

**Option B: Full Installation (All dependencies)**
```bash
pip install -r requirements.txt
```

### 4. **Verify Installation**
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import keras; print('Keras version:', keras.__version__)"
python -c "import flask; print('Flask version:', flask.__version__)"
```

### 5. **Run the Application**
```bash
python app.py
```

The app will be available at: http://127.0.0.1:5000

## Required Files Structure
```
your_project/
├── app.py
├── requirements.txt
├── requirements-minimal.txt
├── efficientnet_aln_model_combined1.keras
├── scaler.save
├── templates/
│   ├── index.html
│   └── result.html
└── static/
    ├── uploads/
    └── confusion.jpg
```

## Troubleshooting

### If you get TensorFlow installation errors:
1. Try Python 3.10 or 3.11 instead of 3.13
2. Update pip: `pip install --upgrade pip`
3. Install TensorFlow separately: `pip install tensorflow==2.20.0rc0`

### If you get Keras version errors:
```bash
pip install keras==3.10.0 --force-reinstall
```

### If you get scikit-learn warnings:
The version warning is normal and doesn't affect functionality.

## Hardware Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 2GB free space
- **CPU**: Any modern processor (GPU not required but can help)
