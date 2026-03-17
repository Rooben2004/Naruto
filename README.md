# Naruto Hand Sign Recognition System 🌀

A real-time hand gesture recognition system for Naruto Jutsu hand seals, powered by Computer Vision and Deep Learning (PyTorch + MediaPipe).

## Project Structure 📁

- `data/raw/`: Contains the training and testing datasets.
- `models/`: Stores the trained PyTorch models (`naruto_model_gpu.pth`).
- `scripts/`: Utility scripts for training.
- `main.py`: The core application for real-time detection via webcam.
- `requirements.txt`: Python dependencies.

## Setup ⚙️

1. **Environment**:
   ```bash
   python -m venv naruto_env
   source naruto_env/bin/activate  # Windows: naruto_env\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For GPU support, ensures you have CUDA installed.*

3. **Train the Model** (Optional, if you have new data):
   ```bash
   python scripts/train.py
   ```

4. **Run Detection**:
   ```bash
   python main.py
   ```

## Hand Signs Supported ✋
Bird, Boar, Dog, Dragon, Hare, Horse, Monkey, Ox, Ram, Rat, Snake, Tiger.

## GPU Support 🚀
This version is optimized for NVIDIA GPUs using PyTorch. If a GPU is detected, it will automatically use CUDA for inference and training.

#Dataset
https://www.kaggle.com/datasets/vikranthkanumuru/naruto-hand-sign-dataset
