# Path: inference.py
import torch
import numpy as np
import cv2
import argparse
import random
import os
import traceback
from model.model import VideoModel 
from utils.cvtransforms import CenterCrop

def preprocess_video(video_path, target_frames=25, frame_size=(96, 96), crop_size=(88, 88)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Tidak dapat membuka file video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        y1 = (h - frame_size[0]) // 2
        x1 = (w - frame_size[1]) // 2
        cropped_frame = frame[y1:y1+frame_size[0], x1:x1+frame_size[1]]
        frames.append(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if not frames:
        raise ValueError("Tidak dapat membaca frame dari video.")

    inputs = np.stack(frames, axis=0)
    
    num_frames = inputs.shape[0]
    if num_frames > target_frames:
        start = (num_frames - target_frames) // 2
        inputs = inputs[start : start + target_frames, :, :]
    elif num_frames < target_frames:
        padding = np.tile(inputs[-1, :, :], (target_frames - num_frames, 1, 1))
        inputs = np.concatenate([inputs, padding], axis=0)

    inputs = CenterCrop(inputs, crop_size)
    inputs = torch.FloatTensor(inputs[:, np.newaxis, ...]) / 255.0
    return inputs.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Inference untuk ARN-Lipreading")
    parser.add_argument('--video', type=str, required=True, help='Path ke file video tunggal.')
    parser.add_argument('--weights', type=str, required=True, help='Path ke file model .pt yang sudah di-train.')
    cli_args = parser.parse_args()

    with open('label_sorted.txt') as f:
        labels = f.read().splitlines()

    class DummyArgs:
        def __init__(self):
            self.border = True
            self.se = True
            self.n_class = len(labels)
            # Pastikan model menggunakan implementasi standar Jamba
            self.use_mamba_kernels = False
    
    model_args = DummyArgs()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoModel(model_args)
    
    checkpoint = torch.load(cli_args.weights, map_location=device)
    state_dict = {k.replace('module.',''): v for k, v in checkpoint.get('video_model', checkpoint).items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model {os.path.basename(cli_args.weights)} berhasil dimuat.")
    
    try:
        video_tensor = preprocess_video(cli_args.video)
        video_tensor = video_tensor.to(device)
        
        with torch.no_grad():
            # Buat dummy border tensor dengan bentuk yang benar [batch, frames]
            # Di dalam model, ini akan diubah menjadi [batch, frames, 1]
            dummy_border_tensor = torch.ones(video_tensor.shape[0], video_tensor.shape[1]).to(device)
            outputs = model(video_tensor, border=dummy_border_tensor)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_label = labels[predicted_idx.item()]
        
        print("\n--- Hasil Prediksi ---")
        print(f"Video: {os.path.basename(cli_args.video)}")
        print(f"Prediksi Kata: ✨ {predicted_label.upper()} ✨")
        print(f"Tingkat Keyakinan: {confidence.item():.2%}")

    except Exception as e:
        print(f"Error saat memproses video: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()