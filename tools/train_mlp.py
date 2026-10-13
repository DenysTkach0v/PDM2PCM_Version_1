#!/usr/bin/env python3
"""
TinyML Baby Cry Classifier Training & C Exporter
=================================================
Trains a lightweight 3-layer Float32 MLP on 28 spectro-temporal features
(extracted by dsp_parity.py) and auto-generates Core/Inc/model_weights.h
for direct execution on STM32F429 Cortex-M4F FPU.

Classes:
    0: SOUND_CLASS_NOISE
    1: SOUND_CLASS_SPEECH
    2: SOUND_CLASS_BABY_CRY

Usage:
    python train_mlp.py [--dataset-dir dataset/] [--export-path ../Core/Inc/model_weights.h]
"""

import os
import sys
import glob
import wave
import argparse
import numpy as np

# Ensure dsp_parity is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dsp_parity import AudioDSPParity, SAMPLE_RATE

# Network Dimensions
INPUT_DIM = 28
HIDDEN1_DIM = 32
HIDDEN2_DIM = 16
OUTPUT_DIM = 3


def relu(x):
    return np.maximum(0.0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class SimpleMLP:
    """Lightweight 3-layer MLP implemented in NumPy with Adam optimizer."""
    def __init__(self, in_dim=INPUT_DIM, h1=HIDDEN1_DIM, h2=HIDDEN2_DIM, out_dim=OUTPUT_DIM):
        # He initialization
        np.random.seed(42)
        self.W1 = np.random.randn(h1, in_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.B1 = np.zeros(h1, dtype=np.float32)

        self.W2 = np.random.randn(h2, h1).astype(np.float32) * np.sqrt(2.0 / h1)
        self.B2 = np.zeros(h2, dtype=np.float32)

        self.W3 = np.random.randn(out_dim, h2).astype(np.float32) * np.sqrt(2.0 / h2)
        self.B3 = np.zeros(out_dim, dtype=np.float32)

        # Normalization statistics
        self.input_mean = np.zeros(in_dim, dtype=np.float32)
        self.input_std = np.ones(in_dim, dtype=np.float32)

    def forward(self, X):
        # Layer 1
        Z1 = np.dot(X, self.W1.T) + self.B1
        A1 = relu(Z1)

        # Layer 2
        Z2 = np.dot(A1, self.W2.T) + self.B2
        A2 = relu(Z2)

        # Layer 3
        Z3 = np.dot(A2, self.W3.T) + self.B3
        probs = softmax(Z3)

        return probs, (X, Z1, A1, Z2, A2, Z3)

    def train(self, X_train, y_train, X_val, y_val, epochs=120, lr=0.003, batch_size=32):
        # Compute input normalization
        self.input_mean = np.mean(X_train, axis=0).astype(np.float32)
        self.input_std = (np.std(X_train, axis=0) + 1e-5).astype(np.float32)

        X_train_norm = (X_train - self.input_mean) / self.input_std
        X_val_norm = (X_val - self.input_mean) / self.input_std

        num_samples = len(X_train)
        num_classes = OUTPUT_DIM

        # One-hot encode targets
        Y_onehot = np.zeros((num_samples, num_classes), dtype=np.float32)
        Y_onehot[np.arange(num_samples), y_train] = 1.0

        # Adam optimizer state
        params = [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        print(f"Training MLP ({INPUT_DIM} -> {HIDDEN1_DIM} -> {HIDDEN2_DIM} -> {OUTPUT_DIM}) on {num_samples} samples...")

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train_norm[indices]
            Y_shuffled = Y_onehot[indices]

            for i in range(0, num_samples, batch_size):
                xb = X_shuffled[i:i + batch_size]
                yb = Y_shuffled[i:i + batch_size]
                bsize = len(xb)

                # Forward pass
                probs, (X_in, Z1, A1, Z2, A2, Z3) = self.forward(xb)

                # Backprop
                dZ3 = (probs - yb) / bsize
                dW3 = np.dot(dZ3.T, A2)
                dB3 = np.sum(dZ3, axis=0)

                dA2 = np.dot(dZ3, self.W3)
                dZ2 = dA2 * (Z2 > 0)
                dW2 = np.dot(dZ2.T, A1)
                dB2 = np.sum(dZ2, axis=0)

                dA1 = np.dot(dZ2, self.W2)
                dZ1 = dA1 * (Z1 > 0)
                dW1 = np.dot(dZ1.T, X_in)
                dB1 = np.sum(dZ1, axis=0)

                grads = [dW1, dB1, dW2, dB2, dW3, dB3]

                # Adam update
                t += 1
                for idx, (p, g) in enumerate(zip(params, grads)):
                    m[idx] = beta1 * m[idx] + (1 - beta1) * g
                    v[idx] = beta2 * v[idx] + (1 - beta2) * (g ** 2)
                    m_hat = m[idx] / (1 - beta1 ** t)
                    v_hat = v[idx] / (1 - beta2 ** t)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if epoch % 20 == 0 or epoch == epochs:
                val_probs, _ = self.forward(X_val_norm)
                val_preds = np.argmax(val_probs, axis=1)
                val_acc = np.mean(val_preds == y_val) * 100.0

                train_probs, _ = self.forward(X_train_norm)
                train_preds = np.argmax(train_probs, axis=1)
                train_acc = np.mean(train_preds == y_train) * 100.0

                print(f"Epoch {epoch:3d}/{epochs} - Train Acc: {train_acc:5.1f}% | Val Acc: {val_acc:5.1f}%")

        print("Training complete!")


def generate_synthetic_audio_dataset(num_samples_per_class=60):
    """
    Generates realistic acoustic patterns for Baby Cry, Speech, and Noise
    if no external wav dataset is provided yet.
    """
    print(f"Generating synthetic acoustic dataset ({num_samples_per_class} samples/class)...")
    data = []
    labels = []
    parity = AudioDSPParity()

    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    # 1. Class 0: Noise / Ambient / Silence
    for _ in range(num_samples_per_class):
        # Pink/White noise + occasional low-frequency room hum
        noise_level = np.random.uniform(0.01, 0.15)
        audio = np.random.normal(0, noise_level, len(t)).astype(np.float32)
        # Add 50/60 Hz hum
        hum_freq = np.random.choice([50.0, 60.0, 100.0, 120.0])
        audio += (0.05 * np.sin(2 * np.pi * hum_freq * t)).astype(np.float32)

        feats = parity.extract_features_from_audio(audio)
        for f in feats:
            data.append(f)
            labels.append(0)

    # 2. Class 1: Adult Speech
    for _ in range(num_samples_per_class):
        # Adult pitch: 100 - 220 Hz + speech formant filters
        f0 = np.random.uniform(90.0, 220.0)
        # Formants at 500, 1500, 2500 Hz
        speech = np.sin(2 * np.pi * f0 * t)
        speech += 0.6 * np.sin(2 * np.pi * (2 * f0) * t)
        speech += 0.3 * np.sin(2 * np.pi * (3 * f0) * t)
        # Syllabic amplitude envelope (3 - 5 Hz modulation)
        env = 0.5 * (1.0 + np.sin(2 * np.pi * np.random.uniform(3.0, 5.0) * t))
        audio = (speech * env * np.random.uniform(0.2, 0.6)).astype(np.float32)
        audio += np.random.normal(0, 0.02, len(t)).astype(np.float32)

        feats = parity.extract_features_from_audio(audio)
        for f in feats:
            data.append(f)
            labels.append(1)

    # 3. Class 2: Baby Cry
    for _ in range(num_samples_per_class):
        # Infant pitch: 400 - 650 Hz with pitch sweep (vibrato / wail)
        f_center = np.random.uniform(420.0, 580.0)
        pitch_sweep = f_center + 60.0 * np.sin(2 * np.pi * np.random.uniform(1.5, 3.5) * t)
        phase = 2 * np.pi * np.cumsum(pitch_sweep) / SAMPLE_RATE

        cry = np.sin(phase)
        # Strong infant cry harmonics (2nd, 3rd, 4th harmonics in 1000 - 3000 Hz)
        cry += 0.7 * np.sin(2 * phase)
        cry += 0.5 * np.sin(3 * phase)
        cry += 0.3 * np.sin(4 * phase)

        # Cry inspiratory/expiratory rhythm (bursts every 0.8 - 1.4s)
        burst_freq = np.random.uniform(0.7, 1.2)
        env = np.clip(np.sin(2 * np.pi * burst_freq * t), 0, 1) ** 1.5
        audio = (cry * env * np.random.uniform(0.35, 0.8)).astype(np.float32)
        audio += np.random.normal(0, 0.02, len(t)).astype(np.float32)

        feats = parity.extract_features_from_audio(audio)
        for f in feats:
            data.append(f)
            labels.append(2)

    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_wav_file(path: str):
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        if n_channels > 1:
            data = data[0::n_channels]

        # Resample if needed
        if framerate != SAMPLE_RATE:
            num_target_samples = int(len(data) * SAMPLE_RATE / framerate)
            data = np.interp(
                np.linspace(0, len(data), num_target_samples, endpoint=False),
                np.arange(len(data)),
                data
            ).astype(np.float32)

        return data


def load_dataset_from_directory(dataset_dir: str):
    parity = AudioDSPParity()
    data, labels = [], []

    class_mapping = {
        "noise": 0, "silence": 0, "background": 0,
        "speech": 1, "voice": 1,
        "cry": 2, "baby_cry": 2, "crying": 2
    }

    found_files = 0
    for subdir, class_id in class_mapping.items():
        folder = os.path.join(dataset_dir, subdir)
        if not os.path.exists(folder):
            continue
        for wav_path in glob.glob(os.path.join(folder, "*.wav")):
            try:
                audio = load_wav_file(wav_path)
                feats = parity.extract_features_from_audio(audio)
                for f in feats:
                    data.append(f)
                    labels.append(class_id)
                found_files += 1
            except Exception as e:
                print(f"Skipping {wav_path}: {e}")

    if found_files == 0:
        return None, None

    print(f"Loaded {found_files} WAV files from '{dataset_dir}'. Total feature windows: {len(data)}")
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)


def export_c_header(model: SimpleMLP, export_path: str):
    """Exports trained MLP weights and normalization scales to Core/Inc/model_weights.h"""
    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)

    inv_std = 1.0 / (model.input_std + 1e-6)

    def format_1d(arr, name):
        lines = [f"static const float32_t {name}[{len(arr)}] = {{"]
        chunk = []
        for val in arr:
            chunk.append(f"{val:+.7e}f")
            if len(chunk) >= 4:
                lines.append("    " + ", ".join(chunk) + ",")
                chunk = []
        if chunk:
            lines.append("    " + ", ".join(chunk) + ",")
        lines.append("};\n")
        return "\n".join(lines)

    def format_2d(mat, name):
        rows, cols = mat.shape
        lines = [f"static const float32_t {name}[{rows}][{cols}] = {{"]
        for r in range(rows):
            lines.append("    {")
            chunk = []
            for c in range(cols):
                chunk.append(f"{mat[r, c]:+.7e}f")
                if len(chunk) >= 4:
                    lines.append("        " + ", ".join(chunk) + ",")
                    chunk = []
            if chunk:
                lines.append("        " + ", ".join(chunk) + ",")
            lines.append("    },")
        lines.append("};\n")
        return "\n".join(lines)

    content = f"""/**
  ******************************************************************************
  * @file           : model_weights.h
  * @brief          : Auto-generated Float32 neural network weights and scaler
  *                   for STM32F429 Baby Cry Detection (TinyML).
  *                   Architecture: {INPUT_DIM} -> {HIDDEN1_DIM} -> {HIDDEN2_DIM} -> {OUTPUT_DIM}
  ******************************************************************************
  */

#ifndef __MODEL_WEIGHTS_H
#define __MODEL_WEIGHTS_H

#include "arm_math.h"

#define NN_INPUT_DIM      {INPUT_DIM}
#define NN_HIDDEN1_DIM    {HIDDEN1_DIM}
#define NN_HIDDEN2_DIM    {HIDDEN2_DIM}
#define NN_OUTPUT_DIM     {OUTPUT_DIM}

/* Feature Scaler: x_norm = (x - NN_INPUT_MEAN) * NN_INPUT_INV_STD */
{format_1d(model.input_mean, "NN_INPUT_MEAN")}
{format_1d(inv_std, "NN_INPUT_INV_STD")}

/* Layer 1: Dense ({HIDDEN1_DIM} x {INPUT_DIM}) + Bias + ReLU */
{format_2d(model.W1, "NN_W1")}
{format_1d(model.B1, "NN_B1")}

/* Layer 2: Dense ({HIDDEN2_DIM} x {HIDDEN1_DIM}) + Bias + ReLU */
{format_2d(model.W2, "NN_W2")}
{format_1d(model.B2, "NN_B2")}

/* Layer 3: Dense ({OUTPUT_DIM} x {HIDDEN2_DIM}) + Bias + Softmax */
{format_2d(model.W3, "NN_W3")}
{format_1d(model.B3, "NN_B3")}

#endif /* __MODEL_WEIGHTS_H */
"""

    with open(export_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully exported C weights header to: {export_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TinyML MLP for Baby Cry Detection and export to STM32")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Directory containing audio subfolders")
    parser.add_argument("--export-path", type=str,
                        default=os.path.join(current_dir, "..", "Core", "Inc", "model_weights.h"),
                        help="Target header file path")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    args = parser.parse_args()

    # Step 1: Load or Generate Dataset
    X, y = None, None
    if os.path.exists(args.dataset_dir):
        X, y = load_dataset_from_directory(args.dataset_dir)

    if X is None or len(X) == 0:
        print(f"No WAV dataset found in '{args.dataset_dir}'.")
        X, y = generate_synthetic_audio_dataset(num_samples_per_class=75)

    # Train / Val split (80 / 20)
    perm = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, y_train = X[perm[:split]], y[perm[:split]]
    X_val, y_val = X[perm[split:]], y[perm[split:]]

    # Step 2: Train MLP
    mlp = SimpleMLP()
    mlp.train(X_train, y_train, X_val, y_val, epochs=args.epochs, lr=args.lr)

    # Step 3: Export to C header
    export_c_header(mlp, args.export_path)


if __name__ == "__main__":
    main()
