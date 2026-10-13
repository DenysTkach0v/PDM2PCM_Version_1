#!/usr/bin/env python3
"""
DSP Parity Engine
=================
Exact 1:1 mathematical replica of the embedded STM32 C DSP pipeline (audio_dsp.c).
Used to extract features for training datasets ensuring 100% numerical parity.
"""

import numpy as np

SAMPLE_RATE = 16000
FFT_SIZE = 512
HALF_FFT = FFT_SIZE // 2  # 256
NUM_MEL_BANKS = 20
NUM_MFCC = 13
CRY_MIN_HZ = 300.0
CRY_MAX_HZ = 3000.0


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class AudioDSPParity:
    def __init__(self):
        # 1. Hann Window matching CMSIS-DSP / audio_dsp.c
        n = np.arange(FFT_SIZE, dtype=np.float32)
        self.hann_window = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (FFT_SIZE - 1)))

        # 2. Mel Filterbank setup (100 Hz to 7500 Hz)
        min_mel = hz_to_mel(100.0)
        max_mel = hz_to_mel(7500.0)
        mel_step = (max_mel - min_mel) / (NUM_MEL_BANKS + 1)

        self.mel_filters = []
        for m in range(NUM_MEL_BANKS):
            left_mel = min_mel + m * mel_step
            center_mel = left_mel + mel_step
            right_mel = center_mel + mel_step

            left_hz = mel_to_hz(left_mel)
            center_hz = mel_to_hz(center_mel)
            right_hz = mel_to_hz(right_mel)

            start_bin = int((left_hz * FFT_SIZE) / SAMPLE_RATE)
            center_bin = int((center_hz * FFT_SIZE) / SAMPLE_RATE)
            stop_bin = int((right_hz * FFT_SIZE) / SAMPLE_RATE)

            if center_bin <= start_bin:
                center_bin = start_bin + 1
            if stop_bin <= center_bin:
                stop_bin = center_bin + 1
            if stop_bin >= HALF_FFT:
                stop_bin = HALF_FFT - 1

            self.mel_filters.append((start_bin, center_bin, stop_bin))

        # 3. DCT-II Basis Matrix
        self.dct_matrix = np.zeros((NUM_MFCC, NUM_MEL_BANKS), dtype=np.float32)
        for k in range(NUM_MFCC):
            for n_idx in range(NUM_MEL_BANKS):
                self.dct_matrix[k, n_idx] = np.cos(np.pi * k * (n_idx + 0.5) / NUM_MEL_BANKS)

        self.bin_resolution = SAMPLE_RATE / FFT_SIZE
        self.bin_cry_start = int(CRY_MIN_HZ / self.bin_resolution)
        self.bin_cry_stop = int(CRY_MAX_HZ / self.bin_resolution)
        self.bin_dc_cut = int(80.0 / self.bin_resolution)

    def process_frame(self, frame_samples: np.ndarray):
        """
        Process single 512-sample frame [-1.0 .. 1.0].
        Returns:
            mfcc: array of 13 floats
            dominant_freq: float (Hz)
            dominant_mag: float
            total_energy: float
            cry_band_energy: float
            cry_energy_ratio: float
        """
        if len(frame_samples) != FFT_SIZE:
            raise ValueError(f"Frame must have {FFT_SIZE} samples, got {len(frame_samples)}")

        # Step 1: Hann Window
        windowed = frame_samples * self.hann_window

        # Step 2: Real FFT
        fft_complex = np.fft.rfft(windowed, n=FFT_SIZE)
        magnitudes = np.abs(fft_complex).astype(np.float32)

        # Step 3: Spectral metrics
        total_energy = float(np.sum(magnitudes[self.bin_dc_cut:HALF_FFT]))
        cry_band_slice = magnitudes[self.bin_cry_start:self.bin_cry_stop + 1]
        cry_band_energy = float(np.sum(cry_band_slice))

        if len(cry_band_slice) > 0:
            max_idx = np.argmax(cry_band_slice)
            max_mag = float(cry_band_slice[max_idx])
            dominant_freq = float((self.bin_cry_start + max_idx) * self.bin_resolution)
        else:
            max_mag = 0.0
            dominant_freq = 0.0

        cry_energy_ratio = (cry_band_energy / total_energy) if total_energy > 1e-4 else 0.0

        # Step 4: Mel Filterbank
        mel_energies = np.zeros(NUM_MEL_BANKS, dtype=np.float32)
        for m, (start, center, stop) in enumerate(self.mel_filters):
            energy = 0.0
            # Left slope
            if center > start:
                weights_left = (np.arange(start, center) - start) / (center - start)
                energy += np.sum(magnitudes[start:center] * weights_left)
            # Right slope
            if stop > center:
                weights_right = (stop - np.arange(center, stop + 1)) / (stop - center)
                energy += np.sum(magnitudes[center:stop + 1] * weights_right)

            mel_energies[m] = np.log(energy + 1e-5)

        # Step 5: DCT-II for 13 MFCCs
        mfcc = np.dot(self.dct_matrix, mel_energies)

        return mfcc, dominant_freq, max_mag, total_energy, cry_band_energy, cry_energy_ratio

    def extract_features_from_audio(self, audio: np.ndarray, hop_size: int = 256, window_frames: int = 30):
        """
        Extract rolling temporal statistics (28 features) from audio sequence.
        hop_size = 256 samples (16 ms step = 50% overlap).
        window_frames = 30 frames (~0.5 - 1.0s context).
        Returns:
            features: 2D array of shape [num_windows, 28]
        """
        # Ensure float32 normalized [-1.0 .. 1.0]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif np.max(np.abs(audio)) > 1.0:
            audio = audio.astype(np.float32) / np.max(np.abs(audio))

        num_samples = len(audio)
        if num_samples < FFT_SIZE:
            return np.empty((0, 28), dtype=np.float32)

        # Frame extraction
        frames_mfcc = []
        frames_cry_ratio = []
        frames_total_energy = []

        for start_idx in range(0, num_samples - FFT_SIZE + 1, hop_size):
            frame = audio[start_idx:start_idx + FFT_SIZE]
            mfcc, _, _, total_e, _, cry_ratio = self.process_frame(frame)
            frames_mfcc.append(mfcc)
            frames_cry_ratio.append(cry_ratio)
            frames_total_energy.append(total_e)

        frames_mfcc = np.array(frames_mfcc, dtype=np.float32)  # [T, 13]
        frames_cry_ratio = np.array(frames_cry_ratio, dtype=np.float32)
        frames_total_energy = np.array(frames_total_energy, dtype=np.float32)

        num_frames = len(frames_mfcc)
        if num_frames < window_frames:
            # Pad or take global stats
            window_frames = max(1, num_frames)

        feature_windows = []
        step = max(1, window_frames // 2)

        for w_start in range(0, num_frames - window_frames + 1, step):
            w_mfcc = frames_mfcc[w_start:w_start + window_frames]  # [W, 13]
            w_ratio = frames_cry_ratio[w_start:w_start + window_frames]
            w_energy = frames_total_energy[w_start:w_start + window_frames]

            mean_mfcc = np.mean(w_mfcc, axis=0)  # 13
            std_mfcc = np.std(w_mfcc, axis=0)    # 13
            mean_ratio = np.mean(w_ratio)        # 1
            mean_energy = np.mean(w_energy)      # 1

            # 28 features total
            vec = np.concatenate([mean_mfcc, std_mfcc, [mean_ratio, mean_energy]]).astype(np.float32)
            feature_windows.append(vec)

        if not feature_windows:
            # Fallback for short clips
            mean_mfcc = np.mean(frames_mfcc, axis=0)
            std_mfcc = np.std(frames_mfcc, axis=0)
            vec = np.concatenate([mean_mfcc, std_mfcc, [np.mean(frames_cry_ratio), np.mean(frames_total_energy)]])
            feature_windows.append(vec.astype(np.float32))

        return np.array(feature_windows, dtype=np.float32)
