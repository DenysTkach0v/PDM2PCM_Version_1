#!/usr/bin/env python3
"""
STM32 UART Audio Stream Recorder
================================
Captures raw 16-bit 16 kHz PCM audio streamed from STM32F429 over UART DMA
(packets starting with 0xAA 0xBB followed by 32 bytes of int16 PCM) and saves to .wav.

Usage:
    python uart_recorder.py --port COM3 --baud 115200 --out recording.wav --duration 10
"""

import argparse
import sys
import time
import wave
import struct

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None


def find_serial_ports():
    if serial is None:
        return []
    return [port.device for port in list_ports.comports()]


def record_uart_audio(port: str, baudrate: int, output_wav: str, duration_sec: float):
    if serial is None:
        print("Error: 'pyserial' is not installed. Run: pip install pyserial")
        sys.exit(1)

    print(f"Connecting to {port} at {baudrate} baud...")
    try:
        ser = serial.Serial(port, baudrate, timeout=1.0)
    except Exception as e:
        print(f"Failed to open serial port {port}: {e}")
        available = find_serial_ports()
        print(f"Available ports: {available}")
        sys.exit(1)

    # Audio format: 16 kHz, 16-bit signed LE, mono
    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2
    PACKET_HEADER = bytes([0xAA, 0xBB])
    SAMPLES_PER_PACKET = 16
    PACKET_PAYLOAD_SIZE = SAMPLES_PER_PACKET * BYTES_PER_SAMPLE  # 32 bytes

    total_samples_needed = int(SAMPLE_RATE * duration_sec)
    collected_pcm_bytes = bytearray()

    print(f"Recording {duration_sec}s of audio to '{output_wav}' ({SAMPLE_RATE} Hz, 16-bit mono)...")
    print("Press Ctrl+C to stop recording early.\n")

    start_time = time.time()
    packet_count = 0
    sync_errors = 0

    try:
        ser.reset_input_buffer()
        while len(collected_pcm_bytes) < total_samples_needed * BYTES_PER_SAMPLE:
            # Sync to 0xAA 0xBB header
            b1 = ser.read(1)
            if not b1:
                continue
            if b1 == bytes([0xAA]):
                b2 = ser.read(1)
                if b2 == bytes([0xBB]):
                    # Valid header, read payload
                    payload = ser.read(PACKET_PAYLOAD_SIZE)
                    if len(payload) == PACKET_PAYLOAD_SIZE:
                        collected_pcm_bytes.extend(payload)
                        packet_count += 1
                        elapsed = time.time() - start_time
                        samples_rec = len(collected_pcm_bytes) // 2
                        sec_rec = samples_rec / SAMPLE_RATE
                        print(f"\rCaptured: {sec_rec:.2f}s / {duration_sec:.1f}s [{packet_count} packets]", end="")
                    else:
                        sync_errors += 1
                else:
                    sync_errors += 1

            if duration_sec > 0 and (time.time() - start_time) > (duration_sec + 3.0):
                print("\nRecording timed out.")
                break

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        ser.close()

    print(f"\nFinalizing WAV file: {len(collected_pcm_bytes) // 2} samples written.")

    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(collected_pcm_bytes)

    print(f"Saved successfully to '{output_wav}' (Sync errors: {sync_errors})")


def main():
    parser = argparse.ArgumentParser(description="Record PCM stream from STM32 UART DMA")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--out", type=str, default="stm32_mic_recording.wav", help="Output WAV filename")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")

    args = parser.parse_args()

    if args.list_ports:
        ports = find_serial_ports()
        print(f"Available serial ports: {ports}")
        return

    record_uart_audio(args.port, args.baud, args.out, args.duration)


if __name__ == "__main__":
    main()
