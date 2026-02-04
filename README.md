STM32 Audio Signal Processing Pipeline

This project implements a real-time audio processing system using an STM32 microcontroller and the MP34DT01 digital MEMS microphone. 
It covers the entire digital signal processing (DSP) chain, from raw PDM data acquisition to high-level feature extraction.


**Key Features**
Data Acquisition: Interfacing with the MP34DT01 microphone via PDM protocol.

PDM to PCM Conversion: Real-time decimation and filtering to obtain a standard audio format.

Frequency Analysis: Implementation of Fast Fourier Transform (FFT) to analyze the audio spectrum.

Feature Extraction: Computing Mel-frequency cepstral coefficients (MFCC) for potential machine learning applications.

Tech Stack & Components
Hardware:

MCU: STM32 Microcontroller.

Microphone: **MP34DT01** (Digital MEMS, PDM output).
