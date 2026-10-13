/**
  ******************************************************************************
  * @file           : audio_dsp.c
  * @brief          : Реалізація алгоритмічного стеку (DSP & Edge AI)
  *                   для виділення спектральних ознак (FFT, MFCC) та
  *                   класифікації звуків (дитячий плач, голос, шум).
  ******************************************************************************
  */

#include "audio_dsp.h"
#include "audio_nn.h"
#include <string.h>
#include <math.h>

/* --- ВНУТРІШНІ ЗМІННІ ТА БУФЕРИ ДЛЯ ОБРОБКИ СИГНАЛУ --- */
static arm_rfft_fast_instance_f32 rfft_instance;

// Буфер для накопичення вхідних PCM відліків (кільцевий/накопичувальний)
static float32_t pcm_acc_buffer[AUDIO_FFT_SIZE];
static uint16_t  pcm_acc_index = 0;

// Робочі буфери для FFT (розмір 512)
static float32_t hann_window[AUDIO_FFT_SIZE];
static float32_t windowed_signal[AUDIO_FFT_SIZE];
static float32_t rfft_output[AUDIO_FFT_SIZE];
static float32_t magnitudes[AUDIO_FFT_HALF_SIZE]; // 256 амплітудних бінів

// Матриця мел-фільтрів (Mel Filterbank) та DCT
typedef struct {
    uint16_t start_bin;
    uint16_t center_bin;
    uint16_t stop_bin;
} MelFilter_t;

static MelFilter_t mel_filters[AUDIO_NUM_MEL_BANKS];
static float32_t dct_matrix[AUDIO_NUM_MFCC][AUDIO_NUM_MEL_BANKS];

// Ковзний часовий буфер для розрахунку 28 спектро-часових ознак TinyML
static float32_t mfcc_history[AUDIO_TEMPORAL_WINDOW_FRAMES][AUDIO_NUM_MFCC];
static float32_t cry_ratio_history[AUDIO_TEMPORAL_WINDOW_FRAMES];
static float32_t total_energy_history[AUDIO_TEMPORAL_WINDOW_FRAMES];
static uint16_t  history_head = 0;
static uint16_t  history_count = 0;

/* Допоміжні функції для розрахунку мел-шкали */
static inline float32_t HzToMel(float32_t hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float32_t MelToHz(float32_t mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/**
 * @brief Ініціалізація алгоритмічного конвеєра:
 *        1) CMSIS-DSP RFFT
 *        2) Вікно Ханна (зменшення розтікання спектру)
 *        3) Банк трикутних мел-фільтрів (100 Гц .. 7500 Гц)
 *        4) Матриця Дискретного Косинусного Перетворення (DCT-II)
 */
uint8_t AudioDSP_Init(void) {
    // 1. Ініціалізація RFFT з бібліотеки CMSIS-DSP
    if (arm_rfft_fast_init_f32(&rfft_instance, AUDIO_FFT_SIZE) != ARM_MATH_SUCCESS) {
        return 1; // Помилка ініціалізації
    }

    // 2. Попередній розрахунок вікна Ханна: w[n] = 0.5 * (1 - cos(2*PI*n / (N-1)))
    for (uint16_t i = 0; i < AUDIO_FFT_SIZE; i++) {
        hann_window[i] = 0.5f * (1.0f - arm_cos_f32(2.0f * PI * (float32_t)i / (float32_t)(AUDIO_FFT_SIZE - 1)));
    }

    // 3. Побудова банку 20 трикутних мел-фільтрів у діапазоні 100 Гц .. 7500 Гц
    float32_t min_mel = HzToMel(100.0f);
    float32_t max_mel = HzToMel(7500.0f);
    float32_t mel_step = (max_mel - min_mel) / (float32_t)(AUDIO_NUM_MEL_BANKS + 1);

    for (uint8_t m = 0; m < AUDIO_NUM_MEL_BANKS; m++) {
        float32_t left_mel   = min_mel + (float32_t)m * mel_step;
        float32_t center_mel = left_mel + mel_step;
        float32_t right_mel  = center_mel + mel_step;

        float32_t left_hz   = MelToHz(left_mel);
        float32_t center_hz = MelToHz(center_mel);
        float32_t right_hz  = MelToHz(right_mel);

        // Переводимо частоту в Гц в індекс спектрального біну FFT
        mel_filters[m].start_bin  = (uint16_t)((left_hz * AUDIO_FFT_SIZE) / AUDIO_SAMPLE_RATE);
        mel_filters[m].center_bin = (uint16_t)((center_hz * AUDIO_FFT_SIZE) / AUDIO_SAMPLE_RATE);
        mel_filters[m].stop_bin   = (uint16_t)((right_hz * AUDIO_FFT_SIZE) / AUDIO_SAMPLE_RATE);

        if (mel_filters[m].center_bin <= mel_filters[m].start_bin) {
            mel_filters[m].center_bin = mel_filters[m].start_bin + 1;
        }
        if (mel_filters[m].stop_bin <= mel_filters[m].center_bin) {
            mel_filters[m].stop_bin = mel_filters[m].center_bin + 1;
        }
        if (mel_filters[m].stop_bin >= AUDIO_FFT_HALF_SIZE) {
            mel_filters[m].stop_bin = AUDIO_FFT_HALF_SIZE - 1;
        }
    }

    // 4. Попередній розрахунок базису DCT-II для швидкого вилучення 13 MFCC коефіцієнтів
    for (uint8_t k = 0; k < AUDIO_NUM_MFCC; k++) {
        for (uint8_t n = 0; n < AUDIO_NUM_MEL_BANKS; n++) {
            dct_matrix[k][n] = arm_cos_f32(PI * (float32_t)k * ((float32_t)n + 0.5f) / (float32_t)AUDIO_NUM_MEL_BANKS);
        }
    }

    // Очищення буфера накопичення та часової історії
    memset(pcm_acc_buffer, 0, sizeof(pcm_acc_buffer));
    memset(mfcc_history, 0, sizeof(mfcc_history));
    memset(cry_ratio_history, 0, sizeof(cry_ratio_history));
    memset(total_energy_history, 0, sizeof(total_energy_history));
    pcm_acc_index = 0;
    history_head = 0;
    history_count = 0;

    return 0; // Успішно
}

/**
 * @brief Додавання PCM-семплів у буфер аналізу.
 *        16-бітні цілі числа конвертуються в нормалізовані float [-1.0 .. 1.0].
 */
bool AudioDSP_FeedSamples(const int16_t *samples, uint16_t count) {
    bool frame_ready = false;

    for (uint16_t i = 0; i < count; i++) {
        // Нормалізація int16 (-32768 .. 32767) у діапазон [-1.0f .. 1.0f]
        pcm_acc_buffer[pcm_acc_index++] = (float32_t)samples[i] / 32768.0f;

        // Коли буфер заповнено до AUDIO_FFT_SIZE (512 відліків = 32 мс)
        if (pcm_acc_index >= AUDIO_FFT_SIZE) {
            pcm_acc_index = 0; // Для 50% перекриття (overlap) можна робити зсув memmove
            frame_ready = true;
        }
    }

    return frame_ready;
}

/**
 * @brief Повний конвеєр спектрального аналізу та вилучення ознак:
 *        1) Накладання вікна Ханна
 *        2) CMSIS-DSP arm_rfft_fast_f32 (Дійсне ШПФ)
 *        3) Розрахунок модуля спектральних ліній (Magnitudes)
 *        4) Пошук домінантної частоти та оцінка смуги плачу (Cry Band 300-3000 Гц)
 *        5) Розрахунок 13 кепстральних коефіцієнтів MFCC
 */
void AudioDSP_ProcessFrame(AudioFeatures_t *out_features) {
    if (!out_features) return;

    // КРОК 1: Накладання віконної функції Ханна для запобігання витоку спектру
    for (uint16_t i = 0; i < AUDIO_FFT_SIZE; i++) {
        windowed_signal[i] = pcm_acc_buffer[i] * hann_window[i];
    }

    // КРОК 2: Виконання швидкого перетворення Фур'є (FFT) за допомогою CMSIS-DSP
    arm_rfft_fast_f32(&rfft_instance, windowed_signal, rfft_output, 0);

    // КРОК 3: Обчислення амплітудного спектру |X[k]|
    // Нульовий бін (DC) і бін Найквіста в rfft_fast упаковані в rfft_output[0] та [1]
    magnitudes[0] = fabsf(rfft_output[0]);
    // Комплексні числа: парний індекс = Real, непарний = Imag
    arm_cmplx_mag_f32(&rfft_output[2], &magnitudes[1], (AUDIO_FFT_HALF_SIZE - 1));

    // КРОК 4: Акустичний спектральний аналіз (Виявлення сигнатури плачу)
    // 1 біна = 16000 / 512 = 31.25 Гц
    const float32_t bin_resolution = (float32_t)AUDIO_SAMPLE_RATE / (float32_t)AUDIO_FFT_SIZE;
    uint16_t bin_cry_start = (uint16_t)(CRY_FREQ_MIN_HZ / bin_resolution); // ~300 Гц (бін 9)
    uint16_t bin_cry_stop  = (uint16_t)(CRY_FREQ_MAX_HZ / bin_resolution); // ~3000 Гц (бін 96)
    uint16_t bin_dc_cut    = (uint16_t)(80.0f / bin_resolution);           // Ігноруємо постійний шум < 80 Гц

    float32_t total_energy = 0.0f;
    float32_t cry_energy = 0.0f;
    float32_t max_mag = 0.0f;
    uint16_t  max_bin = 0;

    for (uint16_t i = bin_dc_cut; i < AUDIO_FFT_HALF_SIZE; i++) {
        float32_t m = magnitudes[i];
        total_energy += m;

        // Перевіряємо смугу дитячого плачу (300 .. 3000 Гц)
        if (i >= bin_cry_start && i <= bin_cry_stop) {
            cry_energy += m;
            if (m > max_mag) {
                max_mag = m;
                max_bin = i;
            }
        }
    }

    out_features->total_energy = total_energy;
    out_features->cry_band_energy = cry_energy;
    out_features->dominant_magnitude = max_mag;
    out_features->dominant_frequency = (float32_t)max_bin * bin_resolution;
    out_features->cry_energy_ratio = (total_energy > 0.0001f) ? (cry_energy / total_energy) : 0.0f;

    // КРОК 5: Розрахунок MFCC (Mel-frequency cepstral coefficients)
    float32_t mel_energies[AUDIO_NUM_MEL_BANKS];

    for (uint8_t m = 0; m < AUDIO_NUM_MEL_BANKS; m++) {
        float32_t energy = 0.0f;
        uint16_t start  = mel_filters[m].start_bin;
        uint16_t center = mel_filters[m].center_bin;
        uint16_t stop   = mel_filters[m].stop_bin;

        // Ліва сторона трикутного мел-фільтра
        for (uint16_t k = start; k < center; k++) {
            float32_t weight = (float32_t)(k - start) / (float32_t)(center - start);
            energy += magnitudes[k] * weight;
        }
        // Права сторона трикутного мел-фільтра
        for (uint16_t k = center; k <= stop; k++) {
            float32_t weight = (float32_t)(stop - k) / (float32_t)(stop - center);
            energy += magnitudes[k] * weight;
        }

        // Логарифмування енергії для психоакустичної компресії
        mel_energies[m] = logf(energy + 1e-5f);
    }

    // Застосування DCT (Дискретного Косинусного Перетворення) для декорреляції
    for (uint8_t k = 0; k < AUDIO_NUM_MFCC; k++) {
        float32_t sum = 0.0f;
        for (uint8_t n = 0; n < AUDIO_NUM_MEL_BANKS; n++) {
            sum += mel_energies[n] * dct_matrix[k][n];
        }
        out_features->mfcc[k] = sum;
    }

    // КРОК 6: Оновлення часового ковзного буфера (Rolling Temporal Buffer)
    memcpy(mfcc_history[history_head], out_features->mfcc, sizeof(out_features->mfcc));
    cry_ratio_history[history_head] = out_features->cry_energy_ratio;
    total_energy_history[history_head] = out_features->total_energy;

    history_head = (history_head + 1) % AUDIO_TEMPORAL_WINDOW_FRAMES;
    if (history_count < AUDIO_TEMPORAL_WINDOW_FRAMES) {
        history_count++;
    }

    // КРОК 7: Розрахунок 28 спектро-часових ознак (Mean + Std Dev + Energy)
    float32_t inv_N = 1.0f / (float32_t)history_count;

    // 13 середніх значень MFCC
    for (uint8_t k = 0; k < AUDIO_NUM_MFCC; k++) {
        float32_t sum = 0.0f;
        for (uint16_t f = 0; f < history_count; f++) {
            sum += mfcc_history[f][k];
        }
        out_features->temporal_features[k] = sum * inv_N;
    }

    // 13 стандартних відхилень MFCC
    for (uint8_t k = 0; k < AUDIO_NUM_MFCC; k++) {
        float32_t mean = out_features->temporal_features[k];
        float32_t var_sum = 0.0f;
        for (uint16_t f = 0; f < history_count; f++) {
            float32_t diff = mfcc_history[f][k] - mean;
            var_sum += diff * diff;
        }
        out_features->temporal_features[AUDIO_NUM_MFCC + k] = sqrtf(var_sum * inv_N);
    }

    // 2 узагальнені енергетичні ознаки
    float32_t sum_ratio = 0.0f;
    float32_t sum_energy = 0.0f;
    for (uint16_t f = 0; f < history_count; f++) {
        sum_ratio += cry_ratio_history[f];
        sum_energy += total_energy_history[f];
    }
    out_features->temporal_features[26] = sum_ratio * inv_N;
    out_features->temporal_features[27] = sum_energy * inv_N;
}

/**
 * @brief Класифікатор Edge AI (TinyML Neural Network):
 *        Виконує прямий прохід нейронної мережі Float32 на апаратному FPU Cortex-M4.
 *        Класифікує вектор із 28 ознак:
 *        - SOUND_CLASS_NOISE
 *        - SOUND_CLASS_SPEECH
 *        - SOUND_CLASS_BABY_CRY
 */
void AudioDSP_Classify(const AudioFeatures_t *features, AudioPrediction_t *out_pred) {
    if (!features || !out_pred) return;

    // Поріг тиші: якщо загальна амплітуда занадто мала, це фоновий шум/тиша
    if (features->total_energy < 0.10f && features->dominant_magnitude < 0.02f) {
        out_pred->prob_noise    = 0.98f;
        out_pred->prob_speech   = 0.01f;
        out_pred->prob_baby_cry = 0.01f;
        out_pred->predicted_class = SOUND_CLASS_NOISE;
        return;
    }

    // Прямий прохід TinyML моделі
    AudioNN_Infer(features->temporal_features, out_pred);
}
