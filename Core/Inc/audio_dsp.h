/**
  ******************************************************************************
  * @file           : audio_dsp.h
  * @brief          : Заголовочний файл алгоритмічного стеку (DSP & Edge AI)
  *                   для аналізу аудіосигналу та виявлення дитячого плачу.
  *
  * АЛГОРИТМІЧНИЙ СТЕК:
  * 1. PDM -> PCM: Децимація та фільтрація (бібліотека ST PDM2PCM).
  * 2. Framing & Windowing: Накопичення вікна (512 відліків = 32 мс) + вікно Ханна.
  * 3. FFT (Fast Fourier Transform): CMSIS-DSP arm_rfft_fast_f32 для спектрального аналізу.
  * 4. Акустичний аналіз плачу: Оцінка спектральної енергії в Cry Band (300-3000 Гц)
  *    та пошук основного тону (Pitch F0 ~350-600 Гц).
  * 5. MFCC (Mel-frequency cepstral coefficients): Психоакустична фільтрація
  *    (Mel Filterbank) + DCT для формування вектора ознак для AI.
  * 6. TinyML / Edge AI: Модель класифікації звуку (Cry vs Speech vs Noise).
  ******************************************************************************
  */

#ifndef __AUDIO_DSP_H
#define __AUDIO_DSP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "arm_math.h"

/* --- КОНФІГУРАЦІЯ АУДІО ТА ДИСПЕТЧЕРА ОЗНАК --- */
#define AUDIO_SAMPLE_RATE     16000    // Частота дискретизації PCM (16 кГц)
#define AUDIO_FFT_SIZE        512      // Розмір вікна FFT (512 семплів = 32 мс при 16 кГц)
#define AUDIO_FFT_HALF_SIZE   (AUDIO_FFT_SIZE / 2) // 256 частотних бінів
#define AUDIO_NUM_MEL_BANKS   20       // Кількість трикутних мел-фільтрів
#define AUDIO_NUM_MFCC        13       // Кількість кепстральних коефіцієнтів MFCC

/* Діапазон частот, характерний для дитячого плачу */
#define CRY_FREQ_MIN_HZ       300.0f   // Нижня межа основного тону плачу
#define CRY_FREQ_MAX_HZ       3000.0f  // Верхня межа ключових гармонік плачу
#define CRY_F0_MIN_HZ         350.0f   // Типовий діапазон частоти основного тону (Pitch F0)
#define CRY_F0_MAX_HZ         750.0f

#define AUDIO_TEMPORAL_WINDOW_FRAMES 32 // Кількість фреймів у ковзному буфері (~1 с)
#define AUDIO_NUM_FEATURES           28 // 13 mean MFCC + 13 std MFCC + 1 cry ratio + 1 energy

/**
 * @brief Класи звуків для Edge AI моделі
 */
typedef enum {
    SOUND_CLASS_NOISE = 0,    // Побутовий шум, тиша, вентилятор
    SOUND_CLASS_SPEECH,       // Мова дорослої людини (низький тон < 300 Гц)
    SOUND_CLASS_BABY_CRY      // Дитячий плач (високий тон 400-600 Гц, сильні форманти)
} SoundClass_t;

/**
 * @brief Структура для збереження вилучених акустичних ознак
 */
typedef struct {
    float32_t dominant_frequency;                     // Домінантна (пікова) частота в Гц
    float32_t dominant_magnitude;                     // Амплітуда домінантної частоти
    float32_t total_energy;                           // Повна спектральна енергія фрейму
    float32_t cry_band_energy;                        // Енергія в діапазоні 300 - 3000 Гц
    float32_t cry_energy_ratio;                       // Частка енергії діапазону плачу (0.0 .. 1.0)
    float32_t mfcc[AUDIO_NUM_MFCC];                   // Вектор з 13 коефіцієнтів MFCC
    float32_t temporal_features[AUDIO_NUM_FEATURES];  // 28 спектро-часових ознак для TinyML
} AudioFeatures_t;

/**
 * @brief Результат класифікації звуку моделлю Edge AI
 */
typedef struct {
    float32_t prob_noise;             // Ймовірність шуму (0.0 .. 1.0)
    float32_t prob_speech;            // Ймовірність звичайної мови
    float32_t prob_baby_cry;          // Ймовірність плачу дитини
    SoundClass_t predicted_class;     // Фінальний передбачений клас
} AudioPrediction_t;

/* --- ФУНКЦІЇ АЛГОРИТМІЧНОГО СТЕКУ --- */

/**
 * @brief Ініціалізація внутрішніх структур FFT, віконної функції Ханна та банку мел-фільтрів
 * @retval 0 - успішно, 1 - помилка ініціалізації CMSIS-DSP
 */
uint8_t AudioDSP_Init(void);

/**
 * @brief Додавання нових PCM-семплів (наприклад, по 16 штук з PDM2PCM фільтра) у накопичувальний буфер
 * @param samples Вказівник на масив нових 16-бітних відліків PCM
 * @param count Кількість відліків (зазвичай 16)
 * @retval true якщо зібрано повний фрейм (512 відліків) і він готовий до аналізу
 */
bool AudioDSP_FeedSamples(const int16_t *samples, uint16_t count);

/**
 * @brief Повний розрахунок фрейму: вікно Ханна -> FFT -> Магнітуди -> MFCC -> Акустичні ознаки
 * @param out_features Вказівник на структуру для запису розрахованих ознак
 */
void AudioDSP_ProcessFrame(AudioFeatures_t *out_features);

/**
 * @brief Класифікація поточного фрейму звуку (TinyML / Акустичний класифікатор)
 * @param features Вказівник на поточні вилучені ознаки
 * @param out_pred Вказівник на структуру результату класифікації
 */
void AudioDSP_Classify(const AudioFeatures_t *features, AudioPrediction_t *out_pred);

#ifdef __cplusplus
}
#endif

#endif /* __AUDIO_DSP_H */
