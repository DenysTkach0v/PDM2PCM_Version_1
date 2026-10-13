/**
  ******************************************************************************
  * @file           : audio_nn.h
  * @brief          : TinyML Neural Network Inference Engine for Baby Cry Detection
  *                   using STM32F429 Cortex-M4 Hardware FPU.
  ******************************************************************************
  */

#ifndef __AUDIO_NN_H
#define __AUDIO_NN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "arm_math.h"
#include "audio_dsp.h"

#define AUDIO_NN_FEATURE_DIM  28

/**
 * @brief Run Float32 MLP forward inference with Cortex-M4 FPU.
 *        Architecture: 28 (Input) -> 32 (ReLU) -> 16 (ReLU) -> 3 (Softmax)
 * @param features Pointer to 28-element normalized feature vector:
 *                 [0..12]: Mean MFCCs
 *                 [13..25]: Std Dev MFCCs
 *                 [26]: Mean Cry Band Energy Ratio
 *                 [27]: Mean Total Energy
 * @param out_pred Pointer to prediction result structure
 */
void AudioNN_Infer(const float32_t *features, AudioPrediction_t *out_pred);

#ifdef __cplusplus
}
#endif

#endif /* __AUDIO_NN_H */
