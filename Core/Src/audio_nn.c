/**
  ******************************************************************************
  * @file           : audio_nn.c
  * @brief          : TinyML Neural Network Forward Inference implementation
  *                   executed on Cortex-M4 hardware FPU.
  ******************************************************************************
  */

#include "audio_nn.h"
#include "model_weights.h"
#include <math.h>

void AudioNN_Infer(const float32_t *features, AudioPrediction_t *out_pred) {
    if (!features || !out_pred) return;

    // Buffer for standardized input features
    float32_t x_norm[NN_INPUT_DIM];
    for (uint8_t i = 0; i < NN_INPUT_DIM; i++) {
        x_norm[i] = (features[i] - NN_INPUT_MEAN[i]) * NN_INPUT_INV_STD[i];
    }

    // LAYER 1: Dense (28 -> 32) + ReLU
    float32_t h1[NN_HIDDEN1_DIM];
    for (uint8_t i = 0; i < NN_HIDDEN1_DIM; i++) {
        float32_t sum = NN_B1[i];
        for (uint8_t j = 0; j < NN_INPUT_DIM; j++) {
            sum += NN_W1[i][j] * x_norm[j];
        }
        // ReLU activation
        h1[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    // LAYER 2: Dense (32 -> 16) + ReLU
    float32_t h2[NN_HIDDEN2_DIM];
    for (uint8_t i = 0; i < NN_HIDDEN2_DIM; i++) {
        float32_t sum = NN_B2[i];
        for (uint8_t j = 0; j < NN_HIDDEN1_DIM; j++) {
            sum += NN_W2[i][j] * h1[j];
        }
        // ReLU activation
        h2[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    // LAYER 3: Dense (16 -> 3) (Logits)
    float32_t logits[NN_OUTPUT_DIM];
    for (uint8_t i = 0; i < NN_OUTPUT_DIM; i++) {
        float32_t sum = NN_B3[i];
        for (uint8_t j = 0; j < NN_HIDDEN2_DIM; j++) {
            sum += NN_W3[i][j] * h2[j];
        }
        logits[i] = sum;
    }

    // Numerically stable Softmax
    float32_t max_logit = logits[0];
    if (logits[1] > max_logit) max_logit = logits[1];
    if (logits[2] > max_logit) max_logit = logits[2];

    float32_t exp0 = expf(logits[0] - max_logit);
    float32_t exp1 = expf(logits[1] - max_logit);
    float32_t exp2 = expf(logits[2] - max_logit);
    float32_t sum_exp = exp0 + exp1 + exp2;

    if (sum_exp > 0.0f) {
        out_pred->prob_noise    = exp0 / sum_exp;
        out_pred->prob_speech   = exp1 / sum_exp;
        out_pred->prob_baby_cry = exp2 / sum_exp;
    } else {
        out_pred->prob_noise    = 0.333f;
        out_pred->prob_speech   = 0.333f;
        out_pred->prob_baby_cry = 0.334f;
    }

    // Final predicted class decision
    if (out_pred->prob_baby_cry >= out_pred->prob_speech &&
        out_pred->prob_baby_cry >= out_pred->prob_noise) {
        out_pred->predicted_class = SOUND_CLASS_BABY_CRY;
    } else if (out_pred->prob_speech >= out_pred->prob_noise) {
        out_pred->predicted_class = SOUND_CLASS_SPEECH;
    } else {
        out_pred->predicted_class = SOUND_CLASS_NOISE;
    }
}
