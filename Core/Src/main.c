/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body - PDM Mic to FFT/Stream
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "pdm2pcm.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "arm_math.h"  // CMSIS-DSP Library
#include "audio_dsp.h" // Модуль спектрального аналізу та Edge AI
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/**
 * @brief Режими роботи пристрою:
 * MODE_STREAM_PCM : Передача 16-бітного звуку на ПК через UART DMA (пакети 0xAA 0xBB + 32 байти PCM).
 *                   Використовується для перевірки мікрофона, прослуховування та збору датасету плачу.
 * MODE_DSP_EDGE_AI: Повний бортовий аналіз на мікроконтролері.
 *                   Конвертація PDM -> PCM -> Вікно Ханна -> FFT -> MFCC -> TinyML класифікація.
 */
typedef enum {
    MODE_STREAM_PCM = 0, // Стрімінг звуку на комп'ютер
    MODE_DSP_EDGE_AI = 1 // Обробка та виявлення плачу на STM32
} OperatingMode_t;

/**
 * @brief Події подвійного буфера PDM для зв'язку між DMA перериваннями та головним циклом
 */
typedef enum {
    PDM_EVENT_NONE = 0,
    PDM_EVENT_HALF_READY, // Перша половина PDM буфера готова (байти 0..127)
    PDM_EVENT_FULL_READY  // Друга половина PDM буфера готова (байти 128..255)
} PDM_BufferEvent_t;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

// ==============================================================================
// ГОЛОВНЕ НАЛАШТУВАННЯ: ОБЕРІТЬ РЕЖИМ ТУТ
// ==============================================================================
#define CURRENT_MODE        MODE_DSP_EDGE_AI // або MODE_STREAM_PCM

// ------------------------------------------------------------------------------
// МАТЕМАТИКА БУФЕРІВ PDM -> PCM:
// 1. Тактова частота I2S = 1.024 МГц (32 кГц * 32 біти на стерео-фрейм).
// 2. Коефіцієнт децимації = 64 (PDM1_filter_config.decimation_factor).
// 3. Частота дискретизації PCM = 1.024 МГц / 64 = 16 000 Гц (16 кГц).
// 4. За одну половину DMA буфера приймається 128 байт = 1024 біти PDM.
// 5. 1024 біти PDM / 64 = 16 семплів PCM (16 відліків int16_t = 32 байти звуку).
// ------------------------------------------------------------------------------
#define PDM_BUF_SIZE        256 // Розмір DMA буфера PDM у байтах (128 слів по 16 біт)
#define PCM_BUF_SIZE        16  // Кількість вихідних семплів PCM за одну децимацію

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CRC_HandleTypeDef hcrc;

I2S_HandleTypeDef hi2s2;
DMA_HandleTypeDef hdma_spi2_rx;

UART_HandleTypeDef huart1;
DMA_HandleTypeDef hdma_usart1_tx;

/* USER CODE BEGIN PV */

// Буфери сирого звуку
uint8_t PDM_Buffer[PDM_BUF_SIZE];
int16_t PCM_Buffer[PCM_BUF_SIZE];

// Прапорець події від DMA I2S (переривання Half / Full Transfer)
volatile PDM_BufferEvent_t pdm_event = PDM_EVENT_NONE;

// Подвійний буфер UART TX для стрімінгу на ПК без блокування і спотворень
// Формат пакета: [0xAA] [0xBB] + [32 байти PCM] = 34 байти
static uint8_t uart_tx_dma_buf[2][34];
static uint8_t uart_tx_idx = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_CRC_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_I2S2_Init(void);
/* USER CODE BEGIN PFP */
static void StreamPCM_To_UART(const int16_t *pcm_data, uint16_t count);
static void PrintDSPStatus(const AudioFeatures_t *features, const AudioPrediction_t *pred);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/**
 * @brief Відправка 16-бітного PCM аудіо на ПК через UART DMA (non-blocking).
 *        Використовує подвійну буферизацію для усунення перешкод у звуці.
 */
static void StreamPCM_To_UART(const int16_t *pcm_data, uint16_t count) {
    if (huart1.gState == HAL_UART_STATE_READY) {
        // Заголовок синхронізації для ПК скрипта (0xAA, 0xBB)
        uart_tx_dma_buf[uart_tx_idx][0] = 0xAA;
        uart_tx_dma_buf[uart_tx_idx][1] = 0xBB;

        // Копіюємо 16 семплів int16 (32 байти)
        memcpy(&uart_tx_dma_buf[uart_tx_idx][2], pcm_data, count * sizeof(int16_t));

        // Відправляємо через DMA
        HAL_UART_Transmit_DMA(&huart1, uart_tx_dma_buf[uart_tx_idx], 34);

        // Перемикаємо буфер для наступного виклику
        uart_tx_idx ^= 1;
    }
}

/**
 * @brief Виведення результатів спектрального аналізу та Edge AI у термінал UART.
 */
static void PrintDSPStatus(const AudioFeatures_t *features, const AudioPrediction_t *pred) {
    static uint32_t last_print = 0;
    uint32_t now = HAL_GetTick();

    // Друкуємо кожні 250 мс, або негайно, якщо виявлено дитячий плач
    if ((now - last_print >= 250) || (pred->predicted_class == SOUND_CLASS_BABY_CRY)) {
        last_print = now;
        char msg[128];

        if (pred->predicted_class == SOUND_CLASS_BABY_CRY) {
            LED_RED_ON();
            LED_GREEN_OFF();
            snprintf(msg, sizeof(msg),
                     "\r\n[!!! BABY CRY DETECTED !!!] Pitch: %.1f Hz | CryBand: %.0f%% | CryProb: %.2f\r\n",
                     features->dominant_frequency,
                     features->cry_energy_ratio * 100.0f,
                     pred->prob_baby_cry);
        } else if (pred->predicted_class == SOUND_CLASS_SPEECH) {
            LED_RED_OFF();
            LED_GREEN_ON();
            snprintf(msg, sizeof(msg),
                     "[SPEECH] Pitch: %.1f Hz | CryBand: %.0f%% | SpeechProb: %.2f\r\n",
                     features->dominant_frequency,
                     features->cry_energy_ratio * 100.0f,
                     pred->prob_speech);
        } else {
            LED_RED_OFF();
            LED_GREEN_TOGGLE(); // Індикація фонового моніторингу (heartbeat)
            snprintf(msg, sizeof(msg),
                     "[NOISE/SILENCE] DomFreq: %.1f Hz | TotalEnergy: %.2f | NoiseProb: %.2f\r\n",
                     features->dominant_frequency,
                     features->total_energy,
                     pred->prob_noise);
        }

        HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 50);
    }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_CRC_Init();
  MX_USART1_UART_Init();
  MX_PDM2PCM_Init();
  MX_I2S2_Init();
  /* USER CODE BEGIN 2 */

  // 1. Ініціалізація алгоритмічного стеку DSP (CMSIS-DSP RFFT, Hann Window, Mel Filterbank, DCT)
  if (AudioDSP_Init() != 0) {
      char err[] = "Audio DSP Init FAILED!\r\n";
      HAL_UART_Transmit(&huart1, (uint8_t*)err, strlen(err), 100);
      Error_Handler();
  }

  // 2. Стартове інформаційне повідомлення
  char start_msg[256];
  snprintf(start_msg, sizeof(start_msg),
           "\r\n========================================\r\n"
           " STM32F429 Audio DSP & Edge AI Pipeline\r\n"
           " PDM Mic: MP34DT01 | Fs: 16000 Hz\r\n"
           " Active Mode: %s\r\n"
           "========================================\r\n\r\n",
           (CURRENT_MODE == MODE_DSP_EDGE_AI) ? "On-board DSP + AI (Cry Detector)" : "PCM Stream to PC");
  HAL_UART_Transmit(&huart1, (uint8_t*)start_msg, strlen(start_msg), 100);

  // 3. Очищення буферів перед стартом
  memset(PDM_Buffer, 0, sizeof(PDM_Buffer));
  memset(PCM_Buffer, 0, sizeof(PCM_Buffer));
  pdm_event = PDM_EVENT_NONE;

  // 4. Запуск постійного прийому PDM-потоку з мікрофона через I2S + DMA
  // Розмір вказується у 16-бітних словах, тому ділимо PDM_BUF_SIZE на 2
  if (HAL_I2S_Receive_DMA(&hi2s2, (uint16_t*)PDM_Buffer, PDM_BUF_SIZE / 2) != HAL_OK) {
      Error_Handler();
  }

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    // Перевіряємо, чи DMA заповнив половину (Ping) або повний буфер (Pong)
    if (pdm_event != PDM_EVENT_NONE)
    {
        // Визначаємо адресу поточної готової половини буфера PDM
        uint16_t *pdm_chunk = (pdm_event == PDM_EVENT_HALF_READY) ?
                              (uint16_t*)&PDM_Buffer[0] :
                              (uint16_t*)&PDM_Buffer[PDM_BUF_SIZE / 2];

        pdm_event = PDM_EVENT_NONE; // Скидаємо прапорець

        // КРОК 1: PDM -> PCM Конвертація (Децимація на 64 + цифрова фільтрація)
        // 1024 біти сирого PDM перетворюються на 16 якісних 16-бітних відліків PCM
        if (MX_PDM2PCM_Process(pdm_chunk, (uint16_t*)PCM_Buffer) == 0)
        {
            if (CURRENT_MODE == MODE_STREAM_PCM)
            {
                // Режим 1: Передача звуку на ПК через UART DMA (для запису датасету чи прослуховування)
                StreamPCM_To_UART(PCM_Buffer, PCM_BUF_SIZE);
            }
            else if (CURRENT_MODE == MODE_DSP_EDGE_AI)
            {
                // Режим 2: Бортовий аналіз на мікроконтролері
                // Додаємо нові 16 семплів у буфер накопичення вікна аналізу (512 відліків = 32 мс)
                if (AudioDSP_FeedSamples(PCM_Buffer, PCM_BUF_SIZE))
                {
                    AudioFeatures_t features;
                    AudioPrediction_t prediction;

                    // КРОК 2, 3, 4: Вікно Ханна -> ШПФ (FFT) -> MFCC (Мел-спектр) -> Вилучення ознак
                    AudioDSP_ProcessFrame(&features);

                    // КРОК 5: Класифікація TinyML / Edge AI (Дитячий плач vs Голос vs Шум)
                    AudioDSP_Classify(&features, &prediction);

                    // Відображення результатів у UART терміналі
                    PrintDSPStatus(&features, &prediction);
                }
            }
        }
    }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  __HAL_CRC_DR_RESET(&hcrc);
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

}

/**
  * @brief I2S2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2S2_Init(void)
{

  /* USER CODE BEGIN I2S2_Init 0 */

  /* USER CODE END I2S2_Init 0 */

  /* USER CODE BEGIN I2S2_Init 1 */

  /* USER CODE END I2S2_Init 1 */
  hi2s2.Instance = SPI2;
  hi2s2.Init.Mode = I2S_MODE_MASTER_RX;
  hi2s2.Init.Standard = I2S_STANDARD_LSB;
  hi2s2.Init.DataFormat = I2S_DATAFORMAT_16B;
  hi2s2.Init.MCLKOutput = I2S_MCLKOUTPUT_DISABLE;
  hi2s2.Init.AudioFreq = I2S_AUDIOFREQ_32K;
  hi2s2.Init.CPOL = I2S_CPOL_LOW;
  hi2s2.Init.ClockSource = I2S_CLOCK_PLL;
  hi2s2.Init.FullDuplexMode = I2S_FULLDUPLEXMODE_DISABLE;
  if (HAL_I2S_Init(&hi2s2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2S2_Init 2 */

  /* USER CODE END I2S2_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 921600;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream3_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream3_IRQn);
  /* DMA2_Stream7_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream7_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream7_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOG, LED_GREEN_PIN | LED_RED_PIN, GPIO_PIN_RESET);

  /*Configure GPIO pins : PG13 PG14 (LED3 Green, LED4 Red) */
  GPIO_InitStruct.Pin = LED_GREEN_PIN | LED_RED_PIN;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/**
  * @brief  Колбек, що викликається DMA по заповненню ПЕРШОЇ половини PDM-буфера (Ping).
  */
void HAL_I2S_RxHalfCpltCallback(I2S_HandleTypeDef *hi2s)
{
    if (hi2s->Instance == SPI2) {
        pdm_event = PDM_EVENT_HALF_READY;
    }
}

/**
  * @brief  Колбек, що викликається DMA по заповненню ДРУГОЇ половини PDM-буфера (Pong).
  */
void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s)
{
    if (hi2s->Instance == SPI2) {
        pdm_event = PDM_EVENT_FULL_READY;
    }
}

/**
  * @brief  Колбек помилки I2S.
  */
void HAL_I2S_ErrorCallback(I2S_HandleTypeDef *hi2s)
{
    if (hi2s->Instance == SPI2) {
        Error_Handler();
    }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
