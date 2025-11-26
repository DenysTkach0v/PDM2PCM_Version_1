/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body - PDM Mic to FFT/Stream
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "pdm2pcm.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "arm_math.h" // CMSIS-DSP Library
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
// Створюємо тип для зручного перемикання режимів
typedef enum {
  MODE_FFT_ANALYSIS, // Аналіз на мікроконтролері
  MODE_STREAM_PCM    // Передача сирих даних на ПК
} OperatingMode;
/* USER CODE END PTD */

/* Private define --------------------------------/ Аналіз на мікроконтролері----------------------------*/
/* USER CODE BEGIN PD */
// --- ГОЛОВНІ НАЛАШТУВАННЯ ---
#define CURRENT_MODE MODE_STREAM_PCM // ОБЕРІТЬ РЕЖИМ ТУТ
#define PDM_BUF_SIZE 128
#define PCM_BUF_SIZE 16
#define FFT_SIZE 1024       // Розмір FFT (повинен бути ступенем двійки)
#define SAMPLE_RATE 16000   // Частота дискретизації звуку в Гц
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CRC_HandleTypeDef hcrc;
I2S_HandleTypeDef hi2s2;
DMA_HandleTypeDef hdma_spi2_rx;
UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */
// --- Буфери ---
uint8_t PDM_Buffer[PDM_BUF_SIZE];
int16_t PCM_Buffer[PCM_BUF_SIZE];

// --- Змінні для FFT ---
float32_t pcmAccumulator[FFT_SIZE]; // Буфер для накопичення даних для FFT
float32_t fftInput[FFT_SIZE];
float32_t fftOutput[FFT_SIZE];
uint16_t pcmIndex = 0;
arm_rfft_fast_instance_f32 fftHandler; // Обробник FFT з бібліотеки CMSIS-DSP

// --- Прапорці для зв'язку між перериваннями та головним циклом ---
volatile uint8_t PDM_Data_Available = 0; // 0=нема даних, 1=готова перша половина, 2=готова друга
volatile uint8_t FFT_Ready = 0;          // Прапорець готовності до FFT-аналізу
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_CRC_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_I2S2_Init(void);
/* USER CODE BEGIN PFP */
void ProcessFFT(void);
void FindDominantFrequency(void);
void AccumulatePCMSamples(int16_t* pcm_data);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/**
  * @brief Функція для копіювання нових PCM-семплів у великий буфер для FFT.
  */
void AccumulatePCMSamples(int16_t* pcm_data) {
    for (uint16_t i = 0; i < PCM_BUF_SIZE; i++) {
        pcmAccumulator[pcmIndex++] = (float32_t)pcm_data[i];
        // Коли буфер заповнено, встановлюємо прапорець для запуску FFT
        if (pcmIndex >= FFT_SIZE) {
            pcmIndex = 0;
            FFT_Ready = 1;
        }
    }
}

/**
  * @brief Виконує FFT-аналіз над накопиченими даними.
  */
void ProcessFFT(void) {
    // Застосовуємо віконну функцію (Ханна) для зменшення "розтікання" спектру
    for (uint16_t i = 0; i < FFT_SIZE; i++) {
        float32_t window = 0.5f * (1.0f - arm_cos_f32(2.0f * PI * i / (FFT_SIZE - 1)));
        // Нормалізуємо int16 в діапазон [-1.0, 1.0] і множимо на вікно
        fftInput[i] = (pcmAccumulator[i] / 32768.0f) * window;
    }
    // Виконуємо дійсне швидке перетворення Фур'є
    arm_rfft_fast_f32(&fftHandler, fftInput, fftOutput, 0);
    // Обчислюємо амплітудний спектр (величину комплексних чисел)
    arm_cmplx_mag_f32(fftOutput, fftOutput, FFT_SIZE / 2);
}

/**
  * @brief Знаходить домінантну частоту в спектрі та відправляє результат.
  */
void FindDominantFrequency(void) {
    char msg[100];
    uint32_t maxIndex;
    float32_t maxValue;

    // Ігноруємо перші біни (постійну складову та низькочастотний шум), починаючи з 50 Гц
    uint16_t startBin = (uint16_t)((50.0f * FFT_SIZE) / SAMPLE_RATE);
    uint16_t searchSize = (FFT_SIZE / 2) - startBin;
    if (searchSize <= 0) return;

    // Знаходимо індекс максимального значення в спектрі
    arm_max_f32(&fftOutput[startBin], searchSize, &maxValue, &maxIndex);

    // Розраховуємо частоту, що відповідає цьому індексу
    float32_t dominantFreq = ((float32_t)(maxIndex + startBin) * SAMPLE_RATE) / FFT_SIZE;

    // Відправляємо результат через UART
    sprintf(msg, "Dominant Freq: %.1f Hz (Magnitude: %.2f)\r\n", dominantFreq, maxValue);
    HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_CRC_Init();
  MX_USART1_UART_Init();
  MX_PDM2PCM_Init();
  MX_I2S2_Init();

  /* USER CODE BEGIN 2 */
  // ВИПРАВЛЕНО: Критично важливий крок! Ініціалізація обробника FFT.
  // Без цього виклику режим MODE_FFT_ANALYSIS ніколи не запрацює.
  if (arm_rfft_fast_init_f32(&fftHandler, FFT_SIZE) != ARM_MATH_SUCCESS) {
      char msg[] = "FFT Init FAILED!\r\n";
      HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);
      Error_Handler();
  }

  // Виводимо стартові повідомлення
  char msg[100];
  sprintf(msg, "\r\n--- System Initialized ---\r\n");
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);
  if (CURRENT_MODE == MODE_FFT_ANALYSIS) {
      sprintf(msg, "Mode: FFT Analysis on-board\r\nFFT Size: %d\r\n", FFT_SIZE);
  } else {
      sprintf(msg, "Mode: Streaming PCM data to PC\r\n");
  }
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);

  // Очищуємо буфери
  memset(PDM_Buffer, 0, PDM_BUF_SIZE);
  memset(pcmAccumulator, 0, sizeof(pcmAccumulator));
  pcmIndex = 0;

  // Запускаємо прийом даних з мікрофона через I2S+DMA
  // HAL-функція очікує розмір у 16-бітних словах, тому ділимо на 2
  if (HAL_I2S_Receive_DMA(&hi2s2, (uint16_t*)PDM_Buffer, PDM_BUF_SIZE / 2) != HAL_OK) {
      Error_Handler();
  }
  sprintf(msg, "I2S DMA Started. Listening for audio...\r\n\r\n");
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */
    /* USER CODE BEGIN 3 */

    // ВИПРАВЛЕНО: Повністю переписана логіка для чистоти та ефективності.
    if (PDM_Data_Available > 0) {
        uint8_t* pPdmBuf; // Вказівник на ту частину буфера, яка готова

        if (PDM_Data_Available == 1) {
            pPdmBuf = &PDM_Buffer[0]; // Перша половина
        } else {
            pPdmBuf = &PDM_Buffer[PDM_BUF_SIZE / 2]; // Друга половина
        }
        PDM_Data_Available = 0; // Скидаємо прапорець

        // Конвертуємо PDM в PCM один раз
        if (MX_PDM2PCM_Process((uint16_t*)pPdmBuf, (uint16_t*)PCM_Buffer) == 0) {
            // Тепер виконуємо дію залежно від обраного режиму
            if (CURRENT_MODE == MODE_STREAM_PCM) {
                // Режим 1: Просто відправляємо сирі дані на ПК
                HAL_UART_Transmit(&huart1, (uint8_t*)PCM_Buffer, PCM_BUF_SIZE * sizeof(int16_t), 100);
            } else { // CURRENT_MODE == MODE_FFT_ANALYSIS
                // Режим 2: Накопичуємо дані для аналізу
                AccumulatePCMSamples(PCM_Buffer);
            }
        }
    }

    // ВИПРАВЛЕНО: Блок обробки FFT винесено з обробників DMA.
    // Він спрацює одразу, як тільки накопичиться достатньо даних.
    if (FFT_Ready == 1) {
        FFT_Ready = 0; // Скидаємо прапорець
        ProcessFFT();
        FindDominantFrequency();
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
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

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

  // РЕКОМЕНДАЦІЯ: Для точної частоти дискретизації звуку краще використовувати
  // окремий тактовий генератор PLLI2S.
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_I2S;
  PeriphClkInitStruct.PLLI2S.PLLI2SN = 192; // Ці значення потрібно розрахувати
  PeriphClkInitStruct.PLLI2S.PLLI2SR = 2;  // для вашої точної частоти
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief CRC Initialization Function
  */
static void MX_CRC_Init(void)
{
  hcrc.Instance = CRC;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  __HAL_CRC_DR_RESET(&hcrc);
}

/**
  * @brief I2S2 Initialization Function
  */
static void MX_I2S2_Init(void)
{
  hi2s2.Instance = SPI2;
  hi2s2.Init.Mode = I2S_MODE_MASTER_RX;
  hi2s2.Init.Standard = I2S_STANDARD_LSB;
  hi2s2.Init.DataFormat = I2S_DATAFORMAT_16B;
  hi2s2.Init.MCLKOutput = I2S_MCLKOUTPUT_DISABLE;
  hi2s2.Init.AudioFreq = I2S_AUDIOFREQ_32K; // Частота I2S_SCK. Справжня аудіо-частота залежить від децимації.
  hi2s2.Init.CPOL = I2S_CPOL_LOW;
  hi2s2.Init.ClockSource = I2S_CLOCK_PLL; // Використовуємо PLLI2S для кращої точності
  hi2s2.Init.FullDuplexMode = I2S_FULLDUPLEXMODE_DISABLE;
  if (HAL_I2S_Init(&hi2s2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART1 Initialization Function
  */
static void MX_USART1_UART_Init(void)
{
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
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
}

/**
  * @brief Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{
  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream3_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream3_IRQn);
}

/**
  * @brief GPIO Initialization Function
  */
static void MX_GPIO_Init(void)
{
  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
}

/* USER CODE BEGIN 4 */

/**
  * @brief  Колбек, що викликається по заповненню першої половини DMA-буфера.
  */
void HAL_I2S_RxHalfCpltCallback(I2S_HandleTypeDef *hi2s)
{
    if (hi2s->Instance == SPI2) {
        PDM_Data_Available = 1; // Встановлюємо прапорець
    }
}

/**
  * @brief  Колбек, що викликається по заповненню всього DMA-буфера.
  */
void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s)
{
    if (hi2s->Instance == SPI2) {
        PDM_Data_Available = 2; // Встановлюємо прапорець
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
  */
void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

