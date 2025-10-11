/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body - PDM Microphone to Virtual COM Port
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
#include <arm_math.h>
#include "arm_const_structs.h"
#define CURRENT_MODE MODE_STREAM_PCM


/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define PDM_BUF_SIZE 128
#define PCM_BUF_SIZE 16
#define FFT_SIZE 512 // 512  1024 2048
#define SAMPLE_RATE 16000

uint8_t PDM_Buffer[PDM_BUF_SIZE];
int16_t PCM_Buffer[PCM_BUF_SIZE];
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
arm_rfft_fast_instance_f32 fftHandler;

volatile uint8_t PDM_Half_Transfer = 0;
volatile uint8_t PDM_Full_Transfer = 0;
volatile uint8_t FFT_Ready = 0;
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CRC_HandleTypeDef hcrc;

I2S_HandleTypeDef hi2s2;
DMA_HandleTypeDef hdma_spi2_rx;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */
float32_t fftInput[FFT_SIZE * 2];
float32_t fftOutput[FFT_SIZE];
float32_t pcmAccumulator[FFT_SIZE];
uint16_t pcmIndex = 0;
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
void FindDominantFrequencyTable(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
void ProcessFFT(void) {
  for (uint16_t i =0; i<FFT_SIZE;i++) {
    float32_t window = 0.5f *(1.0 - arm_cos_f32(2.0f*PI*i/(FFT_SIZE-1)));
    fftInput[i]=(pcmAccumulator[i]/32768.0f)*window;
  }
  arm_rfft_fast_f32(&fftHandler,fftInput,fftOutput,0);
  arm_cmplx_mag_f32(fftOutput,fftOutput, FFT_SIZE/2);
}
void FindDominantFrequency(void)
{
  char msg[100];
  uint32_t maxIndex;
  float32_t maxValue;

  // Пропускаємо низькі частоти (DC та шум) - починаємо з 50 Hz
  uint16_t startBin = (uint16_t)((50.0f * FFT_SIZE) / SAMPLE_RATE);
  uint16_t searchSize = (FFT_SIZE / 2) - startBin;

  // Знаходимо максимум
  arm_max_f32(&fftOutput[startBin], searchSize, &maxValue, &maxIndex);

  // Обчислюємо частоту
  float32_t dominantFreq = ((float32_t)(maxIndex + startBin) * SAMPLE_RATE) / FFT_SIZE;

  sprintf(msg, "Dominant Freq: %.1f Hz (Mag: %.2f)\r\n", dominantFreq, maxValue);
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);

  // Топ-5 частот
  sprintf(msg, "Top 5 frequencies:\r\n");
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);

  // Копіюємо для пошуку топ-5
  float32_t fftCopy[FFT_SIZE / 2];
  arm_copy_f32(fftOutput, fftCopy, FFT_SIZE / 2);

  for (int n = 0; n < 5; n++)
  {
    arm_max_f32(&fftCopy[startBin], searchSize, &maxValue, &maxIndex);
    float32_t freq = ((float32_t)(maxIndex + startBin) * SAMPLE_RATE) / FFT_SIZE;

    sprintf(msg, "  %d: %.1f Hz (%.2f)\r\n", n + 1, freq, maxValue);
    HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);

    // Обнуляємо знайдений пік
    fftCopy[maxIndex + startBin] = 0;
  }

  sprintf(msg, "---\r\n");
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void) {
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
  char msg[100];

  // Перевірка I2S clock
  uint32_t i2s_clock = HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_I2S);
  sprintf(msg, "I2S Clock: %lu Hz\r\n", i2s_clock);
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);

  sprintf(msg, "PDM Filter initialized\r\n");
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);

  sprintf(msg, "Decimation: %d\r\n", PDM1_filter_config.decimation_factor);
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);

  sprintf(msg, "Output samples: %d\r\n", PDM1_filter_config.output_samples_number);
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);

  // Тестуємо PDM фільтр з тестовими даними
  uint8_t test_pdm[128];
  for (int i = 0; i < 128; i++) {
    test_pdm[i] = 0xAA;  // Паттерн 10101010
  }

  int16_t test_pcm[16];
  uint8_t filter_result = MX_PDM2PCM_Process((uint16_t*)test_pdm, (uint16_t*)test_pcm);
  sprintf(msg, "Filter test result: %d\r\n", filter_result);
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);

  if (filter_result == 0) {
    sprintf(msg, "Test PCM[0-2]: %d %d %d\r\n", test_pcm[0], test_pcm[1], test_pcm[2]);
    HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);
  }

  HAL_Delay(1000);

  // Очищаємо буфер
  memset(PDM_Buffer, 0, PDM_BUF_SIZE);
  memset(pcmAccumulator,0,sizeof(pcmAccumulator));
  pcmIndex=0;
  // Запускаємо I2S DMA
  if (HAL_I2S_Receive_DMA(&hi2s2, (uint16_t*)PDM_Buffer, PDM_BUF_SIZE/2) != HAL_OK) {
    sprintf(msg, "I2S DMA Start ERROR!\r\n");
    HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);
    Error_Handler();
  } else {
    sprintf(msg, "I2S DMA Started OK\r\n");
    HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);
  }
  sprintf(msg, "\r\nListening for audio...\r\n\r\n");

  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), 100);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  uint32_t process_count = 0;
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    // Обробка першої половини буфера
    if (PDM_Half_Transfer == 1)
    {
      PDM_Half_Transfer = 0;
      process_count++;
      uint8_t result = MX_PDM2PCM_Process((uint16_t*)&PDM_Buffer[0], (uint16_t*)PCM_Buffer);


      // Діагностика RAW PDM даних
      if (process_count % 100 == 0) {
        char dbg[100];
        sprintf(dbg, "PDM[0-3]: %02X %02X %02X %02X\r\n",
                PDM_Buffer[0], PDM_Buffer[1], PDM_Buffer[2], PDM_Buffer[3]);
        HAL_UART_Transmit(&huart1, (uint8_t*)dbg, strlen(dbg), 100);
      }

      // Конвертуємо PDM → PCM
      result = MX_PDM2PCM_Process((uint16_t*)&PDM_Buffer[0], (uint16_t*)PCM_Buffer);

      if (result == 0)
      {
        // Діагностика PCM даних
        for (uint16_t i = 0; i < PCM_BUF_SIZE; i++)
        {
          pcmAccumulator[pcmIndex++] = (float32_t)PCM_Buffer[i];

          // Коли накопичили достатньо для FFT
          if (pcmIndex >= FFT_SIZE)
          {
            pcmIndex = 0;
            FFT_Ready = 1;
          }
        }

        // Відправляємо бінарні дані (розкоментуйте коли все працює)
        HAL_UART_Transmit(&huart1, (uint8_t*)PCM_Buffer, PCM_BUF_SIZE * 2, 100);
      }
    }

    if (PDM_Full_Transfer == 1)
    {
      PDM_Full_Transfer = 0;

      uint8_t result = MX_PDM2PCM_Process((uint16_t*)&PDM_Buffer[PDM_BUF_SIZE / 2], (uint16_t*)PCM_Buffer);

      if (result == 0)
      {
        // for (uint16_t i = 0; i < PCM_BUF_SIZE; i++)
        // {
        //   pcmAccumulator[pcmIndex++] = (float32_t)PCM_Buffer[i];
        //
        //   if (pcmIndex >= FFT_SIZE)
        //   {
        //     pcmIndex = 0;
        //     FFT_Ready = 1;
        //   }
        // }
         HAL_UART_Transmit(&huart1, (uint8_t*)PCM_Buffer, PCM_BUF_SIZE * 2, 100);
      }
      if (FFT_Ready == 1)
      {
        FFT_Ready = 0;

        ProcessFFT();
        FindDominantFrequency();
      }
    }
  }
}

  /* USER CODE END 3 */


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
  hi2s2.Init.Standard = I2S_STANDARD_PHILIPS;
  hi2s2.Init.DataFormat = I2S_DATAFORMAT_16B;
  hi2s2.Init.MCLKOutput = I2S_MCLKOUTPUT_DISABLE;
  hi2s2.Init.AudioFreq = I2S_AUDIOFREQ_48K;
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

  /* DMA interrupt init */
  /* DMA1_Stream3_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream3_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */
  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */
  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/**
  * @brief  Half transfer callback (перша половина буфера заповнена)
  */
void HAL_I2S_RxHalfCpltCallback(I2S_HandleTypeDef *hi2s)
{
  if (hi2s->Instance == SPI2)
  {
    PDM_Half_Transfer = 1;
  }
}

/**
  * @brief  Full transfer callback (весь буфер заповнений)
  */
void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s)
{
  if (hi2s->Instance == SPI2)
  {
    PDM_Full_Transfer = 1;
  }
}

/**
  * @brief  Error callback
  */
void HAL_I2S_ErrorCallback(I2S_HandleTypeDef *hi2s)
{
  if (hi2s->Instance == SPI2)
  {
    // Можна додати обробку помилок
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
