#include "main.h"
#include "usb_device.h"
#include "usbd_cdc_if.h"
#include <stdio.h>
#include <string.h>

/* Bench-only firmware. No motor peripherals and no IMU configuration writes.
 * PB8/PB9 stay as inputs until an explicit single-address read command.
 * The initial 16 KiB containing the WeAct HID bootloader is excluded by the linker.
 */
extern PCD_HandleTypeDef hpcd_USB_OTG_FS;
extern USBD_HandleTypeDef hUsbDeviceFS;
volatile uint8_t bench_command;
static char tx[240];
static uint32_t pull_mode = GPIO_NOPULL;
static uint32_t half_period_us = 50;
#define SCL GPIO_PIN_8
#define SDA GPIO_PIN_9

void SysTick_Handler(void) { HAL_IncTick(); }
void OTG_FS_IRQHandler(void) { HAL_PCD_IRQHandler(&hpcd_USB_OTG_FS); }
void Error_Handler(void) { __disable_irq(); while (1) {} }

void SystemClock_Config(void) {
    RCC_OscInitTypeDef o = {0}; RCC_ClkInitTypeDef c = {0};
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    o.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    o.HSEState = RCC_HSE_ON;
    o.PLL.PLLState = RCC_PLL_ON; o.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    o.PLL.PLLM = 25; o.PLL.PLLN = 192; o.PLL.PLLP = RCC_PLLP_DIV2; o.PLL.PLLQ = 4;
    if (HAL_RCC_OscConfig(&o) != HAL_OK) Error_Handler();
    c.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    c.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK; c.AHBCLKDivider = RCC_SYSCLK_DIV1;
    c.APB1CLKDivider = RCC_HCLK_DIV2; c.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&c, FLASH_LATENCY_3) != HAL_OK) Error_Handler();
}
static void delay_us(uint32_t us) {
    uint32_t start = DWT->CYCCNT;
    while ((uint32_t)(DWT->CYCCNT - start) < us * (SystemCoreClock / 1000000)) {}
}
static void bus_inputs(void) {
    GPIO_InitTypeDef g = {0}; g.Pin = SCL | SDA; g.Mode = GPIO_MODE_INPUT;
    g.Pull = pull_mode; HAL_GPIO_Init(GPIOB, &g);
}
static void line(uint16_t pin, int high) {
    HAL_GPIO_WritePin(GPIOB, pin, high ? GPIO_PIN_SET : GPIO_PIN_RESET); delay_us(half_period_us);
}
static int scl_high(void) {
    line(SCL, 1); uint32_t t = DWT->CYCCNT;
    while (!HAL_GPIO_ReadPin(GPIOB, SCL)) {
        if ((uint32_t)(DWT->CYCCNT - t) > SystemCoreClock / 500) return 0;
    }
    return 1;
}
static int start(void) {
    line(SDA, 1); if (!scl_high()) return 0;
    line(SDA, 0); line(SCL, 0); return 1;
}
static void stop(void) { line(SDA, 0); scl_high(); line(SDA, 1); }
static int write_byte(uint8_t v) {
    for (int i = 0; i < 8; ++i) {
        line(SDA, !!(v & 0x80)); if (!scl_high()) return -1;
        line(SCL, 0); v <<= 1;
    }
    line(SDA, 1); if (!scl_high()) return -1;
    int ack = !HAL_GPIO_ReadPin(GPIOB, SDA); line(SCL, 0); return ack;
}
static int read_byte(uint8_t *v, int ack) {
    *v = 0; line(SDA, 1);
    for (int i = 0; i < 8; ++i) {
        if (!scl_high()) return 0;
        *v = (*v << 1) | !!HAL_GPIO_ReadPin(GPIOB, SDA); line(SCL, 0);
    }
    line(SDA, !ack); if (!scl_high()) return 0;
    line(SCL, 0); line(SDA, 1); return 1;
}
static int read_jy(uint8_t reg, uint8_t *data, int len) {
    if (!HAL_GPIO_ReadPin(GPIOB, SCL) || !HAL_GPIO_ReadPin(GPIOB, SDA)) return 1;
    GPIO_InitTypeDef g = {0};
    HAL_GPIO_WritePin(GPIOB, SCL | SDA, GPIO_PIN_SET);
    g.Pin = SCL | SDA; g.Mode = GPIO_MODE_OUTPUT_OD; g.Pull = pull_mode;
    g.Speed = GPIO_SPEED_FREQ_LOW; HAL_GPIO_Init(GPIOB, &g);
    int rc = 2;
    if (!start()) goto done;
    int ack;
    rc = 3; ack = write_byte(0x50 << 1); if (ack != 1) { if (ack < 0) rc += 10; goto done; }
    rc = 4; ack = write_byte(reg); if (ack != 1) { if (ack < 0) rc += 10; goto done; }
    rc = 2; if (!start()) goto done;
    rc = 5; ack = write_byte((0x50 << 1) | 1); if (ack != 1) { if (ack < 0) rc += 10; goto done; }
    rc = 6;
    for (int i = 0; i < len; ++i) if (!read_byte(data + i, i + 1 < len)) goto done;
    rc = 0;
done:
    stop(); bus_inputs(); return rc;
}
static void reply(void) {
    uint32_t t = HAL_GetTick();
    while (CDC_Transmit_FS((uint8_t *)tx, strlen(tx)) != USBD_OK) {
        if (HAL_GetTick() - t > 200) return;
    }
    /* Keep the static buffer unchanged until the endpoint completes. */
    while (hUsbDeviceFS.pClassData && ((USBD_CDC_HandleTypeDef *)hUsbDeviceFS.pClassData)->TxState) {
        if (HAL_GetTick() - t > 200) return;
    }
}
int main(void) {
    SCB->VTOR = 0x08004000;
    HAL_Init(); SystemClock_Config();
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    __HAL_RCC_GPIOB_CLK_ENABLE(); bus_inputs();
    /* Detach from the previous HID identity before starting CDC. */
    __HAL_RCC_GPIOA_CLK_ENABLE(); GPIO_InitTypeDef g = {0};
    g.Pin = GPIO_PIN_12; g.Mode = GPIO_MODE_OUTPUT_PP; g.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &g); HAL_GPIO_WritePin(GPIOA, GPIO_PIN_12, GPIO_PIN_RESET);
    HAL_Delay(100); MX_USB_DEVICE_Init();
    while (1) {
        uint8_t command = bench_command; bench_command = 0;
        if (command == 'S' || command == 'P' || command == 'N' || command == 'L' || command == 'T' || command == 'H' || (command >= '1' && command <= '9')) {
            if (command == 'T' || command == 'H') half_period_us = command == 'T' ? 5 : 50;
            if (command >= '1' && command <= '9') half_period_us = command - '0';
            if (command == 'P' || command == 'N') {
                pull_mode = command == 'P' ? GPIO_PULLUP : GPIO_NOPULL;
                bus_inputs(); HAL_Delay(5);
            }
            if (command == 'L') {
                HAL_GPIO_WritePin(GPIOB, SCL | SDA, GPIO_PIN_SET);
                g.Pin = SCL | SDA; g.Mode = GPIO_MODE_OUTPUT_OD;
                g.Pull = pull_mode; g.Speed = GPIO_SPEED_FREQ_LOW;
                HAL_GPIO_Init(GPIOB, &g); HAL_Delay(5);
            }
            snprintf(tx, sizeof(tx), "STATUS v=3 tick=%lu core=%lu scl=%u sda=%u pull=%s mode=%s half_us=%lu addr=0x50\r\n",
                HAL_GetTick(), SystemCoreClock, !!HAL_GPIO_ReadPin(GPIOB,SCL), !!HAL_GPIO_ReadPin(GPIOB,SDA),
                pull_mode == GPIO_PULLUP ? "weak_internal" : "none", command == 'L' ? "released_OD" : "input",half_period_us);
            if (command == 'L') bus_inputs();
            reply();
        } else if (command == 'D' || command == 'I' || command == 'V') {
            uint8_t bytes[24] = {0};
            uint8_t reg = command == 'D' ? 0x34 : command == 'I' ? 0x1a : 0x2e;
            int len = command == 'D' ? 24 : 2;
            uint32_t t = HAL_GetTick(); int rc = read_jy(reg, bytes, len);
            int used = snprintf(tx, sizeof(tx), "READ tick=%lu reg=%02X rc=%d us=%lu raw=", t, reg, rc, (HAL_GetTick()-t)*1000);
            if (rc == 0) for (int i=0;i<len;++i) used += snprintf(tx+used,sizeof(tx)-used,"%02X",bytes[i]);
            snprintf(tx+used,sizeof(tx)-used,"\r\n"); reply();
        }
        HAL_Delay(1);
    }
}
