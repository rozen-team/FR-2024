#include <Arduino.h>
#include <Servo.h>

#define RED_LEDS 7
#define BLUE_LEDS 7
#define GREEN_LEDS 7
#define WHITE_LEDS 7
#define YELLOW_LEDS 7


void setup() {
  pinMode(RED_LEDS, OUTPUT);
}

void loop() {
  digitalWrite(RED_LEDS, HIGH);
  delay(1000);
  digitalWrite(RED_LEDS, LOW);
  delay(1000);
}