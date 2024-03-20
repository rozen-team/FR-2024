#include <Arduino.h>
#include <SoftwareSerial.h>
#include <Servo.h>

#define RED_LEDS 7
#define BLUE_LEDS 7
#define GREEN_LEDS 7
#define WHITE_LEDS 7
#define YELLOW_LEDS 7

SoftwareSerial BT (10, 11); // RX, TX

Servo BigLeft;
Servo BigRight;
Servo SmallLeft;
Servo SmallRight;


void open();
void close();
void indicate();

int get_stateBT() {
  int state = 0;
  if (BT.available()) {
    state = BT.read();
  }
  return state;
}

void setup() {
  pinMode(RED_LEDS, OUTPUT);
  pinMode(BLUE_LEDS, OUTPUT);
  pinMode(GREEN_LEDS, OUTPUT);
  pinMode(WHITE_LEDS, OUTPUT);
  pinMode(YELLOW_LEDS, OUTPUT);

  BigLeft.attach(2);
  BigRight.attach(3);
  SmallLeft.attach(4);
  SmallRight.attach(4);
}

void loop() {
  uint8_t state = get_stateBT();

  switch (state) {
    case 1:
      digitalWrite(RED_LEDS, HIGH);
      break;
    case 2:
      digitalWrite(BLUE_LEDS, HIGH);
      break;
    case 3:
      digitalWrite(GREEN_LEDS, HIGH);
      break;
    case 4:
      digitalWrite(WHITE_LEDS, HIGH);
      break;
  }
}
