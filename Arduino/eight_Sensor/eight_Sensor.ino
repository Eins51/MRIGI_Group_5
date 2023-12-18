#include <Wire.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>
#include <SparkFun_VL53L5CX_Library.h>

QWIICMUX myMux;

#define MUX_ADDR 0x70
#define NUMBER_OF_SENSORS 8

int imageResolution = 0;
int imageWidth = 0;

SparkFun_VL53L5CX myImagers[NUMBER_OF_SENSORS];
VL53L5CX_ResultsData measurementData[NUMBER_OF_SENSORS];

void setup() {
  Serial.begin(115200);
  Serial.println("Qwiic Mux Shield Read Example");

  Wire.begin();

  if (myMux.begin() == false) {
    Serial.println("Mux not detected. Freezing...");
    while (1)
      ;
  }
  Serial.println("Mux detected");

  for (byte x = 0; x < NUMBER_OF_SENSORS; x++) {
    enableMuxPort(x);
    myImagers[x].begin();
    myImagers[x].setAddress(0x45 + x);
    myImagers[x].setResolution(8 * 8);
    myImagers[x].setRangingFrequency(15);
    myImagers[x].startRanging();

    Serial.print("Imager address:");
    Serial.println(myImagers[x].getAddress(), HEX);

    disableMuxPort(x);
  }

  imageResolution = myImagers[0].getResolution();
  imageWidth = sqrt(imageResolution);
  Serial.println();
}

void loop() {
  for (byte sensorNumber = 0; sensorNumber < NUMBER_OF_SENSORS; sensorNumber++) {
    enableMuxPort(sensorNumber);
    myMux.setPort(sensorNumber);


    while (!myImagers[sensorNumber].isDataReady()) {
      delay(10);
    }

    if (myImagers[sensorNumber].getRangingData(&measurementData[sensorNumber])) {
      Serial.print("Sensor ");
      Serial.println(sensorNumber + 1);

      for (int y = 0; y < imageWidth * imageWidth; y += imageWidth) {
        for (int x = imageWidth - 1; x >= 0; x--) {
          Serial.print("\t");
          Serial.print(measurementData[sensorNumber].distance_mm[x + y]);
        }
        Serial.println();
      }
      Serial.println();
    }

    disableMuxPort(sensorNumber);
  }

  delay(500);
}

void enableMuxPort(byte portNumber) {
  if (portNumber > 7)
    portNumber = 7;

  Wire.requestFrom(MUX_ADDR, 1);
  if (!Wire.available())
    return;
  byte settings = Wire.read();

  settings |= (1 << portNumber);

  Wire.beginTransmission(MUX_ADDR);
  Wire.write(settings);
  Wire.endTransmission();
}

void disableMuxPort(byte portNumber) {
  if (portNumber > 7)
    portNumber = 7;

  Wire.requestFrom(MUX_ADDR, 1);
  if (!Wire.available())
    return;
  byte settings = Wire.read();

  settings &= ~(1 << portNumber);

  Wire.beginTransmission(MUX_ADDR);
  Wire.write(settings);
  Wire.endTransmission();
}
