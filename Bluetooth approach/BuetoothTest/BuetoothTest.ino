#include <SoftwareSerial.h>
#include <ArduinoJson.h>
#include <string.h>

const byte rxPin = A8;
const byte txPin = A9;
const int Trig = A3;
const int Echo = A2;
const int PWM2A = 11;      //M1 motor
const int PWM2B = 3;       //M2 motor
const int PWM0A = 6;       //M3 motor
const int PWM0B = 5;       //M4 motor
const int DIR_CLK = 4;     // Data input clock line
const int DIR_EN = 7;      //Equip the L293D enabling pins
const int DATA = 8;        // USB cable
const int DIR_LATCH = 12;  // Output memory latch clock

int current_direction = 0;
int target_direction = 0;

const int directions[] = {
  39 /*forward*/,
  216 /*backward*/,
  149 /*turn right*/,
  106 /*turn left*/,
  116 /*left translate*/,
  139 /*right translate*/,
  36 /*upleft diagonal*/,
  3 /*upright diagonal*/,
  80 /*lowleft diagonal*/,
  136 /*lowright diagonal*/,
  0 /*stop*/
};

SoftwareSerial BTSerial(rxPin, txPin);

float checkDistance() {
  digitalWrite(Trig, LOW);
  delayMicroseconds(2);
  digitalWrite(Trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(Trig, LOW);
  return pulseIn(Echo, HIGH) / 58.0;
}

void controlMotors(int wheel1Speed, int wheel2Speed, int wheel3Speed, int wheel4Speed, int dir) {
  analogWrite(PWM2A, wheel1Speed);  // Motor 1
  analogWrite(PWM2B, wheel2Speed);  // Motor 2
  analogWrite(PWM0A, wheel3Speed);  // Motor 3
  analogWrite(PWM0B, wheel4Speed);  // Motor 4

  digitalWrite(DIR_LATCH, LOW);            //DIR_LATCH sets the low level and writes the direction of motion in preparation
  shiftOut(DATA, DIR_CLK, MSBFIRST, dir);  //Write Dir motion direction value
  digitalWrite(DIR_LATCH, HIGH);           //DIR_LATCH sets the high level and outputs the direction of motion
}

void setup() {
  pinMode(rxPin, INPUT);
  pinMode(txPin, OUTPUT);
  BTSerial.begin(9600);
  Serial.begin(9600);
  //Configure as output mode
  pinMode(DIR_CLK, OUTPUT);
  pinMode(DATA, OUTPUT);
  pinMode(DIR_EN, OUTPUT);
  pinMode(DIR_LATCH, OUTPUT);
  pinMode(PWM0B, OUTPUT);
  pinMode(PWM0A, OUTPUT);
  pinMode(PWM2A, OUTPUT);
  pinMode(PWM2B, OUTPUT);
  pinMode(Trig, OUTPUT);
  pinMode(Echo, INPUT);
}

void loop() {

  // Send data to Python as before
  float distance = checkDistance();

  DynamicJsonDocument doc(256);
  
  doc["distance"] = distance;
  doc["current_direction"] = current_direction;
  doc["target_direction"] = target_direction;

  String output;
  serializeJson(doc, output);
  BTSerial.print('<');  // Start marker
  BTSerial.print(output);
  BTSerial.println('>');  // End marker

  delay(150);

  // Read and process incoming data from Python
  if (BTSerial.available()) {
    String incomingData = BTSerial.readStringUntil('\n');
    Serial.println("Received Data: " + incomingData);  // Debugging raw data

    // Extract JSON substring between '{' and '}' markers
    int startIndex = incomingData.indexOf('{');
    int endIndex = incomingData.lastIndexOf('}');

    if (startIndex != -1 && endIndex != -1 && endIndex > startIndex) {
      String jsonString = incomingData.substring(startIndex, endIndex + 1);

      DynamicJsonDocument responseDoc(256);
      DeserializationError error = deserializeJson(responseDoc, jsonString);

      if (!error) {
        int wheel1 = responseDoc["W1"];
        int wheel2 = responseDoc["W2"];
        int wheel3 = responseDoc["W3"];
        int wheel4 = responseDoc["W4"];
        current_direction = responseDoc["D"];

        // Ensure wheel speeds are within valid PWM range (0-255)
        wheel1 = constrain(wheel1, 0, 255);
        wheel2 = constrain(wheel2, 0, 255);
        wheel3 = constrain(wheel3, 0, 255);
        wheel4 = constrain(wheel4, 0, 255);

        // Control the motors
        controlMotors(wheel1, wheel2, wheel3, wheel4, directions[current_direction]);


      } else {
        Serial.println("Error parsing JSON");
      }
    } else {
      Serial.println("Invalid data format, JSON not found");
    }
  }
}