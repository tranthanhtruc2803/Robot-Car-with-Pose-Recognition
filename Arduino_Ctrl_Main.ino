int i=0, x, current_state=0, next_state=0;
char receive;

// Steering angle (Steer)
#define dirPin 13
#define stepPin 12

// DC Motor
const int in2 = 2;
const int in1 = 3;
const int in4 = 5;
const int in3 = 4;
const int enA = 6;
const int enB = 7;

// Step truc ngang (Shake)
const int dirX = 8;
const int stepX = 9;

// Step truc doc (Nod)
const int dirY = 10;
const int stepY = 11;

// Limit switch
#define nodlim A0
#define shakelim A1

void setup() {
  // Step Motor 
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(stepX, OUTPUT);
  pinMode(dirX, OUTPUT);
  pinMode(stepY, OUTPUT);
  pinMode(dirY, OUTPUT);
  
  // DC Motor
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);

  //  Limit switch
  pinMode(nodlim,INPUT_PULLUP); 
  pinMode(shakelim,INPUT_PULLUP);
   
  // Turn off all Output
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  digitalWrite(enA, LOW);
  digitalWrite(enB, LOW);
  digitalWrite(stepX, LOW);
  digitalWrite(dirX, LOW);
  digitalWrite(stepY, LOW);
  digitalWrite(dirY, LOW);
  
  Serial.begin(9600);       // initialize UART with baud rate of 9600 bps
  
  init_setup();
}

void init_setup(){
  while(digitalRead(nodlim)==HIGH)
  {
    inY(50);
  }
//  outY(10);
  while(digitalRead(shakelim)==HIGH)
  {
    rightX(50);
  }
  leftX(300);
  }
  
void left() {
  // Set the spinning direction CW:
  digitalWrite(dirPin, LOW);
  // turn right: 1600 = 90 degree
  for(x = 0; x < 800; x++) 
  {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
  }
}

void right() {
  // Set the spinning direction CCW:
  digitalWrite(dirPin, HIGH);
  // turn left: 1600 = 90 degree
  for(x = 0; x < 800; x++) // Cho chay 1 goc
  {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
  }
}

//  Shake
void leftX(int a) {
  // Set the spinning direction CW:
  digitalWrite(dirX, HIGH);
  // turn left: 1600 = 90 degree
  for(x = 0; x < a; x++) 
  {
    digitalWrite(stepX, HIGH);
    delayMicroseconds(1500);
    digitalWrite(stepX, LOW);
    delayMicroseconds(1500);
  }
}

void rightX(int a) {
  // Set the spinning direction CW:
  digitalWrite(dirX, LOW);
  // Cho chay 1 goc
  // 1600 = 90 degree
  for(x = 0; x < a; x++) 
  {
    digitalWrite(stepX, HIGH);
    delayMicroseconds(1500);
    digitalWrite(stepX, LOW);
    delayMicroseconds(1500);
  }
}

void shake(){
  leftX(200);
  delay(500);
  rightX(200);
  delay(100);
  rightX(200);
  delay(500);
  leftX(200);
  delay(500);
  }
  
//  Nod  
void outY(int a) {
  // Set the spinning direction CW:
  digitalWrite(dirY, HIGH);
  for(x = 0; x < a; x++) 
  {
    digitalWrite(stepY, HIGH);
    delayMicroseconds(1500);
    digitalWrite(stepY, LOW);
    delayMicroseconds(1500);
  }
}

//  Nod
void inY(int a) {
  // Set the spinning direction CW:
  digitalWrite(dirY, LOW);
  for(x = 0; x < a; x++) 
  {
    digitalWrite(stepY, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepY, LOW);
    delayMicroseconds(500);
  }
}

void nod(){
  while(digitalRead(nodlim)==HIGH)
  {
    inY(50);
  }
//  delay(1100);
  }

void forward(){
  stop_car();
  digitalWrite(in1,LOW);
  digitalWrite(in2,HIGH);
  digitalWrite(in4,LOW);
  digitalWrite(in3,HIGH);
  analogWrite(enA, 255);
  analogWrite(enB, 255);
  delay(4500);
  // Stop
  analogWrite(enA, 0);
  analogWrite(enB, 0);
}

void backward(){
  stop_car();
  digitalWrite(in2,LOW);
  digitalWrite(in1,HIGH);
  digitalWrite(in3,LOW);
  digitalWrite(in4,HIGH);
  analogWrite(enA, 250);
  analogWrite(enB, 250);
  delay(4000);
  // Stop
  analogWrite(enA, 0);
  analogWrite(enB, 0);
}

void forwA(){
  digitalWrite(in1,LOW);
  digitalWrite(in2,HIGH);
  analogWrite(enB, 0);
  analogWrite(enA, 200);
  delay(4500);
  analogWrite(enA, 0);
  }
  
void forwB(){
  digitalWrite(in4,LOW);
  digitalWrite(in3,HIGH);
  analogWrite(enA, 0);
  analogWrite(enB, 200);
  delay(4500);
  analogWrite(enB, 0);
  }
  
void turn_right() {
  stop_car();
  right();
  delay(50);
  forwA();
  left();
}  

void turn_left() {
  stop_car();
  left();
  delay(50);
  forwB();
  right();
}

void stop_car(){
  digitalWrite(in2,LOW);
  digitalWrite(in1,LOW);
  digitalWrite(in3,LOW);
  digitalWrite(in4,LOW);
  analogWrite(enA, 0);
  analogWrite(enB, 0);
  }

void loop() {
  if (Serial.available()) {
    char data_rcvd = char(Serial.read());   // read one byte from serial buffer and save to data_rcvd

    Serial.println(data_rcvd);
    switch (data_rcvd){
      case 'f':
        next_state = 1;
        break;
      case 'b':
        next_state = 2;
        break;
      case 'l':
        next_state = 3;
        break;
      case 'r':
        next_state = 4;
        break;
      case 'x':
        next_state = 5;
        break;
      case '6':
        next_state = 6;
        break;
      case '7':
        next_state = 7;
        break;
       case '8':
        next_state = 8;
        break;
      default:
        break;
    } 
    if (current_state != next_state) 
    {
      current_state = next_state;
      switch (current_state){
        case 1:
        forward();
        Serial.print("Forward");
        break;
      case 2:
        backward();
        Serial.print("Backward");
        break;
      case 3:
        turn_left();
        Serial.print("Left");
        break;
      case 4:
        turn_right();
        Serial.print("Right");
        break;
      case 5:
        stop_car();
        break;
      case 6:
        outY(700);
        break;
      case 7:
        nod();
        break;
       case 8:
        shake();
        break;
      default:
        break;
      }
    } 
  }
}
