#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <SoftwareSerial.h>

SoftwareSerial esp(3,1);

int x;

int toggleState_1 = 1; //Define integer to remember the toggle state for relay 1
int toggleState_2 = 1; //Define integer to remember the toggle state for relay 2


// Update these with values suitable for your network.

const char* ssid = "TTT"; //WiFI Name
const char* password = "meowmeow"; //WiFi Password

const char* mqttServer = "walle.cloud.shiftr.io";
const char* mqttUserName = "walle"; // MQTT username
const char* mqttPwd = "Lxh8LfIOtu79daE4"; // MQTT password
const char* clientID = "esp_car"; // client id

#define sub0 "forward"
#define sub1 "backward"
#define sub2 "left"
#define sub3 "right"


WiFiClient espClient;
PubSubClient client(espClient);

unsigned long lastMsg = 0;
#define MSG_BUFFER_SIZE  (80)
char msg[MSG_BUFFER_SIZE];
int value = 0;

void setup_wifi() {
 delay(10);
 WiFi.begin(ssid, password);
 while (WiFi.status() != WL_CONNECTED) {
 delay(500);
 Serial.print(".");
 }
 Serial.println("");
 Serial.println("WiFi connected");
 Serial.println("IP address: ");
 Serial.println(WiFi.localIP());
}

void reconnect() {
 while (!client.connected()) {
 if (client.connect(clientID, mqttUserName, mqttPwd)) {
 Serial.println("MQTT connected");
      // ... and resubscribe
      client.subscribe(sub0);
      client.subscribe(sub1);
      client.subscribe(sub2);
      client.subscribe(sub3);
    } 
    else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");

  // Topic 1: forward
  if (strstr(topic, sub0))
  {
    if ((char)payload[0] == '1') {
      esp.write('1');     
    }     
  }

  // Topic 2: backward
  else if (strstr(topic, sub1))
  {
    if ((char)payload[0] == '1') {
      esp.write('2');
    }   
  }

  // Topic 3: turn_left
  else if (strstr(topic, sub2))
  {
    if ((char)payload[0] == '1') {
      esp.write('3');
    } 
  }

  // Topic 4: turn_right
  else if (strstr(topic, sub3))
  {
    if ((char)payload[0] == '1') {
      esp.write('4');
    }
  }  
  
  else
  {
    Serial.println("unsubscribed topic");
  }
}



void setup() {
  Serial.begin(9600);
  esp.begin(9600);
  setup_wifi();
  client.setServer(mqttServer, 1883);
  client.setCallback(callback);
}


void loop() {
  
  if (!client.connected()) {
    
    reconnect();
  }
  client.loop();
}
