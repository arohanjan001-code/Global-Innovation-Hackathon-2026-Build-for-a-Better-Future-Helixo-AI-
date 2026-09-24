/*
 * ============================================
 *  Helixo AI — Innovation Stall Demo
 *  Arduino UNO — LED Control via Serial
 * ============================================
 *
 *  This sketch receives commands from the Helixo AI
 *  website (via Web Serial API over USB) and controls
 *  the built-in LED on Pin 13.
 *
 *  Commands:
 *    'H' → Helmet Detected   → LED ON  (Ignition Enabled)
 *    'N' → No Helmet         → LED OFF (Ignition Cut)
 *
 *  Board: Arduino UNO
 *  Baud Rate: 9600
 * ============================================
 */

const int LED_PIN = 13;   // Built-in LED on Arduino UNO

void setup() {
  // Initialize Serial at 9600 baud
  Serial.begin(9600);

  // Set LED pin as output
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Startup blink sequence (3 blinks = "I'm alive!")
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }

  // Send ready message
  Serial.println("HELIXO_READY");
}

void loop() {
  // Check if data is available on Serial
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'H') {
      // Helmet detected → Turn LED ON (Ignition Enabled)
      digitalWrite(LED_PIN, HIGH);
      Serial.println("LED_ON");
    }
    else if (command == 'N') {
      // No helmet → Turn LED OFF (Ignition Cut)
      digitalWrite(LED_PIN, LOW);
      Serial.println("LED_OFF");
    }
    // Ignore any other characters (newlines, etc.)
  }
}
