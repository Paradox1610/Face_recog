import serial

try:
    arduino = serial.Serial('COM9', baudrate=9600, timeout=1)  # Replace 'COM9' with your Arduino port
    print("Successfully connected to Arduino!")
except Exception as e:
    print("Error:", e)
