import serial
import time

# Change this to your ESP32/Arduino COM port
PORT = "COM3"
BAUD_RATE = 9600

robot = None


def connect_robot():
    global robot

    try:
        robot = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        return True

    except Exception as e:
        print("Robot connection error:", e)
        return False


def send_command(command):
    if robot is not None and robot.is_open:
        robot.write((command + "\n").encode())
        return f"Command sent: {command}"

    return "Robot is not connected"


def disconnect_robot():
    global robot

    if robot is not None and robot.is_open:
        robot.close()
        return "Robot disconnected"

    return "Robot already disconnected"