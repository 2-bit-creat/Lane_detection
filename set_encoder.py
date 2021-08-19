import cv2
import serial
import time
import numpy as np
import ctypes

def writeBuffer(data, speed, _steer):
    direction = 0

    speed = np.uint16(speed)
    _steer = np.uint16(_steer)
    #print(speed)
    speed_Lo = speed & 0xFF
    speed_Hi = speed >> 8
    #print(speed_Hi, speed_Lo)
    steer_Lo = _steer & 0xFF
    steer_Hi = _steer >> 8

    sum = direction + speed_Lo + speed_Hi + steer_Lo + steer_Hi + 220 + 5 + 10 + 13
    clc = np.uint8(~sum)

    data.append(0x53)
    data.append(0x54)
    data.append(0x58)
    data.append(direction)
    data.append(speed_Lo)
    data.append(speed_Hi)
    data.append(steer_Lo)
    data.append(steer_Hi)
    data.append(0xDC)
    data.append(0x05)
    data.append(0x00)
    data.append(0x0D)
    data.append(0x0A)
    data.append(clc)


def serWrite(ser, _speed, _steer):
    data = []
    writeBuffer(data, _speed, _steer)

    for i in range(0, len(data), 1):
        data[i] = np.uint8(data[i])
    print(data)
    ser.write(data)
    cv2.waitKey(25)

# def run_motor(steer, flag):
#     ser = serial.Serial("COM3", 115200)
#     _st = 0
#     _sp = 0

def set_cam():
    cam = cv2.VideoCapture(int(_index)+cv2.CAP.DSHOW)

if __name__ == "__main__":
    seri = serial.Serial("COM3", 115200)
    cap = cv2.VideoCapture()
    # while True:
    #     _, frame = cap.read()
    #     cv2.imshow("image", frame)
    #
    #     if cv2.waitKey(25) == ord('q'):
    #         break

    while True:
        serWrite(seri, 300, 1700)









