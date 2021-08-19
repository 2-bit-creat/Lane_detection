import cv2
import serial
import time
import numpy as np
#from multiprocessing import Process, Manager, Queue


def run():
    seri = serial.Serial("COM3", 115200)
    print(serial.Serial)
    while True:
        data = seri.read(23).hex()
        print(data)
        data = ["{0}{1}".format(data[i], data[i + 1]) for i in range(0, len(data), 2)]



        sp_ll = int(data[11], 16)
        sp_lh = int(data[12], 16)
        sp_hl = int(data[13], 16)
        sp_hh = int(data[14], 16)
        _arr = [sp_ll, sp_lh, sp_hl, sp_hh]
        sp_l = (sp_lh << 8) | sp_ll
        sp_h = (sp_hh << 8) | sp_hl
        sp = (sp_h << 16) | sp_l

        #print("encoder = {0}".format(sp))
        print(sp)



if __name__ == "__main__":
    run()

# def run():
#     seri = serial.Serial("COM3", 115200)
#     while True:
#         data = seri.read(23).hex()
#         data = ["{0}{1}".format(data[i], data[i + 1]) for i in range(0, len(data), 2)]
#         data = [int(data[i], 16) for i in range(0, len(data), 1)]
#         print(data[i] for i in range())
#
#
#
# if __name__ == "__main__":
#     run()