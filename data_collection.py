"""
ToF Sensor Data Collection and Storage Tool

This script is designed to interface with a serially-connected ToF sensor system.
It allows for real-time data collection from up to 8 sensors, handles the user
interaction for selecting a sensor, and saves the collected data in a timestamped 
file format.

Author: Yi Wang
Copyright: 2023, Yi Wang, Group 5, MRIGI, Imperial College London

Dependencies:
- Python 3.8
- PySerial library for serial communication
- Keyboard library for capturing keypresses
- OS and shutil libraries for file and directory management

Usage:
Run this script directly in a Python environment. Ensure that the connected
serial device is correctly configured at 'COM8'. Use numeric keys (1-8) to select
a sensor, or '0' to cycle through all sensors. Collected data will be stored in
the 'data' folder of the script's directory.

"""


import serial
import time
import os
import keyboard
import shutil


def update_sensor_index():
    for i in range(1, 9): 
        if keyboard.is_pressed(str(i)):
            return i
    if keyboard.is_pressed('0'):
        return 0
    return None


def clear_folder(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


def save_data_to_file(sensor_index, data):
    timestamp = round(time.time(), 3)
    path = "data"
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'{path}/{timestamp}_sensor_data_{sensor_index}.txt'
    with open(filename, 'w') as file:
        for row in data:
            file.write('\t'.join(map(str, row)) + '\n')

    print(f'Data saved to {filename}')
    

def main_function():
    try:
        initialized = 0
        eight_sensor = []
        each_sensor = []
        start_time = time.time()
        count = 1
        ser = serial.Serial('COM8', 115200, timeout=0.01)
        sensor_index = 0
        save_sensor_index = 0
        clear_folder("data")

        while True:
            new_index = update_sensor_index()
            if new_index is not None:
                sensor_index = new_index

            sensor_string = f"Sensor {sensor_index}" if sensor_index != 0 else "Sensor"

            line = ser.readline().decode('iso-8859-1').strip()
            waiting_time = round(time.time() - start_time)
            if waiting_time >= count and initialized != 2:
                count = count + 1
                print("Waiting time: " + str(waiting_time))
            if len(line) != 0:
                if sensor_index == 0:
                    print(line)
                if line == "Mux detected":
                    initialized = 1
                if initialized != 0 and line.startswith(sensor_string):
                    if sensor_index != 0:
                        print(line)
                    initialized = 2
                    save_sensor_index = int(line[7:])
                if initialized == 2 and line[0].isdigit():
                    if sensor_index != 0:
                        print(line)
                    data_row = list(map(int, line.split('\t')[0:]))
                    each_sensor.append(data_row)   
            if len(each_sensor) == 8:
                print(time.time())
                save_data_to_file(save_sensor_index, each_sensor)
                each_sensor = []
                initialized = 1

    except KeyboardInterrupt:
        ser.close()
        print("Program terminated by user. Serial port closed.")


if __name__ == "__main__":
    main_function()

