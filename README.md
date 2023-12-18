# MRIGI_Group_5_Endoscopic Balloon with Integrated Time of Flight (ToF) Sensing

This repository contains a set of tools for collecting and visualizing sensor data in near real-time. The toolset includes scripts for data collection, preprocessing, and 3D visualization, running in a multiprocessing environment for efficient data handling and presentation.

## Contents

- `./Arduino/eight_Sensor/eight_Sensor.ino` - Arduino sketch for interfacing with eight VL53L5CX ToF sensors through a Qwiic Mux.
- `data_collection.py` - Script for collecting data from sensors and storing it in a specified format.
- `data_visualization.py` - Script for near real-time 3D visualization of the collected sensor data.
- `both_processes.py` - Main script that orchestrates the data collection and visualization processes.
- `Demo` - Folder containing demonstration videos and images of the toolset in action.

## Installation

- For the Arduino sketch, ensure you have the Arduino IDE and necessary libraries installed.
- For the python sketch, ensure you have Python 3.8 installed. Install the required dependencies by running:

```bash
pip install serial time shutil os numpy pandas pyvista scipy pillow keyboard multiprocessing
```

## Usage

> ####*Note 1: Arduino Serial Monitor*
>
> *Running the data collection process (`data_collection.py`) and the Arduino Serial Monitor simultaneously may lead to conflicts, as both require access to the same serial port. Ensure that only one of these is accessing the serial port at a time.*

> ####*Note 2: Customization for Different Ports*
>
> *If you are using a different port, make sure to update the port in the code (`ser = serial.Serial('COM8', 115200, timeout=0.01)`, `COM8` in this case) to match your hardware setup.*

> #### *Note 3: Sensor Address Assignment Duration*
>
> *Assigning addresses to the eight sensors may take approximately **one and a half minutes**. Please wait for about 90 seconds after powering up the system before attempting to read sensor data.*

### 1. Arduino Sensor Interface

The `eight_Sensor.ino` sketch is designed to interface with multiple VL53L5CX sensors using a Qwiic Mux. It initializes the sensors, sets their resolution and frequency, and continuously collects distance measurements. The data from each sensor is printed over the serial connection in an array format representing the sensor's field of view.

####Features

- Supports up to 8 VL53L5CX sensors.
- Configurable sensor resolution and ranging frequency.
- Real-time data output in an array format via the serial port.

####Setup

1. Connect the VL53L5CX sensors to the Qwiic Mux.
2. Upload the `eight_Sensor.ino` sketch to your Arduino board.
3. Open the serial monitor to view the sensor data output.

### 2. Main Python Script

To simultaneously run data collection and visualization, follow these steps:

1. Start the main script:

    ```bash
    python both_processes.py
    ```

2. Ensure that the sensor data files are formatted correctly and placed in the 'data' directory. The visualization script will process these files in near real-time.

3. The data collection process will start first, followed by the data visualization process after a short delay.


###3. (Optional) Individual Python Scripts

#### Data Collection Script

- Can be run independently to collect sensor data.
- Allows selection of a single sensor's output or all sensors' data display based on keyboard input.

```
python data_collection.py
```

#### Data Visualization Script

- Can be run independently to visualize pre-collected data.
- Ideal for analyzing and viewing data without active data collection.

```
python data_visualization.py
```
##Demo

![WPS动图制作](E:\Desktop\WPS动图制作.gif)