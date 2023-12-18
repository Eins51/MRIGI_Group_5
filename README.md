# MRIGI_Group_5_Endoscopic Balloon with Integrated Time of Flight (ToF) Sensing

This repository contains a set of tools for collecting and visualizing sensor data in near real-time. The toolset includes scripts for data collection, preprocessing, and 3D visualization, running in a multiprocessing environment for efficient data handling and presentation.

## Contents

- `data_collection.py` - Script for collecting data from sensors and storing it in a specified format.
- `data_visualization.py` - Script for near real-time 3D visualization of the collected sensor data.
- `both_processes.py` - Main script that orchestrates the data collection and visualization processes.

## Installation

Ensure you have Python 3.8 installed. Install the required dependencies by running:

```bash
pip install serial time shutil os numpy pandas pyvista scipy pillow keyboard multiprocessing
```

## Usage

To run the sensor data collection and visualization toolset, follow these steps:

1. Start the main script:

    ```bash
    python both_processes.py
    ```

2. Ensure that the sensor data files are formatted correctly and placed in the 'data' directory. The visualization script will process these files in near real-time.

3. The data collection process will start first, followed by the data visualization process after a short delay.

