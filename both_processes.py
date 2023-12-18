"""
Sensor Data Collection and Visualization Orchestrator

This script serves as an orchestrator for simultaneously running data collection and 
data visualization processes in a multiprocessing environment. It initiates and 
manages separate processes for data collection and visualization, ensuring that both 
operations run concurrently and efficiently.

Author: Yi Wang
Copyright: 2023, Yi Wang, Group 5, MRIGI, Imperial College London

Dependencies:
- Python 3.8
- multiprocessing for running parallel processes
- data_collection module (a custom module for collecting sensor data)
- data_visualization module (a custom module for visualizing sensor data)
- time for handling time-related functions

Usage:
Run this script in a Python environment where the data_collection and data_visualization 
modules are available. The script starts two processes: one for data collection and 
another for data visualization. The data collection process gathers sensor data and 
stores it in a specified directory. The data visualization process reads the stored 
data and visualizes it in near real-time. Ensure that both custom modules are correctly 
implemented and accessible to this script.

"""


import multiprocessing
import data_collection
import data_visualization
import time

if __name__ == "__main__":
    data_collection_process = multiprocessing.Process(target=data_collection.main_function)
    data_collection_process.start()

    time.sleep(1)

    data_visualization_process = multiprocessing.Process(target=data_visualization.visualize_data, args=("data",))
    data_visualization_process.start()

    data_collection_process.join()
    data_visualization_process.join()

