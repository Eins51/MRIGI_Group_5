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

