"""
ToF Sensor Data Visualization Tool

This script is designed for near real-time visualization of sensor data in a 3D environment. 
It processes and interpolates sensor data from multiple sources, generating 3D surface
reconstruction and visual representations. The script supports dynamic updates based on
new sensor data files and allows for interactive control of the visualization parameters.

Author: Yi Wang
Copyright: 2023, Yi Wang, Group 5, MRIGI, Imperial College London

Dependencies:
- Python 3.8
- NumPy for numerical operations
- Pandas for data manipulation
- PyVista for 3D visualization
- SciPy for scientific computations and interpolations
- Pillow for image processing
- Keyboard for capturing keypresses
- Time for handling time-related functions
- OS for file and directory operations
- IO for handling I/O operations

Usage:
Run this script in a Python environment where all dependencies are installed. The script 
listens for new sensor data files in the specified directory and updates the 3D 
visualization accordingly. Interactive controls are available for manipulating the 
visualization during runtime. Ensure the sensor data files are formatted correctly 
and placed in the 'data' directory. 

"""


import os
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import griddata
import time
from PIL import Image
import io
import keyboard


def read_sensor_data_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data_row = list(map(int, line.strip().split('\t')))
            data_row = [x/10 for x in data_row]
            data.append(data_row)
    return data


def median_threshold(df, cell, window, threshold):
    i, j = cell
    start_row, end_row = max(0, i - window), min(df.shape[0], i + window + 1)
    start_col, end_col = max(0, j - window), min(df.shape[1], j + window + 1)

    window_values = df.iloc[start_row:end_row, start_col:end_col]
    median_val = window_values.median().median()

    return df.at[i, j] if abs(df.at[i, j] - median_val) <= threshold else np.nan


def preprocess_data(data):
    df = pd.DataFrame(data)
    df = df.reindex(columns=range(8), method='nearest')
##    print(df)
    ## When the sensor measurement distance is less than a certain value, the output value will be large
    df[(df > 10) & (df < 100)] = np.nan
    df[(df > 100)] = 0
##    print(df)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            df.at[i, j] = median_threshold(df, (i, j), 3, 1)

##    print(df)

    x, y = np.indices(df.shape)
    values = df.values.flatten()
    points = np.column_stack((x.flatten(), y.flatten()))
    known_points = points[~np.isnan(values)]
    known_values = values[~np.isnan(values)]
    grid_x, grid_y = np.indices(df.shape)
    grid_z = griddata(known_points, known_values, (grid_x, grid_y), method='linear')
    if np.any(np.isnan(grid_z)):
        grid_z = griddata(known_points, known_values, (grid_x, grid_y), method='nearest')
    if np.any(np.isnan(grid_z)):
        avg_val = np.nanmean(grid_z)
        grid_z = np.nan_to_num(grid_z, nan=avg_val)
    df = pd.DataFrame(grid_z, columns=df.columns)
    
##    df.interpolate(method='linear', axis=1, inplace=True)
##    print(df)
    return df.values


def generate_ellipse_surface():
    center = (0.0, 0.0)
    major_axis = 3.0
    minor_axis = 1.5

    x, y = np.meshgrid(range(8), range(8))
    distance_data = np.sqrt(((x - center[0]) / major_axis) ** 2 + ((y - center[1]) / minor_axis) ** 2)

    return distance_data


def eight_sensor_blocks():
    rectangle_length = 1.3
    rectangle_width = 0.65
    rectangle_thickness = 0.05
    sensor_blocks = []

    for i in range(8):
        angle = i * (360 / 8)
        center_x = (rectangle_width / 2 + rectangle_width * np.cos(np.deg2rad(45))) * np.cos(np.deg2rad(angle))
        center_y = (rectangle_width / 2 + rectangle_width * np.cos(np.deg2rad(45))) * np.sin(np.deg2rad(angle))
        center_z = 0

        box = pv.Box(bounds=(-rectangle_width / 2, rectangle_width / 2, 
                             -rectangle_thickness / 2, rectangle_thickness / 2, 
                             -rectangle_length / 2, rectangle_length / 2))
        if i % 2 == 0:
            box = box.rotate_z(-angle + 90)
        else:
            box = box.rotate_z(-angle)
        box = box.translate((center_x, center_y, center_z))
        sensor_blocks.append(box)
        
    return sensor_blocks


def convert_to_point_cloud(data):
    rows, cols = data.shape
    points = []

    for i in range(rows):
        for j in range(cols):
            distance = data[i, j]
            points.append([(i-3.5)*0.343, (j-3.5)*0.343, distance+1.81])

    points = np.array(points)
    return points


def transform(points, translation_vector, rotation_vector):
    homogeneous_points = np.column_stack((points, np.ones(points.shape[0])))
    translation_matrix = [
        [1, 0, 0, translation_vector[0]],
        [0, 1, 0, translation_vector[1]],
        [0, 0, 1, translation_vector[2]],
        [0, 0, 0, 1 ]
    ]
    
    alpha, beta, gamma = np.deg2rad(rotation_vector[3:])
##    Rx = np.array([
##        [1, 0, 0, 0],
##        [0, np.cos(alpha), -np.sin(alpha), 0],
##        [0, np.sin(alpha), np.cos(alpha), 0],
##        [0, 0, 0, 1]
##    ])
##    Ry = np.array([
##        [np.cos(beta), 0, np.sin(beta), 0],
##        [0, 1, 0, 0],
##        [-np.sin(beta), 0, np.cos(beta), 0],
##        [0, 0, 0, 1]
##    ])
##    Rz = np.array([
##        [np.cos(gamma), -np.sin(gamma), 0, 0],
##        [np.sin(gamma), np.cos(gamma), 0, 0],
##        [0, 0, 1, 0],
##        [0, 0, 0, 1]
##    ])
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha), rotation_vector[1]*(1 - np.cos(alpha)) + rotation_vector[2]*np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha), rotation_vector[2]*(1-np.cos(alpha)) - rotation_vector[1]*np.sin(alpha)],
        [0, 0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta), rotation_vector[0]*(1-np.cos(beta))-rotation_vector[2]*np.sin(beta)],
        [0, 1, 0, 0],
        [-np.sin(beta), 0, np.cos(beta), rotation_vector[2]*(1-np.cos(beta))+rotation_vector[0]*np.sin(beta)],
        [0, 0, 0, 1]
    ])
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0, rotation_vector[0]*(1-np.cos(gamma))+rotation_vector[1]*np.sin(gamma)],
        [np.sin(gamma), np.cos(gamma), 0, rotation_vector[1]*(1-np.cos(gamma))-rotation_vector[0]*np.sin(gamma)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    rotated_points = np.dot(np.dot(np.dot(Rx, Ry), Rz), homogeneous_points.T).T
    transformed_points = np.dot(translation_matrix, rotated_points.T).T
    transformed_points = transformed_points[:, :3]

    return transformed_points


def grid_transform(grid, translation_vector, rotation_vector):
    translation_matrix = np.array([
        [1, 0, 0, translation_vector[0]],
        [0, 1, 0, translation_vector[1]],
        [0, 0, 1, translation_vector[2]],
        [0, 0, 0, 1 ]
        ])
    alpha, beta, gamma = np.deg2rad(rotation_vector[3:])
    Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1]
        ])
    Ry = np.array([
            [np.cos(beta), 0, np.sin(beta), 0],
            [0, 1, 0, 0],
            [-np.sin(beta), 0, np.cos(beta), 0],
            [0, 0, 0, 1]
        ])
    Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0, 0],
            [np.sin(gamma), np.cos(gamma), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    combined_matrix = np.dot(translation_matrix, Rx)
    combined_matrix = np.dot(combined_matrix, Ry)
    combined_matrix = np.dot(combined_matrix, Rz)
    grid.transform(combined_matrix)


def interpolate_to_surface(data):
    start = -1.13
    stop = 1.13
    num_points = data.shape[0]
    step = (stop - start) / (num_points - 1)

    x = np.arange(start, stop + step, step)
    y = np.arange(start, stop + step, step)
    X, Y = np.meshgrid(x, y)
    Z = griddata((X.flatten(), Y.flatten()), data.flatten(), (X, Y), method='linear')
    Z += 1.81
    return X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32)
##    X = data[:, 0]
##    Y = data[:, 1]
##    Z = data[:, 2]
##    grid_x, grid_y = np.mgrid[min(X):max(X):100j, min(Y):max(Y):100j]
##    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='linear')
##    return grid_x, grid_y, grid_z


def update(frame, plotter):
    plotter.clear()
    eight_sensors = eight_sensor_blocks()
    eight_points = [];
    for i in range(8):
        plotter.add_mesh(eight_sensors[i], color='red')
    for i in range(8 * frame, 8 * frame + 8):
        filename = os.path.join(folder_path, files[i])
        raw_data = read_sensor_data_file(filename)
        new_data = preprocess_data(raw_data)
        points = convert_to_point_cloud(new_data)
        translation_vector = [0.0, 0.0, 0.0]
        rotation_vector = [0.0, 0.0, 0.0, 0, 90, 0]
        points = transform(points, translation_vector, rotation_vector)
        rotation_vector = [0.0, 0.0, 0.0, 0, 0, 45 * i]
        points = transform(points, translation_vector, rotation_vector)
    
        X, Y, Z = interpolate_to_surface(new_data)
        grid = pv.StructuredGrid(X, Y, Z)
        
        eight_points.append(points)
    eight_points = np.concatenate(eight_points, axis=0)
    plotter.add_points(eight_points, color='blue', point_size=8.0, render_points_as_spheres=True)
    
##    plotter.add_mesh(surf, color='blue', smooth_shading=True)
##        plotter.add_mesh(grid, scalars=Z.ravel(), cmap='viridis', show_scalar_bar=True)
##    plotter.show_grid()
##    plotter.set_background('white')
    
    timeRecord = time.time()
##    plotter.add_title(f'Time: {timeRecord}', font_size=18, color=None, font=None, shadow=False)


# Determine the boundary points of each grid
def find_boundary_column(grid, boundary='right'):
    center = np.array([0, 0, 0])
    angles = np.arctan2(grid.points[:, 1] - center[1], grid.points[:, 0] - center[0])

    points = grid.points
    column_points = []
    if boundary == 'right':
##        column_points = points[:8]
        for i in range(8):
            point = points[i * 8]
            column_points.append(point.tolist())
    else:
##        column_points = points[-8:]
        for i in range(8):
            point = points[i * 8 + 7]
            column_points.append(point.tolist())

    return np.array(column_points)


def create_bridge_between_grids(grid1, grid2):
    right_column = find_boundary_column(grid1, 'right')
    left_column = find_boundary_column(grid2, 'left')
##    plotter.add_points(right_column)
##    plotter.add_points(left_column)
    num_points = min(len(right_column), len(left_column))
    bridge_points = np.vstack([right_column, left_column])
    faces = []
    for i in range(num_points - 1):
        faces.extend([4, i, i + num_points, i + num_points + 1, i + 1])
    faces = np.array(faces, dtype=np.int_)
    bridge = pv.PolyData()
    bridge.points = bridge_points
    bridge.faces = faces

    return bridge


def plot_cylinder(up, bottom):
    height = 2
    long_diameter = 6.8
    short_diameter = 6.8
    num_columns = 64
    a = long_diameter / 2
    b = short_diameter / 2
    z_shift = height / 2 + 1.5
    cylinder_polydata = pv.PolyData()

    h = ((a - b)**2) / ((a + b)**2)
    circumference = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
    delta = circumference / num_columns
    num_rows = int(height / delta)

    theta = np.linspace(0, 2 * np.pi, num_columns, endpoint=False)
    z = np.linspace(-height / 2 + z_shift, height / 2 + z_shift, num_rows, endpoint=True)
    Theta, Z = np.meshgrid(theta, z)
    X = a * np.cos(Theta)
    Y = b * np.sin(Theta)
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    
    cells = []
    for i in range(num_rows - 1):
        for j in range(num_columns):
            p1 = i * num_columns + j
            p2 = p1 + num_columns
            p4 = i * num_columns + (j + 1) % num_columns
            p3 = p2 + 1 - num_columns if j == num_columns - 1 else p2 + 1
            
            cells.append([3, p1, p2, p3])
            cells.append([3, p1, p3, p4])

    mesh = pv.PolyData(points, np.hstack(cells))
    cylinder_polydata = cylinder_polydata.merge(mesh)

    connect_points = np.array(points[:64].tolist())
    shift = 36
    connect_points = np.concatenate((connect_points[shift:], connect_points[:shift]))

##    plotter.add_points(connect_points[:8], color = 'blue')
    cells_up_to_connect = []
    num_points = len(connect_points)
    for i in range(num_points):
        p1 = i
        p2 = (i + 1) % num_points
        p3 = num_points + (i + 1) % num_points
        p4 = num_points + i

        cells_up_to_connect.append([3, p1, p2, p3])
        cells_up_to_connect.append([3, p1, p3, p4])

    combined_points = np.vstack((up, connect_points))
    mesh_up_to_connect = pv.PolyData(combined_points, np.hstack(cells_up_to_connect))
    cylinder_polydata = cylinder_polydata.merge(mesh_up_to_connect)
##    plotter.add_mesh(mesh_up_to_connect, color='black', style='wireframe')

    
    z = np.linspace(-height / 2 - z_shift, height / 2 - z_shift, num_rows, endpoint=True)
    Theta, Z = np.meshgrid(theta, z)
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    cells = []
    for i in range(num_rows - 1):
        for j in range(num_columns):
            p1 = i * num_columns + j
            p2 = p1 + num_columns
            p4 = i * num_columns + (j + 1) % num_columns
            p3 = p2 + 1 - num_columns if j == num_columns - 1 else p2 + 1
            
            cells.append([3, p1, p2, p3])
            cells.append([3, p1, p3, p4])
    connect_points = np.array(points[-64:].tolist())
##    plotter.add_points(connect_points)

    shift = 36
    connect_points = np.concatenate((connect_points[shift:], connect_points[:shift]))
    cells_up_to_connect = []
    num_points = len(connect_points)
    for i in range(num_points):
        p1 = i
        p2 = (i + 1) % num_points
        p3 = num_points + (i + 1) % num_points
        p4 = num_points + i

        cells_up_to_connect.append([3, p1, p2, p3])
        cells_up_to_connect.append([3, p1, p3, p4])

    combined_points = np.vstack((connect_points, bottom))
    mesh_up_to_connect = pv.PolyData(combined_points, np.hstack(cells_up_to_connect))
    cylinder_polydata = cylinder_polydata.merge(mesh_up_to_connect)
##    plotter.add_mesh(mesh_up_to_connect, color='black', style='wireframe')


    mesh = pv.PolyData(points, np.hstack(cells))
    cylinder_polydata = cylinder_polydata.merge(mesh)
##    plotter.add_mesh(cylinder_polydata, color='black', opacity=0.5)
    return cylinder_polydata
    

def visualize_data(folder_path):
 
    processed_files = set()
    plotter = pv.Plotter()
    gif_filename = f"animation.gif"
    plotter.open_gif(gif_filename)
    mesh1_added = False
    mesh2_added = False
    update_mesh2 = True
    frame_count = 0
    last_time = time.time()
    text_actor = None
    hz_actor = None

    
    while True:
        
        if keyboard.is_pressed('y'):
            update_mesh2 = True
        elif keyboard.is_pressed('n'):
            update_mesh2 = False
            
        files = [filename for filename in os.listdir(folder_path) if filename.endswith('.txt')]
        new_files = [f for f in files if f not in processed_files]
        if len(new_files) >= 8:
            count_files = 0
            eight_grids = []
            up_points = []
            bottom_points = []
            for file in new_files:
                count_files += 1
                if count_files <= 8:
                    filename = os.path.join(folder_path, file)
                    raw_data = read_sensor_data_file(filename)
                    new_data = preprocess_data(raw_data)
                    translation_vector = [0.0, 0.0, 0.0]
                    X, Y, Z = interpolate_to_surface(new_data)
                    grid = pv.StructuredGrid(X, Y, Z)
                    rotation_vector = [0.0, 0.0, 0.0, 0, -90, 0]
                    combined_matrix = grid_transform(grid, translation_vector, rotation_vector)
                    rotation_vector = [0.0, 0.0, 0.0, 0, 0, 45 * count_files]
                    combined_matrix = grid_transform(grid, translation_vector, rotation_vector)
                    up_points.append(grid.points[-8:].tolist()[::-1])
                    bottom_points.append(grid.points[:8].tolist()[::-1])
                    eight_grids.append(grid)
                    processed_files.add(file)
                else:
                    break
                
            if len(eight_grids) == 8:
                combined_polydata = pv.PolyData()
                for i in range(len(eight_grids) - 1):
                    current_grid = eight_grids[i]
                    next_grid = eight_grids[i + 1]
                    bridge = create_bridge_between_grids(current_grid, next_grid)
                    combined_polydata = combined_polydata.merge(current_grid.extract_surface())
                    combined_polydata = combined_polydata.merge(bridge)
                    if i == len(eight_grids) - 2:
                        combined_polydata = combined_polydata.merge(next_grid.extract_surface())
                final_bridge = create_bridge_between_grids(eight_grids[-1], eight_grids[0])
                combined_polydata = combined_polydata.merge(final_bridge)

                up_points_array = np.vstack(up_points)
                bottom_points_array = np.vstack(bottom_points)
                cylinder_polydata = plot_cylinder(up_points_array, bottom_points_array)
##                combined_polydata = combined_polydata.merge(cylinder_polydata)
                
                distances = np.sqrt(combined_polydata.points[:, 0]**2 + combined_polydata.points[:, 1]**2)
                combined_polydata['Distances'] = distances
                
                
                if not mesh1_added:
                    combined_polydata = combined_polydata
                    mesh_actor = plotter.add_mesh(combined_polydata, scalars="Distances", cmap='viridis', show_scalar_bar=True)
                    mesh1_added = True
                elif mesh1_added:
                    combined_polydata = combined_polydata
                    mesh_actor.GetMapper().SetInputData(combined_polydata)
                    mesh_actor.GetMapper().Update()

                if update_mesh2:
                    cylinder_polydata = cylinder_polydata
                    if not mesh2_added:
                        mesh_actor2 = plotter.add_mesh(cylinder_polydata, color='grey', opacity=1)
                        mesh2_added = True
                    elif mesh2_added:
                        cylinder_polydata = cylinder_polydata
                        mesh_actor2.GetMapper().SetInputData(cylinder_polydata)
                        mesh_actor2.GetMapper().Update()
                else:
                    if mesh2_added and mesh_actor2 is not None:
                        plotter.remove_actor(mesh_actor2)
                        mesh_actor2 = None
                        mesh2_added = False

                current_time = time.time()
                elapsed_time = current_time - last_time
                last_time = current_time
                hz = 1.0 / elapsed_time if elapsed_time else 0

                if text_actor:
                    plotter.remove_actor(text_actor)
                if hz_actor:
                    plotter.remove_actor(hz_actor)

                frame_text = f"Frame: {frame_count}"
                text_actor = plotter.add_text(frame_text, position='upper_left', font_size=10, color='black')

                hz_text = f"Hz: {hz:.2f}"
##                hz_actor = plotter.add_text(hz_text, position='upper_right', font_size=10, color='black')

                plotter.write_frame()
                frame_count += 1


if __name__ == "__main__":
    visualize_data("data")
    



