import os
import sys
import math
import random
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to generate RFID simulation data from self-defined robot trajectory and tags locations.

class RFIDGenerator(object):
    '''
    Define parameters:
        * robot trajectory
        * tags location
        * RFID parameters (e.g. lamda, reading range)
    '''
    def __init__(self, trajectory, location, noise, multipath):
        self.c_speed = 3e8              # the speed of light = 3e8 m/s
        self.frequency = 920.625e6      # frequency of RFID signal = 920.625 MHz
        self.reading_range = 10         # the radius of readable region = 10 m
        self.readable_angle = 120       # the angle of reading range = 120 degrees
        self.c = 0.0                    # the random phase offset, generated randomly
        self.ants_trajectory = trajectory
        self.location = np.array(location)
        self.noise = noise
        self.multipath = multipath
        self.data = []
        self.lamda = self.c_speed / self.frequency
       

    '''
    Generate RFID simulation data:
    * pure data
    * add noise & multipath 
    '''
    def generate(self, root_path, folder):
        
        if self.noise == None:
            if self.multipath == None:
                # the user choose pure mode to generate RFID data
                ant_num = self.ants_trajectory.shape[0]
                trajectory_len = self.ants_trajectory.shape[1]
                # print([ant_num, trajectory_len])
                
                # Generate data for each antenna
                data = []

                for i in range(ant_num):
                    value_ant_trajectory = []
                    # Choose possible data which satisfy the given reading range
                    for j in range(trajectory_len):
                        ant_pos = self.ants_trajectory[i,j,0:3]
                        if j < trajectory_len - 1:
                            ant_orientation = self.ants_trajectory[i,j+1,0:3]-self.ants_trajectory[i,j,0:3]
                        else:
                            ant_orientation = self.ants_trajectory[i,j,0:3]-self.ants_trajectory[i,j-1,0:3]
                        tag_ant_vector = self.location - ant_pos
                        # Calculate the distance between points
                        distance = np.linalg.norm(tag_ant_vector) 
                        if distance < 10:
                            cos_theta = np.dot(ant_orientation, tag_ant_vector) / (np.linalg.norm(ant_orientation) * np.linalg.norm(tag_ant_vector))
                            if cos_theta < (math.sqrt(3)/2) and cos_theta > (-math.sqrt(3)/2):
                                # Satisfy conditions
                                value_ant_trajectory.append(self.ants_trajectory[i,j,:])
                            else:
                                value_ant_trajectory.append(np.zeros(self.ants_trajectory[i,j,:].shape))
                                continue
                        else:
                            value_ant_trajectory.append(np.zeros(self.ants_trajectory[i,j,:].shape))
                            continue
                    
                    # Use value trajectory to calculate phase
                    c = random.uniform(0.0, 2*math.pi)
                    if len(value_ant_trajectory) > 0:
                        value_ant_trajectory = np.array(value_ant_trajectory)
                        tag_ant_vectors = value_ant_trajectory[:,0:3] - self.location
                        distances = np.linalg.norm(tag_ant_vectors, axis=1)
                        phase = 4 * math.pi * distances / self.lamda + c
                        data.append(value_ant_trajectory[:,0])
                        data.append(value_ant_trajectory[:,1])
                        data.append(value_ant_trajectory[:,2])
                        data.append(phase)
                        data.append(value_ant_trajectory[:,3])
                  
                # Simulate interpolation
                data = np.transpose(np.array(data))
                # Choose the best two antennas
                zero_counts = np.zeros(3)
                for i in range(3):
                    this_data = data[:,i*5:(i+1)*5]
                    zero_counts[i] = (~(this_data == 0).all(axis=1)).sum()
                  
                # argsort：get Indexs in descending order 
                sorted_indices = np.argsort(zero_counts)[::-1]  # [::-1] 是为了降序排列  
                # get the biggest two indexs
                top2_indices = sorted_indices[:2]
                data1 = data[:,top2_indices[0]*5:(top2_indices[0]+1)*5]
                data2 = data[:,top2_indices[1]*5:(top2_indices[1]+1)*5]
                data = np.concatenate((data1, data2), axis=1)
                
                # Filter zero values
                zero_count = np.sum(data == 0, axis=1)
                data = data[zero_count <= 3, :]
                self.data = data
                  
                # save data into target folder directory
                # print(len(self.data))
                folder_path = os.path.join(root_path,str(folder))
                self.write_csv(folder_path)
        

    '''
    Write generated RFID data into a .csv file
    '''
    def write_csv(self,folder_path):
        # define the target dir and file path
        if not os.path.exists(folder_path):  
            os.makedirs(folder_path) 
        file_path = os.path.join(folder_path,'data.csv') 
        
        # convert numpy array into pandas DataFrame  
        df1 = pd.DataFrame(self.data)
        # add names of columns
        df1.columns = ['Ant1X', 'Ant1Y', 'Ant1Z','Ant1Phase','Ant1Timestamp','Ant2X', 'Ant2Y', 'Ant2Z','Ant2Phase','Ant2Timestamp']
        
        # write DataFrame into .csv
        df1.to_csv(file_path, index=False)
        print(['write file:', file_path, df1.shape])
        
        label_path = os.path.join(folder_path,'label.csv')
        df2 = pd.DataFrame(self.location).transpose()
        df2.columns = ['TagX','TagY','Tagz']
        df2.to_csv(label_path, index=False)

def generateTrajectory(start_point, end_point, distance_interval):   
    # Calculate the vector between points  
    vector = end_point - start_point
    # Calculate the distance between points
    total_distance = np.linalg.norm(vector)  
    # Calculate the number of points need to be generated 
    num_points = int(np.ceil(total_distance / distance_interval)) + 1  
    # Calculate the actual interval between points
    actual_spacing = vector / (num_points - 1)  
    # Generated points with the same interval
    points = np.array([start_point + i * actual_spacing for i in range(num_points)])  
    return points


if __name__ == '__main__':
    '''
    Define the position of RFID 
    ''' 
    possible_tags_range = np.array([[4,6],[0,0.5],[0.3,2]])
    tag_x_down = 4
    tag_x_up = 6
    tag_y_down = 0
    tag_y_up = 0.5
    tag_z_down = 0.3
    tag_z_up = 2
    tags_locations = []
    for i in range(20):
        tag_location_x = random.uniform(tag_x_down,tag_x_up)
        tag_location_y = random.uniform(tag_y_down,tag_y_up)
        tag_location_z = random.uniform(tag_z_down,tag_z_up)
        tags_locations.append([tag_location_x,tag_location_y,tag_location_z])
    
    '''
    Define candidate locations and trajectories
    * provide possible locations
    * provide nearby trajectories
    * generate randomly
    '''
    # Define the range of start points and end points
    possible_y = np.arange(0.2, 3.3, 0.5)
    robot_start_points_y = []
    robot_end_points_y = []
    # First decide the y axis of robot (y means beside the robot; x means the direction of the robot)
    for i in range(len(possible_y)-1):
        for ii in range(20):
            start_y = random.uniform(possible_y[i], possible_y[i+1])
            end_y = random.uniform(possible_y[i], possible_y[i+1])
            robot_start_points_y.append(start_y)
            robot_end_points_y.append(end_y)
    
    points_num = len(robot_start_points_y)
    robot_start_points = np.zeros((points_num, 2))
    robot_end_points = np.zeros((points_num, 2))
    robot_start_points[:,1] = robot_start_points_y
    robot_end_points[:,1] = robot_end_points_y
    robot_end_points[:,0] = np.ones(points_num) * 10
    
    # Generate trajectories of robot
    robot_trajectories = []
    sampling_interval = 0.04
    for i in range(points_num):
        robot_trajectory = generateTrajectory(robot_start_points[i,:], robot_end_points[i,:], sampling_interval)
        robot_trajectories.append(robot_trajectory)
    
    # Calculate trajectories of antennas(x,y,z)
    ants_pos = np.array([[0,0,0.4],[0,0,1],[0,0,1.5]])
    trajectories = []
    for i in range(len(robot_trajectories)):
        trajectory = np.array(robot_trajectories[i])
        ant_trajectory = np.ndarray((3,len(trajectory),4))
        for j in range(3):
            ant_trajectory[j,:,0] = trajectory[:,0] + ants_pos[j,0]
            ant_trajectory[j,:,1] = trajectory[:,1] + ants_pos[j,1]
            ant_trajectory[j,:,2] = np.ones(len(trajectory)) * ants_pos[j,2]
            ant_trajectory[j,:,3] = range(len(trajectory))
        trajectories.append(ant_trajectory)
    
    
    # Define mode control parameters
    noise = None
    multipath = None
    root_path = 'C:/Users/yangy/Desktop/GeneratedData'
    
    '''
    Start data generation:
    * Traverse all possible trajectories and label coordinates
    '''
    folder_no = 0
    for trajectory in trajectories:
        for location in tags_locations:
            folder_no += 1
            dataMaker = RFIDGenerator(trajectory, location, noise, multipath)
            dataMaker.generate(root_path,folder_no)
    print([folder_no,' sets of data generated.'])