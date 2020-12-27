import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import torch
import sys
import os 
import math
sys.path.append('C:\\Users\\Bryan\\Documents\\Coding\\github\\Self_Taught_Machine_Learning\\Data_Stuff\\')
for line in sys.path:
    print(line)
import Data_Stuff.data_manipulation

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, input("Enter file path to .csv with formatting like this: ./KNN/enterYourCSVnameHERE: "))

if __name__ == '__main__':
    main()



def main():

    return 0



#Distance from unknown datapoint to arbitrary datapoint in terms of coordinate blocks
#p = 1 
def manhattan_distance(dataset_df, the_row):
    distances = []

    for row in dataset_df:
        dist = minkowski_distance(the_row, row, 1)
        distances.append(row, dist)

    return distances



#Distance from unknown datapoint to arbitrary datapoint
#p = 2
def euclidean_distance(dataset_df, the_row):
    distances = []

    for row in dataset_df:
        dist = minkowski_distance(the_row, row, 2)
        distances.append(row, dist)

    return distances



#p = infinity
def max_distance(dataset_df, the_row):
    distances = []

    for row in dataset_df:
        dist = minkowski_distance(the_row, row, math.inf)
        distances.append((row, dist))

    return distances



#Minkowski distance
#row1 behaves as the arbitrary vector we wish to find distances from
#row2 are all the vectors we will be calculating the distance to
#p_val is the p_val the determines if you are doing manhattan, euclidean or max distance
def minkowski_distance(row1, row2, p_val):
    distance = 0.0

    for i in range(len(row1) - 1): #we want everything but the label associated with vector
         distance += ((row1[i] - row2[i]).abs()) ** p_val #go through each dimension of the vector and subtract it from the other

    the_dist = (distance ** (1.0 / p_val)) # (distance)^(1/p)

    return the_dist



def distance_prediction(train_set, test_instance):
    
    dist_list = euclidean_distance(train_set, test_instance[0])
    sorted_dist_list = dist_list.sort(key = (lambda x: x[1])) #we sort the list based off the value of the distance. this will give us smallest values closer to 0th index

    return sorted_dist_list



def get_k_nearest_neighbors(sorted_dist_list, k_value):
    neighbors = []

    for i in range(k_value):
        neighbors.append(sorted_dist_list[i][0]) # find the k number of values closest to our test_instance in terms of the row associated with the distance

    return neighbors

# def get_most_freq_class(k_neighbors_list):
#     classes_dict = {}

#     for i in range(len(k_neighbors_list)): #for all the classes within the k_neighbors_list we populate a class dictionary  
#         if k_neighbors_list[i][-1] not in classes_dict: #if this particular class is not within the dictionary then
#             classes_dict[k_neighbors_list[i][-1]] =+ 1 #we increment the amount of that particular class that is within the dictionary
#         else:
#             classes_dict[k_neighbors_list[i][-1]] = 1 #if the class is not within the dictionary we assign that there is now 1 of that class now

#     sorted_class_dict = sorted(classes_dict.items(), key=lambda x: x[1], reverse=True) #we now sort the dictionary based on how many of a particular class is within it 

#     return sorted_class_dict[0][0] #we return the class with the most amount of presence to show the K_nearest_neighbor label


def get_most_freq_class(k_neighbors_list):
    knl = k_neighbors_list

    return max(knl, key=lambda x: x[-1]) #get the class that occurs most within the list of knl



def visualize_k_neighest_neighbors(k_value, neighbors_list, test_instance):
    
    return 



#Given KNN when K=1 and there are one of each class that need partitioned spaces 
def Voronoi_Partition():
    return 0






