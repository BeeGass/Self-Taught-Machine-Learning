import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import torch
import sys
import os 
import math


def do_kNN(datasets):
    train_set = datasets["train"]
    test_set = datasets["test"]
    validation_set = None
    
    if len(datasets) == 3:
        validation_set = datasets["validation"]

    test_instance = get_testing_instance(test_set)
    distance_list = distance_prediction(train_set, test_instance)
    k_value = int(input("What would you like your k value to be: "))
    neighbors = get_k_nearest_neighbors(distance_list, k_value)
    get_most_freq_class(neighbors)


#goes through test_set dataframe and picks row at random to behave as the testing row
#------INPUTS-------
#input_test_set: the testing set data frame
def get_testing_instance(input_test_set):
    np_arr = input_test_set["features"]

    #the_row = df.sample(n=1) #take only one row to test
    the_row = np.random.choice(np_arr)
    #the_row = np_arr.iloc[]
    print("the testing row", the_row) 
    #TODO GET THE RANDOM SAMPLING DONE!

    return the_row



#Distance from unknown datapoint to arbitrary datapoint in terms of coordinate blocks
#p = 1 
#------INPUTS-------
#dataset_df: a dataframe that should be a training set
#the_testing_row: the vector that represent the point that is going to be tested to see what class it falls under
def manhattan_distance(dataset_df, the_testing_row):
    distances = []

    for a_row in dataset_df:
        dist = minkowski_distance(the_testing_row, a_row, 1)
        distances.append(a_row, dist)

    return distances



#Distance from unknown datapoint to arbitrary datapoint
#p = 2
#------INPUTS-------
#dataset_df: a dataframe that should be a training set
#the_testing_row: the vector that represent the point that is going to be tested to see what class it falls under
def euclidean_distance(dataset_df, the_testing_row):
    distances = []
    feature_set = dataset_df["features"]
    label_set = dataset_df["labels"]

    for i, a_row in enumerate(feature_set):
        dist = minkowski_distance(the_testing_row, a_row, 2)
        distances.append((a_row, dist, label_set.iloc[i]))

    return distances



#p = infinity
#A little description of what this is doing to future Bryan: Basically we know that in a "p" dimensional world that things become to get impossibly large. In order to correctly get the distance between 2 different things we need to find the p-distance in the p-dimensional space
#------INPUTS-------
#dataset_df: a dataframe that should be a training set
#the_testing_row: the vector that represent the point that is going to be tested to see what class it falls under
def max_distance(dataset_df, the_testing_row):
    distances = []

    for a_row in dataset_df:
        dist = minkowski_distance(the_testing_row, a_row, math.inf)
        distances.append((a_row, dist))

    return distances



#Minkowski distance
#------INPUTS-------
#row1 behaves as the arbitrary vector we wish to find distances from
#row2 are all the vectors we will be calculating the distance to
#p_val: the p_val determines if you are doing manhattan, euclidean or max distance
def minkowski_distance(row1, row2, p_val):
    distance = 0.0

    for i in range(len(row1)): #we want everything but the label associated with vector
        print("row1: ", row1[i])
        print("row2: ", row2[i])

        distance += ((row1[i] - row2[i]).abs()) ** p_val #go through each dimension of the vector and subtract it from the other

    the_dist = (distance ** (1.0 / p_val)) # (distance)^(1/p)

    return the_dist


#------INPUTS-------
#train_set: a dataframe that is the training set
#test_instance: the vector that represent the point that is going to be tested 
def distance_prediction(train_set, test_instance):
    
    vec_dist_class_list = euclidean_distance(train_set, test_instance) #returns a list of tuples that has the arbitrary vector being analyzed, the distance to that aribtrary vector, and the class associated with that vector
    print(vec_dist_class_list)
    sorted_vdc_list = vec_dist_class_list.sort(key = (lambda x: x[1])) #we sort the list based off the value of the distance. this will give us smallest values closer to 0th index
    print(sorted_vdc_list)
    
    return sorted_vdc_list


#------INPUTS-------
#sorted_vdc_list: a list of distances from smalles to greatest as well as the vector assocated with distance
#k_value: k is the value determined by the user and reflects the (k) number of shortest distances from the test instances we will accept 
def get_k_nearest_neighbors(sorted_vdc_list, k_value):
    neighbors = []
    print(sorted_vdc_list)

    for i in range(k_value):
        neighbors.append(sorted_vdc_list[i]) # find the k number of values closest to our test_instance in terms of the row associated with the distance

    return neighbors


#------INPUTS-------
#k_neighbors_list: k number of values closest to our test_instance in terms of the row associated with the distance
def get_most_freq_class(k_neighbors_list):
    knl = k_neighbors_list

    return max(knl, key=lambda x: x["Label"]) #get the class that occurs most within the list of knl



def visualize_k_neighest_neighbors(k_value, neighbors_list, test_instance):
    
    return 


#Given KNN when K=1 and there are one of each class that need partitioned spaces 
def Voronoi_Partition():
    return 0





