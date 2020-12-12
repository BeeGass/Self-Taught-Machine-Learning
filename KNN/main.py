import pandas as pd
import numpy as np
import torch
import sys
import os 
import math

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, input("Enter file path to .csv with formatting like this: ./KNN/enterYourCSVnameHERE: "))

if __name__ == '__main__':
    main()

def main():
    i = 0 
    return i

#Distance from unknown datapoint to arbitrary datapoint in terms of coordinate blocks
#p = 1 
def manhattan_distance(dataset_df, the_row):
    for row in dataset_df:
        distance = minkowski_distance(the_row, row, 1)

    return distance

#Distance from unknown datapoint to arbitrary datapoint
#p = 2
def euclidean_distance(dataset_df, the_row):
    for row in dataset_df:
        distance = minkowski_distance(the_row, row, 2)

    return distance

#p = infinity
def max_distance(dataset_df, the_row):
    for row in dataset_df:
        distance = minkowski_distance(the_row, row, math.inf)

    return distance

#Minkowski distance
#row1 behaves as the arbitrary vector we wish to find distances from
#row2 are all the vectors we will be calculating the distance to
#p_val is the p_val the determines if you are doing manhattan, euclidean or max distance
def minkowski_distance(row1, row2, p_val):

    for i in range(len(row1) - 1): #we want everything but the label associated with vector
         distance += (math.abs(row1[i] - row2[i]) ** p_val #go through each dimension of the vector and subtract it from the other

    vec_dist = distance**(1/p_val)

    return vec_dist

#Given KNN when K=1 and there are one of each class that need partitioned spaces 
def Voronoi_Partition():
    return 0






