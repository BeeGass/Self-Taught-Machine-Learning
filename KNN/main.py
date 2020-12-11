import pandas as pd
import numpy as np
import torch
import sys
import os 

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, input("Enter file path to .csv with formatting like this: ./KNN/enterYourCSVnameHERE: "))

if __name__ == '__main__':
    main()

def main():
    i = 0 
    return i

#function that defines the label associated with the inputted input value. x_i -> y_i
def hypothesis(x_i):
    return 0

#Loss function that calculates the effectiveness of the dataset by returning an error percentage
def zero_one_loss():
    return 0

#Loss function that calculates the error in indivual inputs that when added up shows the error rate given the entire dataset
def squared_loss():
    return 0

#Loss function to be used when the dataset has strong outliers that might scew the error if one were to use the Squared_Loss function 
def absolute_loss():
    return 0

#Distance from unknown datapoint to arbitrary datapoint in terms of coordinate blocks
#p = 1 
def manhattan_distance(p_1, p_2):

    return 0

#Distance from unknown datapoint to arbitrary datapoint
#p = 2
def euclidean_distance():
    return 0

#p = 3
def max_distance():
    return 0

#Minkowski distance
def minkowski_distance():
    for index, row in df.iterrows():
        for label, content in df.iteritems():
            
    return 0

#Given KNN when K=1 and there are one of each class that need partitioned spaces 
def Voronoi_Partition():
    return 0






