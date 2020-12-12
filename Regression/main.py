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