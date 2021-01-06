import pandas as pd
import numpy as np
import torch
import sys
import os 
from urllib.request import urlretrieve

if __name__ == '__main__':
    main()

def main():
    dataset = do_data_stuff(sys.argv)

    return dataset

#------INPUTS-------
#argv[1]:
#argv[2]:
def do_data_stuff(argv):
    og_dataset = choose_dataset(argv[1]) 
    output_list = []

    label_str_input = input("Please input the string associated with the column that contains the classes: ")
    the_dataset = format_dataframe(og_dataset, label_str_input)

    if argv[2] == 0:
        train_perc_input = float(input("Please input a value between 0 and 1 that signifies how large of a training set ratio you would like: "))
        train_set, test_set = random_train_test_split(the_dataset, train_perc_input)
        output_list.extend(train_set, test_set)
    
    elif argv[2] == 1:
        train_perc_input = float(input("Please input a value between 0 and 1 that signifies how large of a training set/validation set/testing set ratio you would like: "))
        train_set, test_set, validation_set = random_train_test_validation_split(the_dataset, train_perc_input)
        output_list.extend(train_set, test_set, validation_set)

    return output_list

def choose_dataset(dataset_title_input):
    input = dataset_title_input
    dataset = None 

    if input == "iris": #iris datset
        iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        urlretrieve(iris)
        dataset = pd.read_csv(iris, sep=',')

    else:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, input("Enter file path to .csv with formatting like this: ./KNN/enterYourCSVnameHERE: "))
        dataset = workable_csv(filename)

    return dataset 

#given a csv file this function will convert the CSV into a workable pandas dataframe
def workable_csv(filename):
    #dataframe of inputted CSV file
    df = pd.read_csv(filename)

    return df

#label_str is the string associated with the column that contains the labels
def format_dataframe(dataset_df, label_str):
    
    label_col = dataset_df[label_str].pop() #pop the column off

    X = dataset_df.insert(-1, label_str, label_col) # add it back into the dataframe at the last index 

    return X


#Train/Test Split performed by randomly taking inputs and their associated labels and assigning them to either a training group or a test group
#---------------------------------------------------#----------------#
#                                                   |                |
#                  Training set                     |   Testing Set  |
#                                                   |                |
#---------------------------------------------------#----------------#
#train_per should be a value between 0 and 1, eg. train_perc 0.8
def random_train_test_split(dataset_df, train_perc):
    dataset_df_copy = dataset_df.copy()

    train = dataset_df_copy.sample(frac = train_perc, random_state = 200) # randomly sample, given the seed of 200, whatever the percentage input of rows to produce training set
    test = dataset_df_copy.drop(train.index).sample(frac = 1.0) # remove the training set from the original dataframe to leave the testing dataset which also gets randomly shuffled based off sample
    
    return train, test
    

#Random Test/Validation split will take the test set gained from Random_Train_Test_Split() and split the data within it into test and validation sets, randomly
#recommended ratios are:
# 70% train, 15% val, 15% test
# 80% train, 10% val, 10% test
# 60% train, 20% val, 20% test.
#                                                   <----------test_and_validation------------> 
#---------------------------------------------------#--------------------#--------------------#
#                                                   |                    |                    |
#                  Training set                     |   Validation Set   |    Testing Set     |
#                                                   |                    |                    | 
#---------------------------------------------------#--------------------#--------------------#
def random_train_test_validation_split(dataset_df, train_perc):
    dataset_df_copy = dataset_df.copy()
 
    train = dataset_df_copy.sample(frac = train_perc, random_state = 200) #isolate training dataset by randomly picking rows 
    test_and_validation = dataset_df_copy.drop(train.index) #isolate the test and validation set by dropping the training dataset from the original
    validation = test_and_validation.sample(frac = 0.5) #isolate the test from the test and validation set by randomly sampling rows for the test dataset
    test = test_and_validation.drop(validation).sample(frac = 1.0) #isolate the test set by dropping the validation set from the test and validation settest. Then shuffle the rows

    
    return train, test, validation

#Train/Test Split performed by splitting the inputs and their associated labels based off the time sensitive data
def temporal_train_test_split():\
    #TODO
    return 0 

#Temporal Test/Validation split will take the test set gained from Temporal_Train_Test_Split() and split the data within it into test and 
#validation sets, based off the time component of the dataset
def temporal_test_validation_split():
    #TODO 
    return 0

#for getting caught up on the concepts of what PCA is: https://www.youtube.com/watch?v=FgakZw6K1QQ
#  
def principal_component_analysis(dataset_df, n_components):
    dataset_np = dataset_df.to_numpy() #converts the dataframe into a numpy array

    mean_df = np.mean(dataset_np, axis=0) # here we find the mean of each row of the transformed matrix that was orginally dataset_np. The idea behind this is to eventually subtract our mean_df by dataset_df in order to center it to the origin. This can be best thought of as data that is on a number line and almost all of the data is near x = -53. In order to perform PCA we need to have the data centered at the origin. 
    
    centered_matrix = dataset_df - mean_df #centering of the data about the origin. 

    correlation_matrix = np.cov(centered_matrix, rowvar=False) #this determines how different the multidemensional variables are from one another  

    eigen_values, eigen_vectors = np.linalg.eig(correlation_matrix) #this decomposes the eigenvalues and eigenvectors. "The eigenvectors represent the directions or components for the reduced subspace of B, whereas the eigenvalues represent the magnitudes for the directions" https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/

    sorted_index = np.argsort(eigen_values)[::-1] # sorting in a descending order in order for the eigenvalues to correspond to respective eigenvector 
    sorted_eigen_value = eigen_values[sorted_index]

    sorted_eigen_vectors = eigen_vectors[:,sorted_index]

    eigenvector_subset = sorted_eigen_vectors[:,0:n_components] # returns a n_component dimensional matrix

    reduced_matrix = np.dot(eigenvector_subset.transpose(),centered_matrix.transpose()).transpose()

    return reduced_matrix

#adding noise to your validation, test or even training datasets maybe needed at times for testing models. This is what that is for 
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
def noise_addition(df, mu = 0):
    sigma = float(input("sigma: "))

    if input("would you like to change mu? y/n") == "y":
        mu = float(input("enter mu value: "))

    if sigma < 0:
        print("please put in a sigma greater than or equal to zero")
        noise_addition(df, mu)

    noise = np.random.normal(mu, sigma, df.shape)
    noisy_df = df + noise

    return noisy_df






