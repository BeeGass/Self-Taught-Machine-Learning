import pandas as pd
import numpy as np
import torch
import sys
import os 
from urllib.request import urlretrieve
import argparse
from kNN import do_kNN

def main():
    # Create the parser
    my_parser = argparse.ArgumentParser(prog='lml',
                                        usage='%(prog)s [options] path',
                                        description='Here you will be able to run and visualize different statistical algorithms.',
                                        epilog='The hope behind this program was to make an incredibly explicit program for me to reference in the future with tons of inline comments to help me follow along as well as some other stuff to make this a potentially viable piece of software to use if I ever want to test some stuff out.',
                                        prefix_chars='-'
                                        )
    # Add the arguments
    my_parser.add_argument('--input', action='store', type=int, required=True)
    my_parser.add_argument('--id', action='store', type=int)
    my_parser.add_argument('Path', metavar='path', type=str, help='the path to list')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    input_path = args.Path

    if not os.path.isdir(input_path):
        print('The path specified does not exist')
        sys.exit()

    print('\n'.join(os.listdir(input_path)))

    dataset_dict = do_data_stuff()
    pick_ml_algo(dataset_dict)

def do_data_stuff(ttv_bool = True: bool, pca_bool = False: bool, train_perc_input = 0.8: float, n_components: int):
    the_dataset = choose_dataset()

    #ttv_binary_bool = int(input("0 for train-test split, 1 for train-test-validation split: "))

    if ttv_bool:
        #train_perc_input = float(input("Please input a value between 0 and 1 that signifies how large of a training set ratio you would like: "))
        tt_list = random_train_test_split(the_dataset, train_perc_input) #returns train/test list
        split_dict = feature_label_split(tt_list) #just so you keep things straight, this is a 2 dimensional list. Its a list containing the 2 datasets (train and test) where each "dataset" list holds the dataframes to both the feature vector and the label vector

    
    elif ttv_bool == False:
        #train_perc_input = float(input("Please input a value between 0 and 1 that signifies how large of a training set/validation set/testing set ratio you would like: "))
        ttv_list = random_train_test_validation_split(the_dataset, train_perc_input) #returns train/test/validation list
        split_dict = feature_label_split(ttv_list) #just so you keep things straight, this is a 2 dimensional list. Its a list containing the 3 datasets (train, test and validation) where each "dataset" list holds the dataframes to both the feature vector and the label vector
    

    #pca_bool = int(input("Would you like to perform PCA on this data?\n1 for Yes, 0 for No: "))
    if pca_bool:
        #n_components = int(input("What dimensionality would you like your data to be? Please give the number of dimensions: "))
        eig_vecs = get_eigen_vectors(split_dict["train"]["features"], n_components)

        for a_dataset in split_dict:
            split_dict[a_dataset]["features"] = apply_pca(split_dict[a_dataset]["features"], eig_vecs)
        
    return split_dict



#Function that allows you to pick which algorithm you wish to use. 
#------INPUTS-------
#dataset_list: the list that contains that train, test and potential validation set that will be used by these algorithms 
def pick_ml_algo(dataset_dict):
    print("The algorithms you can pick from:")
    print("- 0: KNN(available)")
    print("- 1: Neural Network(not available)")
    print("- 2: Regression(not available)")
    print("- 3: SVM(not available)")
    print("- 4: Bagging(not available)")
    print("- 5: Boosting(not available)")
    print("- 6: Bayes(not available)")
    print("- 7: QDA(not available)")
    print("- 8: LDA(not available)")
    print("- 9: Decision Tree(not available)")

    algo_number = int(input("pick the number associated with the algorithm you wish to use: "))

    switcher = {
        0: do_kNN(dataset_dict),
        1: do_NN(dataset_dict),
        2: do_regression(dataset_dict),
        3: do_SVM(dataset_dict),
        4: do_bagging(dataset_dict),
        5: do_boosting(dataset_dict),
        6: do_bayes(dataset_dict),
        7: do_qda(dataset_dict),
        8: do_lda(dataset_dict),
        9: do_decision_tree(dataset_dict)
    }

    return switcher.get(algo_number, "invalid algorithm number")



def choose_dataset():
    print("Please input the name of the default dataset which is \"iris\".")
    print("If you have an outside link to a dataset you wish to use please input \"Other\"")
    print("If you wish to upload your own .csv then you will be prompted to do if you type \"Upload\"")
    input_var = str(input("Please provide your preferred dataset now: "))
    dataset = None 

    delim = str(input("Please put in the character for the deliminator used to parse your dataset. Please make sure to not put a space afterward: "))
    if input_var == "iris": #iris datset
        iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        urlretrieve(iris)
        df = pd.read_csv(iris, sep=delim, header=None)
        dataset = df

    elif input_var == "Other": #any other dataset
        link = str(input("Please provide the link to the dataset you wish to use. Please make sure the link ends in \".data\""))
        urlretrieve(link)
        df = pd.read_csv(link, sep=delim, header=None)
        dataset = df

    elif input_var == "Upload":
        dirname = os.path.dirname(__file__)
        print("please make sure that your csv file has the labels at the tail end of columns listed. The program will not work if the labels are placed elsewhere")
        filename = os.path.join(dirname, input("Enter file path to .csv with formatting like this: ./KNN/enterYourCSVnameHERE: "))
        dataset = workable_csv(filename)
    
    else:
        print("you did not provide a valid answer you will be prompted to provide a correct answer now")
        choose_dataset()

    return dataset 



#given a csv file this function will convert the CSV into a workable pandas dataframe
def workable_csv(filename):
    #dataframe of inputted CSV file
    df = pd.read_csv(filename)

    return df



#format_dataframe is to put all the label columns toward the back columns of the dataframe 
#------INPUTS-------
#label_str is the string associated with the first column that contains the labels for the dataset
def format_dataframe(dataset_df, label_str):
    
    label_col = dataset_df.pop(label_str) #pop the column off

    dataset_df[label_str] = label_col # add it back into the dataframe at the last index 
    #dataset_df[label_col].insert(-1, label_str, label_col) # add it back into the dataframe at the last index 

    return dataset_df



#feature_label_split() is so we can isolate our labels from our features so we can perform operations on said features 
def feature_label_split(dataset_list):
    dataset_dict = {}

    the_label_position = int(input("please input the number associated with the column that starts the label columns: "))
    for i, a_dataset in enumerate(dataset_list):

        label_df = a_dataset.iloc[:, the_label_position:] 
        feature_df = a_dataset.iloc[:, :the_label_position] #drop the columns associated with the labels to leave only the feature columns

        if i == 0:
            dataset_dict["train"] = {"features": feature_df, "labels": label_df}
        elif i == 1:
            dataset_dict["test"] = {"features": feature_df, "labels": label_df}
        elif i == 2:
            dataset_dict["validation"] = {"features": feature_df, "labels": label_df}
        else:
            print("how")

    return dataset_dict


#Train/Test Split performed by randomly taking inputs and their associated labels and assigning them to either a training group or a test group
#---------------------------------------------------#----------------#
#                                                   |                |
#                  Training set                     |   Testing Set  |
#                                                   |                |
#---------------------------------------------------#----------------#
#train_per should be a value between 0 and 1, eg. train_perc 0.8
def random_train_test_split(dataset_df, train_perc):
    output_list = []
    dataset_df_copy = dataset_df.copy()

    train = dataset_df_copy.sample(frac = train_perc, random_state = 200) # randomly sample, given the seed of 200, whatever the percentage input of rows to produce training set
    test = dataset_df_copy.drop(train.index).sample(frac = 1.0) # remove the training set from the original dataframe to leave the testing dataset which also gets randomly shuffled based off sample
    
    output_list.append(train)
    output_list.append(test)

    return output_list
    


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
    output_list = []
    dataset_df_copy = dataset_df.copy()
 
    train = dataset_df_copy.sample(frac = train_perc, random_state = 200) #isolate training dataset by randomly picking rows 
    test_and_validation = dataset_df_copy.drop(train.index) #isolate the test and validation set by dropping the training dataset from the original
    validation = test_and_validation.sample(frac = 0.5) #isolate the test from the test and validation set by randomly sampling rows for the test dataset
    test = test_and_validation.drop(validation).sample(frac = 1.0) #isolate the test set by dropping the validation set from the test and validation settest. Then shuffle the rows

    output_list.append(train)
    output_list.append(test)
    output_list.append(validation)
    
    return output_list



#Train/Test Split performed by splitting the inputs and their associated labels based off the time sensitive data
def temporal_train_test_split():\
    #TODO
    return 0 



#Temporal Test/Validation split will take the test set gained from Temporal_Train_Test_Split() and split the data within it into test and 
#validation sets, based off the time component of the dataset
def temporal_test_validation_split():
    #TODO 
    return 0



#for getting caught up on the concepts of what Principal Component Analysis is: https://www.youtube.com/watch?v=FgakZw6K1QQ
#  
def get_eigen_vectors(dataset_df, n_components):
    dataset_np = dataset_df.to_numpy() #converts the dataframe into a numpy array

    mean_df = np.mean(dataset_np, axis=0) # here we find the mean of each row of the transformed matrix that was orginally dataset_np. The idea behind this is to eventually subtract our mean_df by dataset_df in order to center it to the origin. This can be best thought of as data that is on a number line and almost all of the data is near x = -53. In order to perform PCA we need to have the data centered at the origin. 
    
    centered_matrix = dataset_np - mean_df #centering of the data about the origin. 

    correlation_matrix = np.cov(centered_matrix, rowvar=False) #this determines how different the multidemensional variables are from one another  

    eigen_values, eigen_vectors = np.linalg.eig(correlation_matrix) #this decomposes the eigenvalues and eigenvectors. "The eigenvectors represent the directions or components for the reduced subspace of B, whereas the eigenvalues represent the magnitudes for the directions" https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/

    sorted_index = np.argsort(eigen_values)[::-1] # sorting in a descending order in order for the eigenvalues to correspond to respective eigenvector 
    sorted_eigen_value = eigen_values[sorted_index]

    sorted_eigen_vectors = eigen_vectors[:,sorted_index]

    eigenvector_subset = sorted_eigen_vectors[:,0:n_components] # returns a n_component dimensional matrix

    return eigenvector_subset



def apply_pca(dataset_df, eigen_vectors):
    dataset_np = dataset_df.to_numpy() #converts the dataframe into a numpy array

    mean_df = np.mean(dataset_np, axis=0) # here we find the mean of each row of the transformed matrix that was orginally dataset_np. The idea behind this is to eventually subtract our mean_df by dataset_df in order to center it to the origin. This can be best thought of as data that is on a number line and almost all of the data is near x = -53. In order to perform PCA we need to have the data centered at the origin. 
    
    centered_matrix = dataset_np - mean_df #centering of the data about the origin. 

    reduced_matrix = np.dot(eigen_vectors.transpose(),centered_matrix.transpose()).transpose()

    return pd.DataFrame(reduced_matrix)


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



if __name__ == '__main__':
    main()




