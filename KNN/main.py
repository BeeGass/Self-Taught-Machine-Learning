if __name__ == '__main__':
    main()

def main():
    i = 0 
    return i

#Train/Test Split performed by randomly taking inputs and their associated labels and assigning them to either a training group or a test group
def Random_Train_Test_Split():
    return 0

#Random Test/Validation split will take the test set gained from Random_Train_Test_Split() and split the data within it into test and validation sets, randomly
def Random_Test_Validation_Split():
    return 0

#Train/Test Split performed by splitting the inputs and their associated labels based off the time sensitive data
def Temporal_Train_Test_Split():
    return 0

#Temporal Test/Validation split will take the test set gained from Temporal_Train_Test_Split() and split the data within it into test and 
#validation sets, based off the time component of the dataset
def Temporal_Test_Validation_Split():
    return 0

#adding noise to your validation, test or even training datasets maybe needed at times for testing models. This is what that is for 
def Noise_Addition():
    return 0

#function that defines the label associated with the inputted input value. x_i -> y_i
def Hypothesis(x_i):
    return 0

#Loss function that calculates the effectiveness of the dataset by returning an error percentage
def Zero_One_Loss():
    return 0

#Loss function that calculates the error in indivual inputs that when added up shows the error rate given the entire dataset
def Squared_Loss():
    return 0

#Loss function to be used when the dataset has strong outliers that might scew the error if one were to use the Squared_Loss function 
def Absolute_Loss():
    return 0

#Distance from unknown datapoint to arbitrary datapoint in terms of coordinate blocks
#p = 1 
def Manhattan_Distance():
    return 0

#Distance from unknown datapoint to arbitrary datapoint
#p = 2
def Euclidean_Distance():
    return 0

#p = 3
def Max_Distance():
    return 0

#Minkowski distance
def Minkowski_Distance():
    return 0

#Given KNN when K=1 and there are one of each class that need partitioned spaces 
def Voronoi_Partition():
    return 0






