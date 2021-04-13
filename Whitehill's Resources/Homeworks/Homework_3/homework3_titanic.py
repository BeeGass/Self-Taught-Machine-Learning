import pandas
import numpy as np

if __name__ == "__main__":
    # Load training data
    d = pandas.read_csv("train.csv")
    y = d.Survived.to_numpy()
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()

    # Train model using part of homework 3.
    # ...

    # Load testing data
    # ...

    # Compute predictions on test set
    # ...

    # Write CSV file of the format:
    # PassengerId, Survived
    # ..., ...
