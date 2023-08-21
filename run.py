import pandas as pd # CSV file I/O

# read train and test data
train = pd.read_csv("train.csv.zip")
x_test = pd.read_csv("test.csv.zip")

# split X and Y
y_train = train["revenue"] # keep column revenue as Y
x_train = train.drop("revenue", axis = 1) # remove column revenue from train and the remaining is X