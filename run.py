import pandas as pd # CSV file I/O
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder 
from datetime import date
from datetime import datetime

# read train and test data
train = pd.read_csv("train.csv.zip")
x_test = pd.read_csv("test.csv.zip")

# split X and Y
y_train = train["revenue"] # keep column revenue as Y
x_train = train.drop("revenue", axis = 1) # remove column revenue from train and the remaining is X

# date to days
def date_to_days(date_list):
    today = date.today()
    days = []
    for val in date_list["Open Date"].values:
        casting_date = datetime.strptime(val, "%m/%d/%Y").date()
        delta = today - casting_date
        day = delta.days
        days.append(day)

    date_list["Open Date"] = pd.Series(days, name="Open Days")
    return date_list

# convert date to days
x_train = date_to_days(x_train)

