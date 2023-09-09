import pandas as pd # CSV file I/O
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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

# remove unneccessary data
x_train = x_train.drop(["Id", "City"], axis=1)

# separate numeric and non-numeric data
x_train_num = x_train.select_dtypes(include=("int64", "float64"))
x_train_cat = x_train.select_dtypes(include=("object"))

# impute and scale the numeric data
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])


# combine numeric data and categorical data using encoder
num_features_name = list(x_train_num.columns)
full_pipeline = ColumnTransformer([("num features", num_pipeline, num_features_name),
                                   ("ordinal encoding", OrdinalEncoder(), ["City Group"]),
                                   ("one hot encoding", OneHotEncoder(), ["Type"])])


# fit data using the pipeline
x_train_preprocessed = full_pipeline.fit_transform(x_train)