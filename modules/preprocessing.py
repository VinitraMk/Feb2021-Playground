from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

class Preprocessor:
    train_file_path=""
    test_file_path=""
    project_dir=""
    input_dir=""
    X_train=y_train=X_test=y_test=X_valid=y_valid = None

    def __init__(self,project_dir):
        self.project_dir = project_dir
        self.input_dir = os.path.join(self.project_dir,"input")
        self.train_file_path = os.path.join(self.input_dir,"train.csv")
        self.test_file_path = os.path.join(self.input_dir,"test.csv")

    def data_split(self,output_feature):
        app_data = pd.read_csv(self.train_file_path)
        y = app_data[output_feature]
        X = app_data[app_data.columns.difference([output_feature])]
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, random_state=0)
        self.X_test = pd.read_csv(self.test_file_path)
        print("Split data into test, train and valid\n")
        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test

    def handle_categorical_variables(self):
        label_encoder = LabelEncoder()
        oh_encoder = OneHotEncoder(handle_unknown="ignore",sparse=False)
        train = self.X_train.copy()
        valid = self.X_valid.copy()
        test = self.X_test.copy()
        object_cols = train.select_dtypes(include=['object']).columns
        oh_label_cols = [] 
        le_label_cols = []

        for col in object_cols:
            if len(np.unique(train[col])) < 10:
                oh_label_cols.append(col)
            else:
                le_label_cols.append(col)

        print("Labels with OneHot Encoding: ",oh_label_cols)
        print("Labels with LabelEncoder Encoding: ",le_label_cols,"\n")

        self.X_train = pd.DataFrame(oh_encoder.fit_transform(train[oh_label_cols]))
        self.X_valid = pd.DataFrame(oh_encoder.transform(valid[oh_label_cols]))
        self.X_test = pd.DataFrame(oh_encoder.transform(test[oh_label_cols]))
        self.X_train.index = train.index
        self.X_valid.index = valid.index
        self.X_test.index = test.index

        num_train = train.drop(columns=oh_label_cols,axis=1)
        num_valid = valid.drop(columns=oh_label_cols,axis=1)
        num_test = test.drop(columns=oh_label_cols,axis=1)

        self.X_train = pd.concat([num_train,self.X_train],axis=1)
        self.X_valid = pd.concat([num_valid,self.X_valid],axis=1)
        self.X_test = pd.concat([num_test,self.X_test],axis=1)

        for col in le_label_cols:
            self.X_train[col] = label_encoder.fit_transform(self.X_train[col])
            self.X_valid[col] = label_encoder.transform(self.X_valid[col])
            self.X_test[col] = label_encoder.transform(self.X_test[col])

        return self.X_train, self.X_valid, self.X_test

    def handle_missing_values(self):
        imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
        train_copy = self.X_train.copy()
        valid_copy = self.X_valid.copy()
        test_copy = self.X_test.copy()
        self.X_train = pd.DataFrame(imputer.fit_transform(train_copy))
        self.X_valid = pd.DataFrame(imputer.transform(valid_copy))
        self.X_test = pd.DataFrame(imputer.transform(test_copy))
        self.X_train.columns = train_copy.columns
        self.X_valid.columns = valid_copy.columns
        self.X_test.columns = test_copy.columns
        print("Imputed missing values\n")
        return self.X_train, self.X_valid, self.X_test

