import sys
import os
import warnings

#custom imports
from modules.preprocessing import Preprocessor

preprocessor = Preprocessor(os.getcwd())
X_train, y_train, X_valid, y_valid, X_test = preprocessor.data_split("target")

X_train, X_valid, X_test = preprocessor.handle_categorical_variables()
X_train, X_valid, X_test = preprocessor.handle_missing_values()



