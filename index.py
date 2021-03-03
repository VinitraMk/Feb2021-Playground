import sys
import os
import warnings

#custom imports
from modules.preprocessing import Preprocessor
from modules.utils import Utils

from models.linear_regression import LinearRegressor

utils = Utils(os.getcwd())
preprocessor = Preprocessor(os.getcwd())
X_train, y_train, X_valid, y_valid, X_test = preprocessor.data_split("target")

X_train, X_valid, X_test = preprocessor.handle_categorical_variables()
X_train, X_valid, X_test = preprocessor.handle_missing_values()


#model validation
linear_regressor = LinearRegressor(X_train,y_train).get_model()
mae_linear_regressor = utils.get_mae(linear_regressor,X_valid,y_valid)
print("Mean squared error for a linear regressor: ",mae_linear_regressor)
print("Saving model preds to csv...")
test_preds = linear_regressor.predict(X_test)
utils.write_results_to_csv(test_preds,"target","linear_regressor_preds")
