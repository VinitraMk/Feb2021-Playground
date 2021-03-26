import os
import warnings

#custom imports
from modules.preprocessing import Preprocessor
from modules.utils import Utils

from models.linear_regression import LinearRegressor
from models.random_forest import RandomForest

utils = Utils(os.getcwd())
preprocessor = Preprocessor(os.getcwd())
X_train, y_train, X_valid, y_valid, X_test = preprocessor.data_split("target")

X_train, X_valid, X_test = preprocessor.handle_categorical_variables()
X_train, X_valid, X_test = preprocessor.handle_missing_values()

#model validation
print('\n\nBuildling Linear Regressor model')
linear_regressor = LinearRegressor(X_train,y_train).get_model()
mae_linear_regressor = utils.get_mae(linear_regressor,X_valid,y_valid)
print("Mean squared error for a linear regressor: ",mae_linear_regressor)
print("Saving model preds to csv...")
test_preds = linear_regressor.predict(X_test)
utils.write_results_to_csv(test_preds,"target","linear_regressor_preds")

print('\n\nBuidling Random Forest Regressor')
random_forest = RandomForest(X_train,y_train).get_model()
mae_rf_regressor = utils.get_mae(random_forest,X_valid,y_valid)
print("Mean squared error for a random forest regressor: ",mae_rf_regressor)
print("Saving model preds to csv...")
test_preds = random_forest.predict(X_test)
utils.write_results_to_csv(test_preds,"target","random_forest_preds")
