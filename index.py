import os
import warnings

#custom imports
from modules.preprocessing import Preprocessor
from modules.utils import Utils

from models.linear_regression import LinearRegressor
from models.random_forest import RandomForest

from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import cross_val_score 

k_split=False
utils = Utils(os.getcwd())
preprocessor = Preprocessor(os.getcwd(),k_split)

X_train = None
y_train = None
X_valid = None
y_valid = None
X_test = None
id_cols = None

if not(k_split):
    X_train, y_train, X_valid, y_valid, X_test, id_cols = preprocessor.data_split("target")
    X_train, X_valid, X_test = preprocessor.handle_categorical_variables()
    X_train, X_valid, X_test = preprocessor.handle_missing_values()
else:
    X_train, y_train, X_test, id_cols = preprocessor.data_split("target")
    X_train, X_test = preprocessor.handle_categorical_variables()
    X_train, X_test = preprocessor.handle_missing_values()

#model validation
print('Building Linear Regressor model')
linear_regressor = LinearRegressor(X_train,y_train).get_model()
mae_linear_regressor = utils.get_mae(linear_regressor,X_valid,y_valid)
print("Mean squared error for a linear regressor: ",mae_linear_regressor)
print("Saving model preds to csv...")
test_preds = linear_regressor.predict(X_test)
utils.write_results_to_csv(test_preds,id_cols,"target","linear_regressor_preds")

'''
print('\n\nBuidling Random Forest Regressor')
random_forest = RandomForest(X_train,y_train).get_model()
mae_rf_regressor = utils.get_mae(random_forest,X_valid,y_valid)
print("Mean squared error for a random forest regressor: ",mae_rf_regressor)
print("Saving model preds to csv...")
test_preds = random_forest.predict(X_test)
utils.write_results_to_csv(test_preds,id_cols,"target","random_forest_preds")
'''
