from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import os

class Utils:
    project_dir=""
    output_dir=""

    def __init__(self,project_dir):
        self.project_dir = project_dir
        self.output_dir = os.path.join(self.project_dir,"output")

    def get_mae(self,model,X_val,y_val):
        val_preds = model.predict(X_val)
        mae = mean_absolute_error(y_val,val_preds)
        return mae

    def write_results_to_csv(self,preds,id_cols,target_col,file_name):
        prediction = pd.DataFrame(preds,index=id_cols,columns=[target_col]).to_csv(os.path.join(self.output_dir,f"{file_name}.csv"))

