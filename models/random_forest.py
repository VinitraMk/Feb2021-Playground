from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

class RandomForest:
    X = None
    y = None

    def __init__(self,X,y):
        self.X = X
        self.y = y

    def get_model(self):
        model = RandomForestRegressor(n_estimators=800, max_features='log2', max_depth=15, min_samples_split=2, random_state=42, n_jobs=-1)
        model.fit(self.X,self.y)
        return model
