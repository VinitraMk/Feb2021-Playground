from sklearn.ensemble import RandomForestRegressor

class RandomForest:
    X = None
    y = None

    def __init__(self,X,y):
        self.X = X
        self.y = y

    def get_model(self):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(self.X,self.y)
        return model
