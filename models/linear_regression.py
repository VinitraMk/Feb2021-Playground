from sklearn.linear_model import LinearRegression

class LinearRegressor:
    X=None
    y=None

    def __init__(self,X,y):
        self.X = X
        self.y = y

    def get_model(self):
        model = LinearRegression(normalize=True, n_jobs=-1, positive=True)
        model.fit(self.X, self.y)
        return model
