# test_main.py
from sklearn.linear_model import LinearRegression

def test_model():
    X = [[1], [2], [3]]
    y = [1, 2, 3]
    model = LinearRegression()
    model.fit(X, y)
    assert model.score(X, y) == 1.0
