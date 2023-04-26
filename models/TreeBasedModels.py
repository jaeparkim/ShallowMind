# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Define a class for Random Forest Classifier
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=3):
        # Initialize the parameters for Random Forest Classifier
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        # Create an instance of Random Forest Classifier
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)
    
    # Define a method to fit the model on input training data
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    # Define a method to predict the class labels for input test data
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    # Define a method to compute the accuracy of the model on input test data
    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

# Define a class for XGBoost Classifier
class XGBoost:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        # Initialize the parameters for XGBoost Classifier
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        # Create an instance of XGBoost Classifier
        self.model = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate)
    
    # Define a method to fit the model on input training data
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    # Define a method to predict the class labels for input test data
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    # Define a method to compute the accuracy of the model on input test data
    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)
