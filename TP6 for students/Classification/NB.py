from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# data = load_breast_cancer()
# label_names = data['target_names']
# labels = data['target']
# feature_names = data['feature_names']
# features = data['data']
from sklearn.model_selection import cross_val_score
import pandas as pd
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

gnb = GaussianNB()

# Insert code
	# train_test_split
	# predict
	# accuracy
	# confusion

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)