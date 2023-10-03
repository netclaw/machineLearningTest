from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
import pandas as pd

# col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# diabete = read_csv(r"/Users/youssefberkia/Documents/folder/machine learning with python/MLsklit/TP6 for students/Classification/diabetes.csv", header=None, names=col_names)
# diabete.head()
# feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
# X = diabete[feature_cols] # Features
# y = diabete.label # Target variable
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
# features = []
# features.append(('pca', PCA(n_components=3)))
# features.append(('select_best', SelectKBest(k=6)))
# feature_union = FeatureUnion(features)

# estimators = []
# estimators.append(('feature_union', feature_union))
# estimators.append(('logistic', LogisticRegression()))
model = LogisticRegression()
#Diviser le dataset en 21 parties: 1 pour apprentissage et 20 pour test
kfold = KFold(n_splits=20)
#résultats de tests par cross-validarion
results = cross_val_score(model, X, y, cv=kfold)
print(results)