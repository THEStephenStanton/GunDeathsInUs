import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

guns = pd.read_csv("GunDeaths.csv", index_col=0)

# Prep the data (See Analysis.ipynb)
del guns["year"]
del guns["month"]
del guns["hispanic"]
guns = guns.dropna(axis=0, how="any")
guns = guns[guns.intent != "Undetermined"]
indexOfOthers = guns[(guns.place != "Home") & (guns.place != "Street")].index
guns.loc[indexOfOthers, "place"] = "Other"

guns = guns.apply(LabelEncoder().fit_transform)
        
X = guns.iloc[:, 1:]
y = guns.iloc[:, 0]
        
from sklearn import datasets
iris = datasets.load_iris()

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = .20)

# Just gonna try a simple knn classifier, will be good base to compare to
# After testing, n = 10 seems to be about the best we will get
knn = neighbors.KNeighborsClassifier(n_neighbors=10)
knn.fit(XTrain, yTrain)
accuracy = knn.score(XTest, yTest)
print("KNN: {0} %".format(accuracy * 100))

# So since this is a classification problem, we can't use linear regression,
# it will give too much weight to data far from the decision frontier 
# I still want to use a linear approach, so I will choose logistic regression 
logReg = LogisticRegression()
logReg.fit(XTrain, yTrain)
accuracy = logReg.score(XTest, yTest)
print("Logistic Regression: {0} %".format(accuracy * 100))

# Might as well try a decision tree
decisionTree = DecisionTreeClassifier()
decisionTree.fit(XTrain, yTrain)
accuracy = decisionTree.score(XTest, yTest)
print("Decision Tree: {0} %".format(accuracy * 100))

print("Limiting Linear SVC to 5000 points or it will take ages")
time.sleep(1) # just so the message above will show

# Using the ultimate machine learning cheat sheet, it says given
# the parameters, I should choose linear svc
svc = LinearSVC()
svc.fit(XTrain[:5000], yTrain[:5000])
svc.score(XTest, yTest)
accuracy = svc.score(XTest, yTest)
print("SVC: {0} %".format(accuracy * 100))