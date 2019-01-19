'''
Created on Oct 31, 2018

@author: sibalnirbhay
'''

# This example has been taken from SciKit documentation and has been
# modified to suit this assignment. You are free to make changes, but you
# need to perform the task asked in the lab assignment


from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

print(__doc__)

# Loading the Wine data set
wine = datasets.load_wine();

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(wine.data)
X = wine.data.reshape((n_samples, -1))
y = wine.target

  
# Split the data set in two equal parts into 80:20 ratio for train:test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
  
# This is a key step where you define the parameters and their possible values
# that you would like to check.

print("A.\t Decision Tree")
print("B.\t Neural Net")
print("C.\t Support Vector Machine")
print("D.\t Gaussian Naive Bayes")
print("E.\t Logistic Regression")
print("F.\t k -Nearest Neighbors")
print("G.\t Bagging")
print("H.\t Random Forest")
print("I.\t AdaBoost Classifier")
print("J.\t Gradient Boosting Classifier")
print("K.\t XG Boost")
choice = input("  \t Choose an algorithm from the above list: ")

if choice=='a' or choice=='A':
    print("\n**********************************\n")
    print("  \t Decision Tree")
    tuned_parameters = [{'max_depth': [20],
                     'max_features': ['sqrt','log2'],
                     'min_impurity_decrease': [0.0001],
                     'max_leaf_nodes': [100]
                     }]
    algo = DecisionTreeClassifier()
    
elif choice=='b' or choice=='B':
    print("\n**********************************\n")
    print("  \t Neural Net")
    tuned_parameters = [{'activation': ['logistic','relu','tanh'],
                     'learning_rate': ['invscaling','adaptive'],
                     'hidden_layer_sizes': [100,150,200],
                     'max_iter': [1000]
                     }]
    algo = MLPClassifier()

elif choice=='c' or choice=='C':
    print("\n**********************************\n")
    print("  \t Support Vector Machine")
    tuned_parameters = [{'kernel': ['rbf', 'poly', 'sigmoid'],
                     'degree': [2, 3, 4, 5],
                     'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]
                     }]
    algo = SVC()
    
elif choice=='d' or choice=='D':
    print("\n**********************************\n")
    print("  \t Gaussian Naive Bayes")
    tuned_parameters = [{'priors':[None]
                     }]
    algo = GaussianNB()
    
elif choice=='e' or choice=='E':
    print("\n**********************************\n")
    print("  \t Logistic Regression")
    tuned_parameters = [{'penalty':['l1','l2'],
                         'tol':[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                         'C':[0.01, 0.1, 1, 10, 100],
                         'class_weight':['balanced',None]
                     }]
    algo = LogisticRegression()
    
elif choice=='f' or choice=='F':
    print("\n**********************************\n")
    print("  \t k -Nearest Neighbors")
    tuned_parameters = [{'n_neighbors':[3, 5, 7],
                         'weights':['uniform', 'distance'],
                         'algorithm':['ball_tree', 'kd_tree', 'brute'],
                         'p':[1, 2, 3]
                     }]
    algo = KNeighborsClassifier()
    
elif choice=='g' or choice=='G':
    print("\n**********************************\n")
    print("  \t Bagging")
    tuned_parameters = [{'n_estimators':[5, 10, 100, 200],
                         'max_features':[1, 3, 9],
                         'max_samples':[1, 5, 9, 21],
                         'random_state':[1, 2, 3, 5]
                     }]
    algo = BaggingClassifier()
    
elif choice=='h' or choice=='H':
    print("\n**********************************\n")
    print("  \t Random Forest")
    tuned_parameters = [{'n_estimators':[5, 10, 100, 200],
                         'criterion':['gini', 'entropy'],
                         'max_features':['log2', 'sqrt'],
                         'max_depth':[10, 100]
                     }]
    algo = RandomForestClassifier()

elif choice=='i' or choice=='I':
    print("\n**********************************\n")
    print("  \t AdaBoost Classifier")
    tuned_parameters = [{'n_estimators':[5, 10, 50, 100, 200],
                         'learning_rate':[0.1, 0.2, 0.5, 1],
                         'algorithm':['SAMME', 'SAMME.R'],
                         'random_state':[1, 2, 3, 5]
                     }]
    algo = AdaBoostClassifier()
    
elif choice=='j' or choice=='J':
    print("\n**********************************\n")
    print("  \t Gradient Boosting Classifier")
    tuned_parameters = [{'n_estimators':[5, 10, 50, 100, 200],
                         'learning_rate':[0.1, 0.2, 0.5, 1],
                         'min_impurity_decrease': [0.0001],
                         'max_depth':[10, 100]
                     }]
    algo = GradientBoostingClassifier()
    
elif choice=='k' or choice=='K':
    print("\n**********************************\n")
    print("  \t XG Boost")
    tuned_parameters = [{'n_estimators':[5, 10, 50, 100, 200],
                         'learning_rate':[0.1, 0.2, 0.5, 1],
                         'min_child_weight': [5, 10, 20],
                         'max_delta_step':[10, 100]
                     }]
    algo = XGBClassifier()
    
else:
    print("\nINVALID INPUT")
    exit()
  
# We are going to limit ourselves to accuracy score, other options can be
# seen here:
# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# Some other values used are the predcision_macro, recall_macro

scores = ['accuracy']

print()  
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
  
    clf = GridSearchCV(algo, tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)
  
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
  
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy Score: \n")
    print(accuracy_score(y_true, y_pred))
  
    print()
  
# Note the problem is too easy: the hyper-parameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
