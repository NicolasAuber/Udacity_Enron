#!/usr/bin/python

########################
### Import Libraries ###
########################

# Standard libraries
import sys
import time
import pickle
import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import seaborn as sns

# Specific functions 
sys.path.append("./tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

# Machine learning libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree, ensemble, neighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import decomposition
from sklearn.pipeline import Pipeline

# Displays the version of some libraries
print("python: ", sys.version)
print("numpy: ", np.__version__)
print("pandas: ", pd.__version__)
print("seaborn: ", sns.__version__)

t0 = time.time()
########################
### Data Exploration ###
########################

# Loads the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    
# As already seen during Udacity Intro To Machine Learning course, there are 2 entries that are not employees.
# They are removed from the dataset.
no_employees_list = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for item in no_employees_list:
    data_dict.pop(item)
    
# Enron employees
employees_list = list(data_dict.keys())
print("Number of Enron employees: ", len(employees_list))
pprint.pprint(employees_list)

# Features for each Enron employee
features_list = list(data_dict['LAY KENNETH L'].keys())
print("Number of features: ", len(features_list))
pprint.pprint(features_list)

# Let's remove 'email_address' feature and re-order the features manually
features_list = ['poi',
                 'from_messages', 'to_messages', 'from_this_person_to_poi', 'from_poi_to_this_person', 'shared_receipt_with_poi',
                 'salary','bonus','long_term_incentive','deferred_income','deferral_payments','loan_advances','other','expenses','director_fees','total_payments',
                 'exercised_stock_options','restricted_stock','restricted_stock_deferred','total_stock_value']
print("Number of features: ", len(features_list))

data = featureFormat(data_dict, features_list, remove_NaN=False, remove_all_zeroes=False, remove_any_zeroes=False, sort_keys = False)
df = pd.DataFrame(data, index=employees_list, columns=features_list)

print(df.info())
print(df.head())

print("Total: ", df['poi'].count())
print("Number of POI: ", df[df['poi']==True]['poi'].count())
print("Number of non POI: ", df[df['poi']==False]['poi'].count())
print("###")
#df = df[df['salary'].notnull()]
#df = df[df['bonus'].notnull()]
df1 = df[df['from_messages'].notnull()]
print(df1.count())
print("###")
print("Total: ", df1['poi'].count())
print("Number of POI: ", df1[df1['poi']==True]['poi'].count())
print("Number of non POI: ", df1[df1['poi']==False]['poi'].count())

# Checks if some employees have only NaN values
df[df.isnull().sum(axis=1)==len(features_list)-1].index[:]

df['salary'].plot.hist(bins=20)
plt.show()

df['to_messages'].plot.hist(bins=20)
plt.show()

# Displays a scatter plot for 2 features distinguishing the POI and non POI 
def plotGraph(feature1, feature2):
    x_poi = df[df['poi']==True][feature1].tolist()
    x_npoi = df[df['poi']==False][feature1].tolist()
    y_poi = df[df['poi']==True][feature2].tolist()
    y_npoi = df[df['poi']==False][feature2].tolist()
    plt.scatter(x_poi, y_poi, color='r', label = 'poi') 
    plt.scatter(x_npoi, y_npoi, color='b', label = 'non poi') 
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.show()

# Scatter plot 'salary' vs 'bonus'. Prints remarkable values 
plotGraph('salary', 'bonus')
print(df.loc[(df['salary'] > 500000) | (df['bonus'] > 4000000),['poi','salary','bonus']])

# Scatter plot 'total_payments' vs 'total_stock_value'. Prints remarkable values 
plotGraph('total_payments', 'total_stock_value')
print(df.loc[df['total_stock_value'] > 10000000,['poi','total_payments','total_stock_value']])

# Scatter plot 'from_messages' vs 'to_messages'. Prints remarkable values 
plotGraph('from_messages', 'to_messages')
print(df.loc[(df['to_messages'] > 5000) | (df['from_messages'] > 5000),['poi','from_messages','to_messages']])

# Creates new email features focusing on ratios.
df['ratio_from_poi']= df['from_this_person_to_poi'] / df['from_messages']
df['ratio_to_poi']= df['from_poi_to_this_person'] / df['to_messages']
df['ratio_shared_poi']= df['shared_receipt_with_poi'] / df['to_messages']

# Scatter plot 'ratio_from_poi' vs 'ratio_to_poi'. Prints remarkable values 
plotGraph('ratio_from_poi', 'ratio_to_poi')
print(df.loc[(df['ratio_from_poi'] > 0.8) | (df['ratio_to_poi'] > 0.15),['poi','ratio_from_poi','ratio_to_poi']])

#########################
### Feature Selection ###
#########################

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Defines the features with too many NaN values and removes these features
for col in df.columns:
    if df[col].count() < 81:
        features_list.remove(col)
print("Number of features: ", len(features_list))
pprint.pprint(features_list)

################
### Outliers ###
################

### Task 2: Remove outliers

# Removes the outliers from the Enron dataset
outliers_list = ['LAY KENNETH L', 'KAMINSKI WINCENTY J','LOCKHART EUGENE E']
for item in outliers_list:
    data_dict.pop(item)

'''
# Removes the employees with NaN email features. THIS IS NOT USED.
employees_list = list(data_dict.keys())
for employee in employees_list:
    if data_dict[employee]['from_messages'] == "NaN":
        data_dict.pop(employee)
'''

# Enron employees
employees_list = list(data_dict.keys())
print("Number of employees: ", len(employees_list))

# Creates the dataset
my_dataset = data_dict

####################
### New Features ###
####################

### Task 3: Create new feature(s)

# Calculates the ratio between 2 values. NaN for any of the 2 values will result in null values.
def calcRatio(poi_messages, all_messages):
    ratio = 0.
    if poi_messages == "NaN" or all_messages == "NaN":
        ratio = 0.
    else:
        ratio = float(poi_messages) / float(all_messages)
    return ratio

# Generates new features in the dataset
for name in data_dict:
    data_dict[name]['ratio_to_poi'] = calcRatio(data_dict[name]['from_this_person_to_poi'],data_dict[name]['from_messages'])
    data_dict[name]['ratio_from_poi'] = calcRatio(data_dict[name]['from_poi_to_this_person'],data_dict[name]['to_messages'])
    data_dict[name]['ratio_shared_poi'] = calcRatio(data_dict[name]['shared_receipt_with_poi'],data_dict[name]['to_messages'])

print ("###")
print ("Feature selection analysis on email features")
print ("###")

# Creates a list of email features
features_email_list = ['poi',
                       'from_this_person_to_poi','from_poi_to_this_person','shared_receipt_with_poi','from_messages','to_messages',
                       'ratio_from_poi','ratio_to_poi','ratio_shared_poi']

# Evaluates the performance of the classifier
clf = tree.DecisionTreeClassifier()
test_classifier(clf, my_dataset, features_email_list)

# Displays the importance of features
print("Features importance for email features:")
for importance, feature in sorted(zip(clf.feature_importances_, features_email_list[1:]),reverse=True):
    print ('{}: {:.3}'.format(feature, importance))

print ("###")
print ("Univariate feature selection analysis")
print ("###")

# Extracts the labels and features used by the POI classifier
data = featureFormat(data_dict, features_list,remove_all_zeroes=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Creates a pipeline of two steps.
pipe = Pipeline([('sel', SelectKBest()),
                 ('clf', tree.DecisionTreeClassifier())])

### Parameters
k = [1,2,3,4,5]
max_depth = [2,4,6]
min_samples_split = [2,10,20]

# Creates Parameter Space
parameters = [{'sel__k':k},
              {'clf__max_depth':max_depth,
               'clf__min_samples_split':min_samples_split}]

# Conducts Parameter Optmization With Pipeline
# Creating a grid search object
clf = GridSearchCV(pipe, parameters, scoring='recall', cv=5)

# Fits the grid search
clf.fit(features_train, labels_train)

# Prints the Best Parameters
print('Best score:', clf.best_score_)
print('Best estimator:', clf.best_estimator_)
    
print ("###")
print ("Feature selection")
print ("###")

# Adds the newly selected email features to the list of features
append_feature_list = ['ratio_from_poi','ratio_to_poi']
for feature in append_feature_list:
    features_list.append(feature)
    
# Removes the non selected email features from the list of features
remove_feature_list = ['from_this_person_to_poi','from_poi_to_this_person','from_messages']
for feature in remove_feature_list:
    features_list.remove(feature)

###################
### Classifiers ###
###################
print ("###########")
print ("Classifiers")
print ("###########")

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# List of classifiers
classifiers = [GaussianNB(),
               tree.DecisionTreeClassifier(),
               ensemble.AdaBoostClassifier(),
               ensemble.RandomForestClassifier(),
               neighbors.KNeighborsClassifier(),
               SVC(gamma='auto')]

# Evaluates the performance for each classifier
for classifier in classifiers:
    clf = classifier
    test_classifier(clf, my_dataset, features_list)
    
########################
### Algorithm Tuning ###
########################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Extracts the labels and features
data = featureFormat(data_dict, features_list,remove_all_zeroes=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

'''
### THIS IS NOT USED
scaler = MinMaxScaler()
rescaled_data = scaler.fit_transform(data)
rescaled_labels, rescaled_features = targetFeatureSplit(rescaled_data)
features_train, features_test, labels_train, labels_test = train_test_split(rescaled_features, rescaled_labels, test_size=0.3, random_state=42)
'''

################################
### Pipeline: Scaler/PCA/KNN ###
################################
print ("#############")
print ("Pipeline: KNN")
print ("#############")

# Creates an standardscaler object
scaler = MinMaxScaler()

# Creates a pca object
pca = decomposition.PCA()

# Creates a DecisionTreeClassifier
knn = neighbors.KNeighborsClassifier()

# Creates a pipeline of three steps. First, standardizing the data.
# Second, tranforms the data with PCA.
# Third, trains a Decision Tree Classifier on the data.
pipe = Pipeline(steps=[('scaler', scaler),
                       ('pca', pca),
                       ('knn', knn)])

# Creates Parameter Space
# Creating a list of a sequence of integers from 1 to the number of features + 1
n_components = list(range(1,len(features_list),1))

# Creates lists of parameter for KNN Classifier
n_neighbors = list(range(1,10))

# Creates a dictionary of all the parameter options 
# Note that we can access the parameters of steps of a pipeline by using '__’
parameters = dict(pca__n_components=n_components,
                  knn__n_neighbors=n_neighbors,
				  )

# Conducts Parameter Optmization With Pipeline
# Creating a grid search object
clf = GridSearchCV(pipe, parameters, scoring='recall', cv=5)

# Fits the grid search
clf.fit(features_train, labels_train)

# Prints the Best Parameters
print('Best n_neighbors:', clf.best_estimator_.get_params()['knn__n_neighbors'])
print('Best number of components:', clf.best_estimator_.get_params()['pca__n_components'])
print(clf.best_estimator_.get_params()['knn'])

#############################
### Classifier Tuning: DT ###
#############################
print ("#####################")
print ("Classifier Tuning: DT")
print ("#####################")
  
### GridSearchCV DT
criterion = ['gini', 'entropy']
max_depth = [2,4,6,8,10,15,20,50]
min_samples_split = [2,10,20,30,40]

parameters = {'criterion':criterion,
              'max_depth':max_depth,
              'min_samples_split':min_samples_split}

clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, scoring='recall', cv=5)
clf.fit(features_train, labels_train)

# Viewing The Best Parameters
print('Best criterion:', clf.best_estimator_.get_params()['criterion'])
print('Best max_depth:', clf.best_estimator_.get_params()['max_depth'])
print('Best min_samples_split:', clf.best_estimator_.get_params()['min_samples_split'])

'''
#############################
### Classifier Tuning: AB ###
#############################
print ("#####################")
print ("Classifier Tuning: AB")   
print ("#####################")

### GridSearchCV AdaBoost
parameters = {"n_estimators": np.arange(10,300,10),
              "learning_rate": [0.01, 0.05, 0.1, 1]
             }

# run grid search
clf = GridSearchCV(ensemble.AdaBoostClassifier(), parameters, scoring='recall', cv=5)
clf.fit(features_train, labels_train)

# Viewing The Best Parameters
print('')
print('Best n_estimators:', clf.best_estimator_.get_params()['n_estimators'])
print('Best learning_rate:', clf.best_estimator_.get_params()['learning_rate'])

##############################
### Classifier Tuning: KNN ###
##############################
print ("######################")
print ("Classifier Tuning: KNN")  
print ("######################")

### GridSearchCV KNN
parameters = {'n_neighbors': list(range(1,10))}

# run grid search
clf = GridSearchCV(neighbors.KNeighborsClassifier(), parameters, scoring='recall', cv=5)
clf.fit(features_train, labels_train)

# Viewing The Best Parameters
print('')
print('Best n_neighbors:', clf.best_estimator_.get_params()['n_neighbors'])
'''

############################
### Optimized Classifier ###
############################
print ("####################")
print ("Optimized Classifier")  
print ("####################")

clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=6,min_samples_split=2)
test_classifier(clf, my_dataset, features_list)

# Prints the feature importance
print("Features importance:")
for importance, feature in sorted(zip(clf.feature_importances_, features_list[1:]),reverse=True)[:5]:
    print ('{}: importance = {:.3}'.format(feature, importance))

'''
pipe = Pipeline([('sel', SelectKBest(k=5)),
                 ('clf', tree.DecisionTreeClassifier(criterion='gini',max_depth=None,min_samples_split=2))])
test_classifier(pipe, my_dataset, features_list)

features_list = ['poi','other','ratio_to_poi']
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=6,min_samples_split=2)
test_classifier(clf, my_dataset, features_list)
      
clf = ensemble.AdaBoostClassifier(learning_rate=1,n_estimators=20)
test_classifier(clf, my_dataset, features_list)

clf = neighbors.KNeighborsClassifier(n_neighbors=1)
test_classifier(clf, my_dataset, features_list)
'''

##################################################
### Dump Classifier, Dataset and Features List ###
##################################################

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

print ("Total elapsed time:", round(time.time()-t0, 3), "s")