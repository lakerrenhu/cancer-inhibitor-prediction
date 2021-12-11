# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 21:08:51 2021

@author: laker
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "C:\\Users\\laker\\Desktop\\pc2\\cs_class\\CAP5510_bioinformatics\\project\\archive\\"]).decode("utf8"))
from sklearn.neural_network import MLPClassifier
import h5py
from scipy import sparse

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

############################### import the dataset ######################################

print("Modules imported!")
print("Collecting Data...")
hf = h5py.File("C:\\Users\\laker\\Desktop\\pc2\\cs_class\\CAP5510_bioinformatics\\project\\archive\\cdk2.h5", "r")
ids = hf["chembl_id"].value # the name of each molecules
ap = sparse.csr_matrix((hf["ap"]["data"], hf["ap"]["indices"], hf["ap"]["indptr"]), shape=[len(hf["ap"]["indptr"]) - 1, 2039])
mg = sparse.csr_matrix((hf["mg"]["data"], hf["mg"]["indices"], hf["mg"]["indptr"]), shape=[len(hf["mg"]["indptr"]) - 1, 2039])
tt = sparse.csr_matrix((hf["tt"]["data"], hf["tt"]["indices"], hf["tt"]["indptr"]), shape=[len(hf["tt"]["indptr"]) - 1, 2039])
features = sparse.hstack([ap, mg, tt]).toarray() # the samples' features, each row is a sample, and each sample has 3*2039 features
labels = hf["label"].value # the label of each molecule
print("Data collected. Training ANN...")



#Train, validation and holdout split
#train_d, test_d = train_test_split(data_ori,train_size = 0.01, random_state = 42)
X_train, X_test, y_train, y_test = [features[:-100], features[-100:], labels[:-100], labels[-100:]]

################################ ANN #################################
ann = MLPClassifier(verbose=True, warm_start=True, max_iter=200)
ann = SVC()
ann.fit(X_train, y_train)
print("ANN trained. Testing ANN...")
tin = X_test
tout = y_test
tp = 0
tn = 0
fp = 0
fn = 0
for i, a in enumerate(tin):
	if ann.predict([a])[0] == tout[i]:
		if tout[i] == 1:
			tp += 1
		else:
			tn += 1
	else:
		if tout[i] == 1:
			fp += 1
		else:
			fn += 1
scores = cross_val_score(ann, features, labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) # Accuracy: 0.88 

################################### transform data into similarity matrix

from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix, csr_matrix

def cosine(plays):
    normalized = normalize(plays)
    return normalized.dot(normalized.T)


def bhattacharya(plays):
    plays.data = np.sqrt(plays.data)
    return cosine(plays)


def ochiai(plays):
    plays = csr_matrix(plays)
    plays.data = np.ones(len(plays.data))
    return cosine(plays)


def bm25_weight(data, K1=1.2, B=0.8):
    """ Weighs each row of the matrix data by BM25 weighting """
    # calculate idf per term (user)
    N = float(data.shape[0])
    idf = np.log(N / (1 + np.bincount(data.col)))

    # calculate length_norm per document (artist)
    row_sums = np.squeeze(np.asarray(data.sum(1)))
    average_length = row_sums.sum() / N
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    ret = coo_matrix(data)
    ret.data = ret.data * (K1 + 1.0) / (K1 * length_norm[ret.row] + ret.data) * idf[ret.col]
    return ret


def bm25(plays):
    plays = bm25_weight(plays)
    return plays.dot(plays.T)

def get_largest(row, N=10):
    if N >= row.nnz:
        best = zip(row.data, row.indices)
    else:
        ind = np.argpartition(row.data, -N)[-N:]
        best = zip(row.data[ind], row.indices[ind])
    return sorted(best, reverse=True)


def calculate_similar_artists(similarity, artists, artistid):
    neighbours = similarity[artistid]
    top = get_largest(neighbours)
    return [(artists[other], score, i) for i, (score, other) in enumerate(top)]



similarity = bm25(coo_matrix(features)).todense()



U, sigma, Vt = np.linalg.svd(similarity[:,:200], full_matrices=False)
sigma = np.diag(sigma)
print(U.shape,sigma.shape,Vt.shape)


################################## different classifiers ##############################################
#Train, validation and holdout split

X_train, X_test, Y_train, Y_test = train_test_split(U,labels,train_size = 0.80, random_state = 42)



from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}


def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
def procenterror(y_predicted, y_real):
     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)


names = [
         'ElasticNet',
         'SVC',
         'kSVC',
         'KNN',
         'DecisionTree',
         'RandomForestClassifier',
         'GridSearchCV',
         'HuberRegressor',
         'Ridge',
         'Lasso',
         'LassoCV',
         'Lars',
         'BayesianRidge',
         'SGDClassifier',
         'RidgeClassifier',
         'LogisticRegression',
         'OrthogonalMatchingPursuit',
         #'RANSACRegressor',
         ]

classifiers = [
    ElasticNetCV(cv=10, random_state=0),
    SVC(),
    SVC(kernel = 'rbf', random_state = 0),
    KNeighborsClassifier(n_neighbors = 1),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators = 200),
    GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),
    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),
    Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),
    Lasso(alpha=0.05),
    LassoCV(),
    Lars(n_nonzero_coefs=10),
    BayesianRidge(),
    SGDClassifier(),
    RidgeClassifier(),
    LogisticRegression(),
    OrthogonalMatchingPursuit(),
    #RANSACRegressor(),
]
correction= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

temp=zip(names,classifiers,correction)
print(temp)

for name, clf,correct in temp:
    regr=clf.fit(X_train,Y_train)
    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)
    print(name,'%error',procenterror(regr.predict(X_test),Y_test),'rmsle',rmsle(regr.predict(X_test),Y_test))


    # Confusion Matrix
    print(name,'Confusion Matrix')
    print(confusion_matrix(Y_test, np.round(regr.predict(X_test) ) ) )
    print('--'*40)

    # Classification Report
    print('Classification Report')
    print(classification_report(Y_test,np.round( regr.predict(X_test) ) ))

    # Accuracy
    print('--'*40)
    logreg_accuracy = round(accuracy_score(Y_test, np.round( regr.predict(X_test) ) ) * 100,2)
    print('Accuracy', logreg_accuracy,'%')

################### ###################### Qlattice #################################### ##############
import feyn

ql = feyn.connect_qlattice()

# Seeding the QLattice for reproducible results
ql.reset(42)

X = U   
Y=labels
xy = np.hstack((X,Y.reshape(1890,1)))

XY = pd.DataFrame(xy)
XY.columns = ['a0','a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','a16','a17','a18',
              'a19','a20','a21','a22','a23','a24','a25','a26','a27','a28','a29','a30','a31','a32','a33','a34','a35','a36',
              'a37','a38','a39','a40','a41','a42','a43','a44','a45','a46','a47','a48','a49','a50','a51','a52','a53',
              'a54','a55','a56','a57','a58','a59','a60','a61','a62','a63','a64','a65','a66','a67','a68','a69',
              'a70','a71','a72','a73','a74','a75','a76','a77','a78','a79','a80','a81','a82','a83','a84','a85','a86',
              'a87','a88','a89','a90','a91','a92','a93','a94','a95','a96','a97','a98','a99','a100',
              'a101','a102','a103','a104','a105','a106','a107','a108','a109','a110','a111','a112','a113','a114','a115',
              'a116','a117','a118','a119','a120','a121','a122','a123','a124','a125','a126','a127','a128','a129',
              'a130','a131','a132','a133','a134','a135','a136','a137','a138','a139','a140','a141','a142','a143','a144',
              'a145','a146','a147','a148','a149','a150','a151','a152','a153','a154','a155','a156','a157','a158','a159',
              'a160','a161','a162','a163','a164','a165','a166','a167','a168','a169','a170','a171','a172','a173','a174',
              'a175','a176','a177','a178','a179','a180','a181','a182','a183','a184','a185','a186','a187','a188','a189',
              'a190','a191','a192','a193','a194','a195','a196','a197','a198','a199','a200']

#Train, validation and holdout split
xy_train, xy_test = train_test_split(XY,train_size = 0.80, random_state = 42)
#use AIC as a selection criterion prior to updating the QLattice with the best graphs.
#This is a regression task, which is default for auto_run
models = ql.auto_run(xy_train,'a200' ,  criterion='aic')
# Select the best Model
model_base = models[0]
models[0]  # loss = 1.79E-01, 4 features exist
model_base.plot_regression(xy_train)

# accuracy
model_base.accuracy_score(xy_train)  # training RMSE = 0.74537
model_base.accuracy_score(xy_test)  # training RMSE = 0.68518

#model_base.rmse(XY)  #  test RMSE = 0.2908291770568081

   
    
# Confusion Matrix

print(confusion_matrix(xy_test['a200'], np.round(model_base.predict(xy_test.loc[:, xy_test.columns != 'a200']) ) ) )
print('--'*40)

# Classification Report
print('Classification Report')
print(classification_report(xy_test['a200'],np.round( model_base.predict(xy_test.loc[:, xy_test.columns != 'a200']) ) ))

######################################## CNN #######################################################

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras

'''using tensorflow under Anaconda needs to activate 'tensorflow' first.
input command in Anaconda Prompt (anaconda3): conda activate tensorflow '''

# split data into training and test set

xtrain, xtest, ytrain, ytest = train_test_split(U,labels,train_size = 0.80, random_state = 42)

#Defining and fitting the model
#We'll define the Keras sequential model and add a one-dimensional convolutional layer. 
#Input shape becomes as it is defined above (6,1).
#We'll add Flatten and Dense layers and compile it with optimizers.

model = Sequential()
model.add(Conv1D(32, 2, activation="relu", input_shape=(200, 1)))
model.add(Flatten())
model.add(Dense(64, activation="softmax"))
model.add(Dense(1))
#loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss='mse', optimizer="adam", metrics=[keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)])
 
model.summary()

#Next, we'll fit the model with train data.
xtrain = np.asarray(xtrain)  # convert 2d numpy matrix to 2d numpy array
xtest  =  np.asarray(xtest)

xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)
#ytrain = ytrain.reshape(ytrain.shape[0],  1)
#ytest = ytest.reshape(ytest.shape[0],  1)

ytrain.shape
model.fit(xtrain, ytrain, batch_size=12,epochs=200, verbose=0)

#Predicting and visualizing the results
#Now we can predict the test data with the trained model.


#We can evaluate the model, check the mean squared error rate (MSE) of the 
# predicted result, and visualize the result in a plot.

print(model.evaluate(xtrain, ytrain))
#binary_accuracy: 0.9868

print(model.evaluate(xtest, ytest))
#binary_accuracy: 0.8069

# Confusion Matrix

print(confusion_matrix(ytest, np.round(model.predict(xtest) ) ) )
print('--'*40)
#[[ 95  28]
# [ 38 217]]

# Classification Report
print('Classification Report')
print(classification_report(ytest,np.round(model.predict(xtest) ) ))

#precision    recall  f1-score   support

#           0       0.71      0.77      0.74       123
#           1       0.89      0.85      0.87       255

#    accuracy                           0.83       378
#   macro avg       0.80      0.81      0.81       378
# weighted avg       0.83      0.83      0.83       378



















