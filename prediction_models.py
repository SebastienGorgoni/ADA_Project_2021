"""
ADVANCED DATA ANALYSIS 2021

EVENT BASED TRADING ALGORITHM: A REDDIT CASE

Authors: Sebastien Gorgoni & Liam Svoboba

File Name: prediction_models.py

This is an external file for main.py that determine all classification models required. The models are:
    
    - Logistic Regression
    - Gradient Boosting
    - Adaptive Boosting
    - Support Vector Machine
    - Random Forest
    - K Nearest Neighbours
    - Stochastic Gradient Descent
    - Multi Layer Perceptron
    - Naive Bayes
    - Deep Neural Network (DNN/ANN)
    - Long Short Term Memory (LSTM)

Except for LSTM/DNN, the models can be executed with their default hyperparameters, but tuning with grid search and random search can be done as well.  
For each models, metrics such as accuracy, precision, recall, f1, mse and mae have been computed.

Useful Source:
     
    - https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/blob/master/HPO_Classification.ipynb
    - https://www.kaggle.com/hatone/gradientboostingclassifier-with-gridsearchcv)

"""

# Import the required libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from scipy.stats import randint as sp_randint
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
import matplotlib.pyplot as plt

# =============================================================================
# Logistic Regression
# =============================================================================

def logistic(hpt, X_train, X_test, y_train, y_test, measure_predt):
    
    np.random.seed(42)
    
    lr = LogisticRegression()
    
    #Baseline (no hyperparameter tuning)
    if hpt == "baseline":
    
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        
    #Grid Search
    elif hpt == "grid_search":
        
        params = {'penalty': ['l1','l2'], 
                  'tol' : [1e-4,1e-5], 
                  'max_iter' : [10,100,1000], 
                  'fit_intercept' : [True, False],
                  'C': [0.001,0.01,0.1,1,10,100,1000]
        }
        grid = GridSearchCV(lr, param_grid=params, scoring=measure_predt)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        
    #Random Search
    elif hpt == "random_search":

        params = {'penalty': ['l1','l2'], 
                  'tol' : [1e-4,1e-5], 
                  'max_iter' : [10,100,1000], 
                  'fit_intercept' : [True, False],
                  'C': [0.001,0.01,0.1,1,10,100,1000]
        }
        random = RandomizedSearchCV(lr, param_distributions=params, scoring=measure_predt)
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
        
    else:
        raise ValueError("No appropriate hyperparameter tuning method was selected, select between: [baseline, grid_search, random_search]")
    
    
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    print(y_pred)
    print("\n Accuracy score Logistic: ", acc)
    print("\n Precision score Logistic: ", pre)
    print("\n Recall score Logistic: ", rec)
    print("\n F1 score Logistic: ", f1)
    print("\n MSE Logistic: ", mse)
    print("\n MAE Logistic: ", mae)
    
    return (acc, pre, rec, f1, mse, mae, y_pred)

# =============================================================================
# Gradient Boosting
# =============================================================================

def gradboost(hpt, X_train, X_test, y_train, y_test, measure_predt):
    
    np.random.seed(42)
    
    gb = GradientBoostingClassifier()
    
    #Baseline (no hyperparameter tuning)
    if hpt == "baseline":
    
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        
    #Grid Search
    elif hpt == "grid_search":
        
        params = {'loss': ['deviance', 'exponential'],
                  'n_estimators' : [100,200,500],
                  'max_features' : ['log2','sqrt'],
                  'max_depth' : [3,10,50] 
        }
        grid = GridSearchCV(gb, params, cv = 5, scoring=measure_predt)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        
    #Random Search
    elif hpt == "random_search":
        
        params = {'loss': ['deviance', 'exponential'],
                  'n_estimators' : [100,200,500],
                  'max_features' : ['log2','sqrt'],
                  'max_depth' : [3,10,50] 
        }
        random = RandomizedSearchCV(gb, params, cv = 5, scoring=measure_predt)
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
        
    else:
        raise ValueError("No appropriate hyperparameter tuning method was selected, select between: [baseline, grid_search, random_search]")
     

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    print(y_pred)
    print("\n Accuracy score Gradient Boosting: ", acc)
    print("\n Precision score Gradient Boosting: ", pre)
    print("\n Recall score Gradient Boosting: ", rec)
    print("\n F1 score Gradient Boosting: ", f1)
    print("\n MSE Gradient Boosting: ", mse)
    print("\n MAE Gradient Boosting: ", mae)
    
    return (acc, pre, rec, f1, mse, mae, y_pred)

# =============================================================================
# Adaptive Boosting
# =============================================================================

def adaboost(hpt, X_train, X_test, y_train, y_test, measure_predt):
    
    np.random.seed(42)
    
    ada = AdaBoostClassifier()
    
    #Baseline (no hyperparameter tuning)
    if hpt == "baseline":
        
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)
    
    #Grid Search    
    elif hpt == "grid_search":
        
        params = {'n_estimators' : [50,100,150],
                  #'random_state' : [None], 
                  'learning_rate' : [1.,0.8,0.5],
                  'algorithm' : ['SAMME','SAMME.R']
        }
        grid = GridSearchCV(ada, params, cv = 5, scoring=measure_predt)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        
    #Grid Search    
    elif hpt == "random_search":
        
        params = {'n_estimators' : [50,100,150],
                  #'random_state' : [None], 
                  'learning_rate' : [1.,0.8,0.5],
                  'algorithm' : ['SAMME','SAMME.R']
        }
        random = RandomizedSearchCV(ada, params, cv = 5, scoring=measure_predt)
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
        
    else:
        raise ValueError("No appropriate hyperparameter tuning method was selected, select between: [baseline, grid_search, random_search]")
    
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    print(y_pred)
    print("\n Accuracy score ADABoosting: ", acc)
    print("\n Precision score ADABoosting: ", pre)
    print("\n Recall score ADABoosting: ", rec)
    print("\n F1 score ADABoosting: ", f1)
    print("\n MSE Gradient ADABoosting: ", mse)
    print("\n MAE Gradient ADABoosting: ", mae)
    
    return (acc, pre, rec, f1, mse, mae, y_pred)

# =============================================================================
# Support Vector Machine
# =============================================================================
        
def svm(hpt, X_train, X_test, y_train, y_test, measure_predt):
    
    np.random.seed(42)
    
    svm = SVC()
    
    #Baseline (no hyperparameter tuning)
    if hpt == "baseline":
       
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
     
    #Grid Search    
    elif hpt == "grid_search":
       
        params = {
            'C': [0.001,0.01,0.1,1,10,100],
            "kernel":['linear','poly','rbf','sigmoid']
        }   
        grid = GridSearchCV(svm, param_grid=params, cv=3, scoring=measure_predt)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
    
    #Random Search    
    elif hpt == "random_search":
       
        params = {
            'C': [0.001,0.01,0.1,1,10,100],
            "kernel":['linear','poly','rbf','sigmoid']
        }
        n_iter_search=10 
        random = RandomizedSearchCV(svm, param_distributions=params, n_iter=n_iter_search, cv=3, scoring=measure_predt)
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
     
    else:
        raise ValueError("No appropriate hyperparameter tuning method was selected, select between: [baseline, grid_search, random_search]")
    
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    print(y_pred)
    print("\n Accuracy score SVM: ", acc)
    print("\n Precision score SVM: ", pre)
    print("\n Recall score SVM: ", rec)
    print("\n F1 score SVM: ", f1)
    print("\n MSE SVM: ", mse)
    print("\n MAE SVM: ", mae)
    
    return (acc, pre, rec, f1, mse, mae, y_pred)

# =============================================================================
# Random Forest
# =============================================================================

def randomforest(hpt, X_train, X_test, y_train, y_test, measure_predt):
    
    np.random.seed(42)
    
    forest = RandomForestClassifier()
    
    #Baseline (no hyperparameter tuning)
    if hpt == "baseline":
       
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
    
    #Grid Search    
    elif hpt == "grid_search":
        
        params = {
           'n_estimators': [10, 20, 30],
            #'max_features': ['sqrt',0.5],
            'max_depth': [15,20,30,50],
            #'min_samples_leaf': [1,2,4,8],
            #"bootstrap":[True,False],
            "criterion":['gini','entropy']
        }
        grid = GridSearchCV(forest, param_grid=params, cv=3, scoring=measure_predt)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        
    #Random Search    
    elif hpt == "random_search":
        
        params = {
            "n_estimators": sp_randint(10,100),
            #"max_features":sp_randint(1,64),
            "max_depth": sp_randint(5,50),
            "min_samples_split":sp_randint(2,11),
            "min_samples_leaf":sp_randint(1,11),
            "criterion":['gini','entropy']
        }
        n_iter_search=10 
        random = RandomizedSearchCV(forest, param_distributions=params, n_iter=n_iter_search, cv=3, scoring=measure_predt)
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
        
    else:
        raise ValueError("No appropriate hyperparameter tuning method was selected, select between: [baseline, grid_search, random_search, hyperband, bo_gp]")
    
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    print(y_pred)
    print("\n Accuracy score Randomforest: ", acc)
    print("\n Precision score Randomforest: ", pre)
    print("\n Recall score Randomforest: ", rec)
    print("\n F1 score Randomforest: ", f1)
    print("\n MSE Randomforest: ", mse)
    print("\n MAE Randomforest: ", mae)
    
    return (acc, pre, rec, f1, mse, mae, y_pred)

# =============================================================================
# K Nearest Neighbours
# =============================================================================

def knn(hpt, X_train, X_test, y_train, y_test, measure_predt):
    
    np.random.seed(42)
    
    knn = KNeighborsClassifier()    

    #Baseline (no hyperparameter tuning)
    if hpt == "baseline":
       
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
    
    #Grid Search    
    elif hpt == "grid_search":
        
        params = {
            'n_neighbors': range(1,20),
        }
        grid = GridSearchCV(knn, param_grid=params, cv=3, scoring=measure_predt)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        
    #Random Search    
    elif hpt == "random_search":
        
        params = {
            'n_neighbors': range(1,20),
        }
        n_iter_search=10 
        random = RandomizedSearchCV(knn, param_distributions=params, n_iter=n_iter_search, cv=3, scoring=measure_predt)
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
        
    else:
        raise ValueError("No appropriate hyperparameter tuning method was selected, select between: [baseline, grid_search, random_search, hyperband, bo_gp]")
    
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    print(y_pred)
    print("\n Accuracy score KNN: ", acc)
    print("\n Precision score KNN: ", pre)
    print("\n Recall score KNN: ", rec)
    print("\n F1 score KNN: ", f1)
    print("\n MSE KNN: ", mse)
    print("\n MAE KNN: ", mae)
    
    return (acc, pre, rec, f1, mse, mae, y_pred)

# =============================================================================
# Stochastic Gradient Descent
# =============================================================================

def sgd(hpt, X_train, X_test, y_train, y_test, measure_predt):
    
    np.random.seed(42)
    
    sgd = SGDClassifier()
    
    #Baseline (no hyperparameter tuning)
    if hpt == "baseline":
        
        sgd.fit(X_train, y_train)
        y_pred = sgd.predict(X_test)
       
    #Grid Search    
    elif hpt == "grid_search":
        
        params = {
          'penalty':['l2', 'l1'],
          'alpha':[10 ** x for x in range(-4, 2)]
        }
        grid = GridSearchCV(sgd, params, n_jobs=-1, verbose=2, cv=3, scoring=measure_predt)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        
    #Random Search    
    elif hpt == "random_search":
        
        params = {
          'penalty':['l2', 'l1'],
          'alpha':[10 ** x for x in range(-4, 2)]
        }
        random = RandomizedSearchCV(sgd, params, n_jobs=-1, verbose=2, cv=3, scoring=measure_predt)
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
        
    else:
        raise ValueError("No appropriate hyperparameter tuning method was selected, select between: [baseline, grid_search, random_search, hyperband, bo_gp]")
     
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    print(y_pred)
    print("\n Accuracy score SGD: ", acc)
    print("\n Precision score SGD: ", pre)
    print("\n Recall score SGD: ", rec)
    print("\n F1 score SGD: ", f1)
    print("\n MSE SGD: ", mse)
    print("\n MAE SGD: ", mae)

    return (acc, pre, rec, f1, mse, mae, y_pred)

# =============================================================================
# Multi-Layer Perceptron
# =============================================================================

def mlp(hpt, X_train, X_test, y_train, y_test, measure_predt):
    
    np.random.seed(42)
    
    mlp = MLPClassifier()
    
    #Baseline (no hyperparameter tuning)
    if hpt == "baseline":
        
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
       
    #Grid Search    
    elif hpt == "grid_search":
        
        params = {'activation' : ['logistic','tanh','relu'],
                  'hidden_layer_sizes' : [(5,),(10,)], 
                  'max_iter' : [200,300],
                  'alpha' : [0.0001,0.0005]
        }
        grid = GridSearchCV(mlp, params, n_jobs=-1, verbose=2, cv=3, scoring=measure_predt)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        
    #Random Search    
    elif hpt == "random_search":
        
        params = {'activation' : ['logistic','tanh','relu'],
                  'hidden_layer_sizes' : [(5,),(10,)], 
                  'max_iter' : [200,300],
                  'alpha' : [0.0001,0.0005]
        }
        random = RandomizedSearchCV(mlp, params, n_jobs=-1, verbose=2, cv=3, scoring=measure_predt)
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
        
    else:
        raise ValueError("No appropriate hyperparameter tuning method was selected, select between: [baseline, grid_search, random_search, hyperband, bo_gp]")
          
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    print(y_pred)
    print("\n Accuracy score MLP: ", acc)
    print("\n Precision score MLP: ", pre)
    print("\n Recall score MLP: ", rec)
    print("\n F1 score MLP: ", f1)
    print("\n MSE MLP: ", mse)
    print("\n MAE MLP: ", mae)

    return (acc, pre, rec, f1, mse, mae, y_pred)

# =============================================================================
# Naive Bayes
# =============================================================================

def naivebayes(hpt, X_train, X_test, y_train, y_test, measure_predt):
    
    np.random.seed(42)
    
    gnb = GaussianNB()
    
    #Baseline (no hyperparameter tuning)
    if hpt == "baseline":
        
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        
    #Grid Search    
    elif hpt == "grid_search":
        
        params = {'var_smoothing': np.logspace(0,-9, num=20)}
        grid = GridSearchCV(gnb, params, verbose=2, cv=3, scoring=measure_predt)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)

   #Random Search    
    elif hpt == "random_search": 
        
        params = {'var_smoothing': np.logspace(0,-9, num=20)}
        random = RandomizedSearchCV(gnb, params, verbose=2, cv=3, scoring=measure_predt)
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
        
    else:
        raise ValueError("No appropriate hyperparameter tuning method was selected, select between: [baseline, grid_search, random_search, hyperband, bo_gp]")
    
    print(y_pred)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    print(y_pred)
    print("\n Accuracy score Naive Bayes: ", acc)
    print("\n Precision score Naive Bayes: ", pre)
    print("\n Recall score Naive Bayes: ", rec)
    print("\n F1 score Naive Bayes: ", f1)
    print("\n MSE Naive Bayes: ", mse)
    print("\n MAE Naive Bayes: ", mae)

    return (acc, pre, rec, f1, mse, mae, y_pred)

# =============================================================================
# Deep Neural Network
# =============================================================================

def ann(X_train, X_test, y_train, y_test, namefig):
    
    tf.random.set_seed(42)
    
    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid')) 
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    
    model.compile(optimizer = 'sgd', loss='binary_crossentropy', metrics=["accuracy"])
    
    history = model.fit(X_train, y_train, batch_size=5, epochs=50, validation_split=0.2, callbacks=[callback])
    
    #y_pred = model.predict_classes(X_test)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    print(y_pred)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    print(y_pred)
    print("\n Accuracy score ANN: ", acc)
    print("\n Precision score ANN: ", pre)
    print("\n Recall score ANN: ", rec)
    print("\n F1 score ANN: ", f1)
    print("\n MSE ANN: ", mse)
    print("\n MAE ANN: ", mae)
    
    plt.figure(figsize=(10,7))
    #Accuracy
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy DNN')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # "Loss"
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss DNN')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'Plot_model/ann_{namefig}')
    plt.show()
    plt.close()
    
    return (acc, pre, rec, f1, mse, mae, y_pred)

# =============================================================================
# Long-Short Term Memory
# =============================================================================
    
def lstm(dataframe, namefig):
    
    tf.random.set_seed(42)
    
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values.reshape(-1,1)
    
    in_sample = dataframe.loc[(dataframe.index < pd.to_datetime('2020-01-01'))]
    #out_sample = dataframe.loc[(dataframe.index >= pd.to_datetime('2020-01-01'))]
    
    time_steps = 6
    k = 0
    x_final = []
    y_final = []
    for k in range(X.shape[0]-time_steps-1):
        x_final.append(X[k:k+time_steps,:])
        y_final.append(y[(k + time_steps + 1):(k + time_steps + 2), :])
    x_final=np.stack(x_final)
    y_final=np.concatenate(y_final)
    
    #Set the train and test values (train+validation: 90% / test: 10%)
    tr = in_sample.shape[0] - time_steps - 1
    X_train = x_final[:tr,:,:] 
    X_test = x_final[tr:,:,:] 
    y_train = y_final[:tr,:] 
    y_test = y_final[tr:,:] 

    #Create the LSTM Model
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(20),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the Model with setting an exmplict learning rate
    learning_rate = 0.001
    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    
    history = lstm_model.fit(x=X_train, y=y_train, epochs=50, validation_split=0.2, callbacks=[callback])
    
    #Evaluate the model
    lstm_model.evaluate(X_test, y_test)
    
    #Predict the output
    y_pred = (lstm_model.predict(X_test) > 0.5).astype("int32")
    
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    print(y_pred)
    print("\n Accuracy score LSTM: ", acc)
    print("\n Precision score LSTM: ", pre)
    print("\n Recall score LSTM: ", rec)
    print("\n F1 score LSTM: ", f1)
    print("\n MSE LSTM: ", mse)
    print("\n MAE LSTM: ", mae)
    
    plt.figure(figsize=(10,7))
    #Accuracy
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy LSTM')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    #Loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss LSTM')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'Plot_model/lstm_{namefig}')
    plt.show()
    plt.close
    
    return (acc, pre, rec, f1, mse, mae, y_pred)

