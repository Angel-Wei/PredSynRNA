#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:14:49 2020

@author: anqi
"""
'''This script is used for parameter optimization and training process of
logistic regression (LR), Random Forest(RF), Support Vector Machine(SVM),
and XGBoost(XGB).
'''
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
import joblib
from sklearn.metrics import precision_recall_curve
from sklearn.utils import shuffle 
from sklearn.preprocessing import MinMaxScaler
np.random.seed(12345)

def getparas():
    parser = argparse.ArgumentParser(description="progrom usage")
    parser.add_argument("-t", "--train", type=str, help="Training data")
    parser.add_argument("-i", "--independent", type=str, help="Independent test data")
    parser.add_argument("-p", "--path", type=str, help="File path to save the results")
    args = parser.parse_args()
    trainfile = args.train
    testfile = args.independent
    path = args.path
    return trainfile, testfile, path

def get_training_data(trainingfile):
    full_train = pd.read_csv(trainingfile)
    full_train = shuffle(full_train, random_state = 42)
    
    x_exp = full_train.drop(columns = ['ensembl_gene_id', 'gene_symbol', 'utr5', 'utr3', 'label']).reset_index(drop = True)
    # get column names of expression features
    exp_columns = list(x_exp.columns)
    # log2(RPKM + 1) transformation
    x_exp = round(np.log2(x_exp + 1), 2)
    x_exp = np.array(x_exp)   
    print ("Dimension of x_exp: " + str(x_exp.shape))
    y_train = full_train['label'].reset_index(drop = True)
    y_train = np.array(y_train)
    return x_exp, y_train, exp_columns

def get_test_data(testfile):
    full_test = pd.read_csv(testfile)
    full_test = shuffle(full_test, random_state = 42)
    
    x_exp = full_test.drop(columns = ['ensembl_gene_id', 'gene_symbol', 'utr5', 'utr3', 'label']).reset_index(drop = True)
    # log2(RPKM + 1) transformation
    x_exp = round(np.log2(x_exp + 1), 2)
    x_exp = np.array(x_exp)   
    print ("Dimension of x_exp: " + str(x_exp.shape))
    y_test = full_test['label'].reset_index(drop = True)
    y_test = np.array(y_test)
    return x_exp, y_test
    
def para_tuning(model,para,X,Y):
    grid_obj = GridSearchCV(model, para,  scoring = 'average_precision',cv=5)
    grid_obj = grid_obj.fit(X, Y)
    para_best = grid_obj.best_estimator_
    return(para_best)  
    
def metrics(Y_cv_test, predictions, pred_train_prob):
    accuracy = accuracy_score(Y_cv_test,predictions)
    confusion = confusion_matrix(Y_cv_test, predictions)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    specificity = TN / float( TN + FP)
    sensitivity = TP / float(FN + TP)
    strength = (specificity + sensitivity)/2
    mcc = matthews_corrcoef(Y_cv_test, predictions)
    f1 = f1_score(Y_cv_test, predictions)
    fpr, tpr, thresholds = roc_curve(Y_cv_test, pred_train_prob)
    aucvalue = auc(fpr, tpr)
    return accuracy,sensitivity,specificity,strength,mcc,f1,aucvalue

def plot_graph(cv_preds, cv_probs, name, color):
    f1 = plt.figure(figsize = (8, 8))
    f2 = plt.figure(figsize = (8, 8))
    ax1 = f1.add_subplot(111, aspect = 'equal')
    ax2 = f2.add_subplot(111, aspect = 'equal')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    # ROC curve
    fpr, tpr, thresholds_1 = roc_curve(np.asarray(cv_preds), np.asarray(cv_probs))
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color = color, label = name + feature_set + ' (AUC = %0.2f)' % roc_auc)
    ax1.legend(loc = 'lower right')
    
    # Precision recall curve
    precision, recall, thresholds_2 = precision_recall_curve(np.asarray(cv_preds), np.asarray(cv_probs))
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, color = color, label = name + feature_set + ' (AUC = %0.2f)' % pr_auc)
    ax2.legend(loc = 'lower right')
    
    ax1.set_ylabel('True Positive Rate', fontsize = 15)
    ax1.set_xlabel('False Positive Rate', fontsize = 15)
    ax1.plot([0, 1], [0, 1],'r--', color = 'silver', label = 'random guessing = 0.5')
    ax1.legend(loc = 'lower right')    
    ax2.set_ylabel('Precision', fontsize = 15)
    ax2.set_xlabel('Recall', fontsize = 15)
    
    f1.savefig(file_path + "roc_curve.pdf")
    f2.savefig(file_path + "pr_curve.pdf")

def cv(name, model, X_train, Y_train, random_state, columns, fileout):
    seed = random_state
    cv_acc,cv_spe,cv_sen,cv_str,cv_mcc,cv_f1,cv_auc = [],[],[],[],[],[],[]
    cv_preds,cv_probs = [],[]   
    kf = model_selection.KFold(n_splits = 10, shuffle = True, random_state = seed)
    for train_index, test_index in kf.split(X_train):
        X_cv_train, X_cv_test = X_train[train_index,:], X_train[test_index,:]
        Y_cv_train, Y_cv_test = Y_train[train_index], Y_train[test_index]
        model.fit(X_cv_train,Y_cv_train)
        predictions = model.predict(X_cv_test)
        pred_train_prob = model.predict_proba(X_cv_test)[:, 1]
        accuracy,sensitivity,specificity,strength,mcc,f1,aucvalue = metrics(Y_cv_test, predictions, pred_train_prob)
        cv_preds = cv_preds + Y_cv_test.tolist()
        cv_probs = cv_probs + pred_train_prob.tolist()

        cv_acc.append(accuracy)
        cv_sen.append(sensitivity)
        cv_spe.append(specificity)
        cv_str.append(strength)
        cv_mcc.append(mcc)
        cv_f1.append(f1)
        cv_auc.append(aucvalue)

    fileout.write("Accuracy_mean: "+str(np.mean(cv_acc)) + "\n")
    fileout.write("Sensitivity_mean: "+str(np.mean(cv_sen)) + "\n")
    fileout.write("Specificity_mean: "+str(np.mean(cv_spe)) + "\n")
    fileout.write("Strength_mean: "+str(np.mean(cv_str)) + "\n")
    fileout.write("MCC_mean: "+str(np.mean(cv_mcc)) + "\n")
    fileout.write("Fscore_mean: "+str(np.mean(cv_f1)) + "\n")
    fileout.write("AUC_mean: "+str(np.mean(cv_auc)) + "\n\n\n")
    
    if name == 'RF':
        #==============================================================================
        #Visualize features based on importance score
        #==============================================================================
        feature_list = list(columns)
        feature_imp = pd.Series(model.feature_importances_,index = feature_list).sort_values(ascending=False)
        
        #feature_imp
        feature_importance_file = open (file_path + "feature_importance_score.txt", "w")
        feature_importance_file.write("column_name,importance_score"+ "\n")
        for index, value in feature_imp.iteritems():
            feature_importance_file.write(index + "," + str(value) + "\n")
        feature_importance_file.close()
    return cv_preds, cv_probs

def machine_learning(trainingfile,fileout1):
    x_exp, Y_train, columns = get_training_data(trainingfile)
    # specify the X_train data
    X_train = scaler.fit_transform(x_exp)
    models = []
    para_lr = {'C' : [0.001,0.01,0.1,0.25,0.5, 0.8, 1.0, 1.2, 1.5, 2.5], 
           'penalty' : ['l2'],
           'class_weight' : [{1:1.2}]
          }
    para_rf = {'n_estimators': [15, 20, 25, 35, 45, 65, 85], 
            'max_features': ['log2', 'sqrt'], 
            'criterion': ['entropy', 'gini'],
            'max_depth': [3, 5, 10], 
            'min_samples_split': [2,4],
            'class_weight' : [{1:1.2}]
          }
    para_svm = {'C' : [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 150, 175, 256], 
            'kernel' : ['rbf'],
            'gamma' : [0.0000305176, 0.00012207, 0.000488281, 0.001953125, 0.0078125, 0.005, 0.01, 0.03125, 0.05, 0.125, 0.5, 2, 8, 32],
            'class_weight' : [{1:1.2}],
            'probability' : [True]
            }  
    para_xgb = {'n_estimators' : [1, 8, 16, 32, 64, 100, 128, 256],
            'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
            'max_depth' : [3, 5, 10],
            'scale_pos_weight': [1.2],
            'gamma': [0.5, 1, 1.5, 2, 5, 7, 10]               
            }
    
    best_para_lr = para_tuning(LogisticRegression(solver = 'lbfgs', max_iter=10000), para_lr, X_train, Y_train)
    print("Parameter for logistic regression finished")
    print (best_para_lr)
    para_file.write("Parameter for logistic regression: " + "\n" + str(best_para_lr) + "\n")
    
    best_para_rf = para_tuning(RandomForestClassifier(), para_rf, X_train, Y_train)
    print("Parameter for random forest finished")
    print (best_para_rf)
    para_file.write("Parameter for random forest: " + "\n" + str(best_para_rf) + "\n")
    
    best_para_svm = para_tuning(SVC(), para_svm, X_train, Y_train)
    print("Parameter for svm finished")
    print (best_para_svm)
    para_file.write("Parameter for svm: " + "\n" + str(best_para_svm) + "\n")

    best_para_xgb = para_tuning(XGBClassifier(), para_xgb, X_train, Y_train)
    print("Parameter for XGB finished")
    print (best_para_xgb)
    para_file.write("Parameter for XGB: " + "\n" + str(best_para_xgb) + "\n")
    
    models.append(('LR', best_para_lr,'red'))
    models.append(('RF', best_para_rf,'green'))
    models.append(('SVM', best_para_svm,'blue'))
    models.append(('XGB', best_para_xgb, 'magenta'))
    
    # output the cv_preds and cv_probs
    plot_output = open (file_path + "cv_preds_probs.txt", "w")
    plot_output.write("feature set,model,cv_preds,cv_probs")
    plot_output.write("\n")
    
    trained_models = []
    for name, model, color in models:
        fileout1.write(name+": ")
        fileout1.write(str(model))
        fileout1.write("\n")
        
        cv_preds, cv_probs = cv(name, model, X_train, Y_train, 42, columns, fileout1)
        model.fit(X_train, Y_train)
        trained_models.append((name, model))
        # save the model in the current working directory
        joblib_fname = file_path + name + '.model'
        joblib.dump(model, joblib_fname)
        plot_graph(cv_preds, cv_probs, name, color)
        
        # output the predictions: name, cv_preds, cv_probs
        for i in range(len(cv_preds)):
            plot_output.write(feature_set + name + "," + str(cv_preds[i]) + "," + str(cv_probs[i]))
            plot_output.write("\n")
    fileout1.close()
    plot_output.close()
    return trained_models

def test(models,testfile):
    x_exp, Y_test = get_test_data(testfile)
    X_test = scaler.transform(x_exp)
    fileout2 = open (file_path + "test_result.txt", "w")
    
    for name, model in models:
        predictions = model.predict(X_test)
        rounded = [round(x) for x in predictions]
        y_pred = model.predict_proba(X_test)[:, 1]
        accuracy,sensitivity,specificity,strength,mcc,f1,aucvalue = metrics(Y_test,rounded,y_pred)
        fileout2.write(str(model) + "\n")
        fileout2.write("Accuracy_mean: "+str(accuracy) + "\n")
        fileout2.write("Sensitivity_mean: "+str(sensitivity) + "\n")
        fileout2.write("Specificity_mean: "+str(specificity) + "\n")
        fileout2.write("Strength_mean: "+str(strength) + "\n")
        fileout2.write("MCC_mean: "+str(mcc) + "\n")
        fileout2.write("Fscore_mean: "+str(f1) + "\n")
        fileout2.write("AUC_mean: "+str(aucvalue) + "\n\n\n")
    fileout2.close()
       
def main():
    global file_path, para_file, scaler, feature_set
    trainingfile, testfile, file_path = getparas()
    feature_set = "expression_full"
    fileout1 = open(file_path + "train_output.txt", "w")
    para_file = open (file_path + "best_parameter.txt", "w")
    scaler = MinMaxScaler()
    
    # training, test by machine learning models
    models = machine_learning(trainingfile,fileout1)
    test(models,testfile)
if __name__ == '__main__':
        main()
