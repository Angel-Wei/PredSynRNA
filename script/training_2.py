#!/usr/bin/env python3i
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:14:47 2020

@author: anqi
"""

import argparse
import pandas as pd
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
from keras import regularizers
from keras import backend as K
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.utils import shuffle, resample
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
    # log2(RPKM + 1) transformation
    x_exp = round(np.log2(x_exp + 1), 2)
    x_exp = np.array(x_exp)
    print ("Dimension of x_exp: " + str(x_exp.shape))
    y_train = full_train['label']
    y_train = np.array(y_train).astype('int').reshape((-1,1))
    return x_exp, y_train

def resampling(x_train_data, y_train_data):
    # bootstrap the training data and get the X_train_data, Y_train_data
    train = np.concatenate((y_train_data, x_train_data), axis = 1)
    train_pos = train[train[:,0]==1]
    train_neg = train[train[:,0]==0]
    train_pos_boot = resample(train_pos, replace=True, n_samples=train_neg.shape[0])
    print ("negative: " + str(train_neg.shape))
    print("positive before bootstrap: " + str(train_pos.shape) + "; positive after bootstrap: " + str(train_pos_boot.shape))
    train_com = np.vstack((train_pos_boot,train_neg))
    np.random.shuffle(train_com)
    return np.array(train_com[:,1:]), np.array(train_com[:,0].astype('int'))
    
def get_test_data(testfile):
    full_test = pd.read_csv(testfile)
    full_test = shuffle(full_test, random_state = 42)
    
    x_exp = full_test.drop(columns = ['ensembl_gene_id', 'gene_symbol', 'utr5', 'utr3', 'label']).reset_index(drop = True)
    # log2(RPKM + 1) transformation
    x_exp = round(np.log2(x_exp + 1), 2)
    x_exp = np.array(x_exp)
    print ("Dimension of x_exp: " + str(x_exp.shape))
    y_test = full_test['label']
    y_test = np.array(y_test).astype('int').reshape((-1,1))
    return x_exp, y_test
    
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

def plot_graph(cv_preds, cv_probs, color, filepath):
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
    
    f1.savefig(filepath + "roc_curve.pdf")
    f2.savefig(filepath + "pr_curve.pdf")

def objective(params):
    model =  get_ANN_model(params)
    earlystopper = EarlyStopping(monitor='val_loss', patience = 5, verbose = 1)
    fits = model.fit(X_train_data, Y_train_data, batch_size=2**int(params['batch_size']), epochs=100, shuffle=True, validation_data=(X_val_data,Y_val_data), callbacks=[earlystopper])
    auc = np.mean(fits.history['val_roc_auc'])
    loss = 1-auc 
    out_file = file_path + 'ANN_performance_with_diff_hyperparameter.csv'
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params])
    of_connection.close()
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

'''Calculate ROC AUC during model training, obtained from <https://github.com/nathanshartmann/NILC-at-CWI-2018>'''
def binary_PFA(y_true, y_pred, threshold=K.variable(value = 0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

''' P_TA prob true alerts for binary classifier'''
def binary_PTA(y_true, y_pred, threshold=K.variable(value = 0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

def roc_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

'''get the ANN model'''
def get_ANN_model(params):
    model = Sequential()
    model.add(Dense(int(params['hdim']), input_dim = dimension,activation="relu",kernel_regularizer = regularizers.l2(params['l2_reg'])))
    model.add(Dropout(params['drop_out']))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr=params['learning_rate'],epsilon = 10**-8)
    model.compile(loss='binary_crossentropy', optimizer = adam, metrics = [roc_auc])
    print(params)
    return model

def cv(X_train, Y_train, random_state, fileout):
    seed = random_state
    model = get_ANN_model(best)
    cv_acc,cv_spe,cv_sen,cv_str,cv_mcc,cv_f1,cv_auc = [],[],[],[],[],[],[] 
    cv_preds, cv_probs = [],[]
    kf = model_selection.KFold(n_splits = 10, shuffle = True, random_state = seed)
    for train_index, test_index in kf.split(X_train):
        X_cv_train, X_cv_test = X_train[train_index,:], X_train[test_index,:]
        Y_cv_train, Y_cv_test = Y_train[train_index], Y_train[test_index]
        
        # split the X_cv_train and Y_cv_train into 0.9 sub-training and 0.1 sub-validation
        X_subtrain,X_subval,Y_subtrain,Y_subval = train_test_split(X_cv_train, Y_cv_train, test_size = 0.1, random_state = 42)
        # bootstrap resampling of the sub-training data
        print ("Start bootstrap to get the balanced training dataset in each validation during 10-fold.")
        X, Y = resampling(X_subtrain, Y_subtrain)
        save_path = file_path + "synapse_" + name + "_" + feature_set + ".hdf5"
        checkpointer = ModelCheckpoint(filepath = save_path, verbose = 1, save_best_only = True)
        earlystopper = EarlyStopping(monitor='val_loss', patience = 5, verbose = 1)
        model.fit(X, Y, batch_size=2**int(best['batch_size']), epochs=100, shuffle=True, validation_data=(X_subval,Y_subval), callbacks=[checkpointer, earlystopper])
        
        predictions = model.predict(X_cv_test)
        predictions = [round(x[0]) for x in predictions]
        pred_train_prob = [x[0] for x in model.predict_proba(X_cv_test)]
        accuracy,sensitivity,specificity,strength,mcc,f1,aucvalue = metrics(Y_cv_test, predictions, pred_train_prob)
        cv_preds = cv_preds + [x[0] for x in Y_cv_test]
        cv_probs = cv_probs + pred_train_prob

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
    return cv_preds, cv_probs, model

def deep_learning(X_train,Y_train,random_state,fileout):
    
    color = 'red'
    
    fileout.write(name + ": " + str(best) + "\n")
    cv_preds, cv_probs, model = cv(X_train, Y_train, 42, fileout)
    plot_graph(cv_preds, cv_probs, color, file_path)
        
    # output the predictions: name, cv_preds, cv_probs
    # output the values to generate roc-curve and pr-curve
    plot_output = open (file_path + "ann_cv_preds_probs.txt", "w")
    plot_output.write("feature set,model,cv_preds,cv_probs")
    plot_output.write("\n")
    for i in range(len(cv_preds)):
        plot_output.write(feature_set + "," + name + "," + str(cv_preds[i]) + "," + str(cv_probs[i]))
        plot_output.write("\n")
    fileout.close()
    plot_output.close()
    return model

def test(model,testfile):
    X_test_exp, Y_test = get_test_data(testfile)
    X_test = scaler.transform(X_test_exp)
    predictions = model.predict(X_test)
    rounded = [round(x[0]) for x in predictions]
    y_pred = [x[0] for x in model.predict_proba(X_test)]
    accuracy,sensitivity,specificity,strength,mcc,f1,aucvalue = metrics(Y_test,rounded,y_pred)
    fileout2 = open (file_path + name + "_test_result.txt", "w")
    fileout2.write(str(best) + "\n")
    fileout2.write("Accuracy_mean: "+str(accuracy) + "\n")
    fileout2.write("Sensitivity_mean: "+str(sensitivity) + "\n")
    fileout2.write("Specificity_mean: "+str(specificity) + "\n")
    fileout2.write("Strength_mean: "+str(strength) + "\n")
    fileout2.write("MCC_mean: "+str(mcc) + "\n")
    fileout2.write("Fscore_mean: "+str(f1) + "\n")
    fileout2.write("AUC_mean: "+str(aucvalue) + "\n\n\n")
    fileout2.close()
       
def main():
    global file_path, feature_set, name, scaler, dimension
    trainingfile, testfile, file_path = getparas()
    feature_set = "expression_full"
    name = "ann1"
    fileout1 = open(file_path + "train_output.txt", "w")
    para_file = open (file_path + "best_parameter.txt", "w")
    
    x_exp, y_train, = get_training_data(trainingfile)
    dimension = x_exp.shape[1]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_exp)
    
    global X_train_data,X_val_data,Y_train_data,Y_val_data
    x_train_data,X_val_data,y_train_data,Y_val_data = train_test_split(x_train, y_train, test_size = 0.1, random_state = 42)
    X_train_data, Y_train_data = resampling(x_train_data, y_train_data)
    
    # hyperparameter optimization
    space = {
        'hdim': hp.quniform('hdim',64, 256, 64),
        'l2_reg': hp.loguniform('l2_reg', np.log(0.00001), np.log(0.1)),
        'drop_out': hp.uniform('drop_out', 0.0, 1.0),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.2)),
        'batch_size': hp.quniform('batch_size', 6, 8, 1)
    }
    trials = Trials()
    global best
    best_param = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
    best = space_eval(space, best_param)
    para_file.write(str(best))
    para_file.close()
    # training, test by machine learning models
    model = deep_learning(x_train, y_train, 42, fileout1)
    test(model,testfile)

if __name__ == '__main__':
        main()
