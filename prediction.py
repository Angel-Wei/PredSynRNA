import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import joblib
import random
random.seed(30)
from sklearn.preprocessing import MinMaxScaler

def getparas():
    parser = argparse.ArgumentParser(description="progrom usage")
    parser.add_argument("-t", "--train", type=str, help="Training data")
    parser.add_argument("-f", "--data", type=str, help="Prediction data")
    parser.add_argument("-s", "--svm", type=str, help="SVM model")
    parser.add_argument("-r", "--rf", type=str, help="RF model")
    parser.add_argument("-a", "--ann", type=str, help="ANN model")
    parser.add_argument("-p", "--path", type=str, help="File path to save the results")
    args = parser.parse_args()
    trainfile = args.train
    predfile = args.data
    svm = args.svm
    rf = args.rf
    ann = args.ann
    path = args.path
    return trainfile, predfile, svm, rf, ann, path


''' PFA, prob false alert for binary classifier'''
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

'''Calculate ROC AUC during model training, obtained from <https://github.com/nathanshartmann/NILC-at-CWI-2018>'''
def roc_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

def get_train_exp_data(file):
    full = pd.read_csv(file)
    full = shuffle(full, random_state = 42)
    x_exp = full.drop(columns = ['ensembl_gene_id', 'gene_symbol', 'utr5', 'utr3', 'label']).reset_index(drop = True)
    # log2(RPKM + 1) transformation
    x_exp = round(np.log2(x_exp + 1), 2)
    x_exp = np.array(x_exp)   
    return x_exp

def get_pred_exp_data(file):
    full = pd.read_csv(file)
    full = shuffle(full, random_state = 42)
    x_exp = full.drop(columns = ['ensembl_gene_id', 'gene_symbol', 'utr5', 'utr3']).reset_index(drop = True)
    # log2(RPKM + 1) transformation
    x_exp = round(np.log2(x_exp + 1), 2)
    x_exp = np.array(x_exp)   
    return x_exp

def prediction(svm, rf, ann, to_be_predicted, genes, path):
    svm_predicted = svm.predict(to_be_predicted)
    svm_count = 0
    for item in svm_predicted:
        if item == 1:
            svm_count = svm_count + 1
    print ("SVM predicted: " + str(svm_count))

    rf_predicted = rf.predict(to_be_predicted)
    rf_count = 0
    for item in rf_predicted:
        if item == 1:
            rf_count = rf_count + 1
    print ("RF predicted: " + str(rf_count))
    
    ann_predicted = ann.predict(to_be_predicted)
    ann_rounded = [round(x[0]) for x in ann_predicted]
    ann_count = 0
    for item in ann_rounded:
        if item == 1:
            ann_count = ann_count + 1
    print ("ANN predicted: " + str(ann_count))

    # svm: sort the probabilities in descending order and stored the instances predicted as positive
    svm_proba = svm.predict_proba(to_be_predicted)
    g1 = []
    p1 = []
    for j in range(0, len(svm_proba)):
        g1.append(genes[j])
        p1.append(svm_proba[j][1])
    df1 = pd.DataFrame(list(zip(g1, p1)), columns =['name', 'predict_proba']) 
    df1.sort_values(by=['predict_proba'],ascending=False, inplace=True)

    # rf: sort the probabilities in descending order and stored the instances predicted as positive
    rf_proba = rf.predict_proba(to_be_predicted)
    g2 = []
    p2 = []
    for j in range(0, len(rf_proba)):
        g2.append(genes[j])
        p2.append(rf_proba[j][1])
    df2 = pd.DataFrame(list(zip(g2, p2)), columns =['name', 'predict_proba'])
    df2.sort_values(by=['predict_proba'],ascending=False, inplace=True)


    # ann: sort the probabilities in descending order and stored the instances predicted as positive
    ann_proba = [x[0] for x in ann.predict(to_be_predicted)]
    g3 = []
    p3 = []
    for j in range(0, len(ann_proba)):
        g3.append(genes[j])
        p3.append(ann_proba[j])
    df3 = pd.DataFrame(list(zip(g3, p3)), columns =['name', 'predict_proba'])
    df3.sort_values(by=['predict_proba'],ascending=False, inplace=True)

    # write the prioritization results to a single file
    print ("Writting prioritization results...")
    writer = pd.ExcelWriter(path + "prioritization_result.xlsx", engine='xlsxwriter')
    df1.head(svm_count).to_excel(writer, sheet_name='svm_proba_positive', index = False)
    df2.head(rf_count).to_excel(writer, sheet_name='rf_proba_positive', index = False)
    df3.head(ann_count).to_excel(writer, sheet_name='ann_proba_positive', index = False)
    writer.save()
    
def main():
    trainingfile, predfile, svmfile, rffile, annfile, path = getparas()
    full_data = pd.read_csv(predfile)
    full_data = shuffle(full_data, random_state = 42)
    genes = []
    for index, row in full_data.iterrows():
        genes.append(row['gene_symbol'])
    scaler = MinMaxScaler()
    scaler.fit_transform(get_train_exp_data(trainingfile))
    to_be_predicted = scaler.transform(get_pred_exp_data(predfile))
    dependencies = {'roc_auc': roc_auc}
    
    # load machine learning models
    svm = joblib.load(svmfile)
    rf = joblib.load(rffile)
    ann = load_model(annfile, custom_objects = dependencies)
    
    # prediction, prioritization, and writing results
    prediction(svm, rf, ann, to_be_predicted, genes, path)  
if __name__ == '__main__':
        main()

