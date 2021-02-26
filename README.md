# PredSynRNA
Prediction of synaptically localized RNAs in human neurons using developmental brain gene expression data<br />
# Requirements
Python 3<br />
argparse<br />
pandas<br />
numpy<br />
sklearn<br />
keras<br />
tensorflow<br />
joblib<br />
# Prediction and prioritization of candidate human RNAs localized to synapses
The **prediction_data.csv** contains over 7,000 brain-expressed RNAs which were not included in the training set. To predict and prioritize candidate human RNAs localized to synapses, the SVM, RF, and ANN models under "models" folder will be loaded. The following command line can be used:<br /><br />
`python prediction.py -t data/training.csv -f data/prediction_data.csv -s SVM.model -a ANN.hdf5 -r RF.model -p <path>`<br /><br />
And an output file (.xlsx) that includes the genes predicted as positive (dendritically localized) by each model with computed probability scores will be generated. The prediction results are stored in separate sheets. For example, in the sheet of "svm_proba_positive":<br />
| name  | predict_proba |
| ------------- | ------------- |
| HBB  | 0.982107707700408  |
| CRABP1  | 0.973952101206992  |
| CYBA  | 0.969843064843169  |
