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
The **mRNA_prediction_data.csv** contains over 7,000 brain-expressed RNAs and the **lncRNA_prediction_data.csv** contains over 3,000 brain-expressed lncRNAs which were not included in the training set. To predict and prioritize candidate human RNAs localized to synapses, the SVM, RF, and ANN models will be loaded. As for the prediction and prioritization of candidate human mRNAs localized to synapses, the following command line can be used:<br /><br />
`python prediction.py -t data/training.csv -f data/mRNA_prediction_data.csv -s SVM.model -r RF.model -a ANN.hdf5 -p <path>`<br /><br />
An output file (.xlsx) that includes the RNAs predicted as positive (dendritically localized) by each model with computed probability scores will be generated. The prediction results are stored in separate sheets. For example, in the sheet of "svm_proba_positive":<br />
| gene_id | name  | predict_proba |
| ------------- | ------------- |
| ENSG00000244734 | HBB  | 0.982107707700408  |
| ENSG00000166426 | CRABP1  | 0.973952101206992  |
| ENSG00000051523 | CYBA  | 0.969843064843169  |
