# MLCloudSearch
#### Moshe Weinreb
MLCloudSearch presents tooling built to facilitate the use of Scikit-Learn and Tensorflow Keras Models. The name MLCloudSearch was inspired by the Machine Learning technique of searching for the best parameters to use in fitting a model, commonly referred to as Hyper-Parameter Search. Training can be done locally or on the Cloud. Currently, only integration with Google Cloud Platform (GCP) is the only supported cloud platform.


# Overview
The module is part of my Senior Capstone Project, and is
meant to be used in conjunction with the NSRR Sleep Data set found
at https://sleepdata.org/datasets/shhs. With some minor modification, 
the module could be used as a command line tool for training any dataset
set

The module runs a GridSearchCV on a variety of Scikit-Learn models.
(This was built for a classification problem) 
The current list includes:
- Logistic Regression
- SVM (Linear, Kernel (rbf and poly))
- Decision Tree
- Random Forest
- Gradient Boosted Classifier

The setup also includes capabilities to run a Neural Network 
wrapped as a Sckit-Learn Estimator, thus enabling it to fit into a pipeline.
However as of this first release, the estimator must incorporate PCA 
(specifically KernelPCA). Other Scikit-Learn estimators do not have this constraint.
By including specification for several hyper-parameters, the user can search
in an arbitrarily sized DNN, with options on the:
- Number of Principle Components
- Number of inputs
- Number of layers
- Width of each layer
- Optimizer
- Activation functions
- Number of Epochs
- Batch Size

All of these parameters can be configured in a .json file.

# Interface
There are two main files in this repo, Train.py and Evaluate.py.
(Evaluate has only been tested on a Neural Network)
The best explanation comes from a quick demonstration
<pre><code>
> python Train.py --help 
usage: Train.py [-h] [--method {local,cloud}] [--bucket BUCKET]
                [--source-train SOURCE_TRAIN] [--source-test SOURCE_TEST]
                [--model-type {regression,svm,tree,forest,boosted,neural}]
                [--model-params MODEL_PARAMS]
                [--impute {simple,iterative,knn}]
                [--scaling {standard,minmax}] [--pca-method {linear,rbf,poly}]
                [--pca-params PCA_PARAMS] [--data-train DATA_TRAIN]
                [--data-test DATA_TEST] --save-name SAVE_NAME

Model Generator

optional arguments:
  -h, --help            show this help message and exit
  --method {local,cloud}
                        Choose training location [cloud, local]
  --bucket BUCKET       For training on cloud (GCP), this should be the bucket
                        name
  --source-train SOURCE_TRAIN
                        GCP - Name of train data blob [expects a csv]
  --source-test SOURCE_TEST
                        GCP - Name of test data blob [expects a csv]
  --model-type {regression,svm,tree,forest,boosted,neural}
                        Choose from [regression, svm, tree, forest, boosted,
                        neural]
  --model-params MODEL_PARAMS
                        Absolute path to model params [eventually will also be
                        a blob]
  --impute {simple,iterative,knn}
                        Choose method [simple, iterative, knn]
  --scaling {standard,minmax}
                        Choose from [standard minmax]
  --pca-method {linear,rbf,poly}
                        Choose from [linear, rbf] (Code needs to be rewritten
                        for poly)
  --pca-params PCA_PARAMS
                        Absolute path to pca params [eventually moved to same
                        blob as model hyper params]
  --data-train DATA_TRAIN
                        If training locally: absolute path to train data
  --data-test DATA_TEST
                        If training locally: absolute path to data
  --save-name SAVE_NAME
                        If training on cloud (GCP) - Name to save model and
                        output logs. No guarantee on behavior if training
                        locally

</code></pre>
As one can see there are a number of things to consider. First the user has 
the option to choose the method of training. This can either be cloud or local. If training 
is done on the cloud, the user must supply a bucket name which they have access to as well as the
names of the blobs in that bucket to use for the train and test set. (The train setis used for training/evaluation,
while the test set is only used during evaluation and strictly speaking is not needed for Train.py.)

The user has the option of choosing a model type. All are functioning (v0.0.0) except for the decision tree 
(noted by tree).

Model params refers to the dictionary of hyper-parameters, and is not appropriately named. Regular parameters
are hard-coded (v0.0.0) unless included as a single hyper-parameter. 

Other notes (v0.0.0):
- PCA is optional for scikit-learn models. Even then only the option of rbf (KernelPCA) is operational
- As mentioned above PCA is not optional (i.e. its mandatory) for neural (NeuralNetwork)

## Naming conventions
The save-name is a required argument and will be the basis of naming the produced log file 
(Stored/Uploaded in/from ./Logs) as well as the saved model (./SavedModels/Experimental/{networkname})

# Outputs
There are two areas that outputs are stored. The training log (which just includes 
the evaluation of the best trained model) is stored under ./Logs and is named as a combination
of the provided save-name and model type. The full grid is stored as a .pkl file. In the event that
one is training a Neural Network then two model files are output, the .pkl file contains the grid, but in
order to run evaluation without needing to refit the Tensorflow.Keras model during evaluation, the model is 
stored in a .h5 file (using tensorflow.keras.models.save() which preserves the weights) and manually added 
back into the grid before being returned.


# Evaluation
Evaluation requires running Evaluate.py and supplying similarly named arguments. It runs the 
models and prints out a confusion matrix


# Setup
Setup on both the Cloud and Local training look very similar, with some
of the difference coming on the need to upload and download training data
and output as opposed to providing paths to the data.


# Configuration
The pca and model hyper-parameters are stored under the ./HyperParams folder (in pca.json and model.json),
but this is just my convention. There is no real reason to require two different files, but
is the way development happened. The items in the json file need to be named according to the 
model-naming convention. So the parameters for a regression model need to be under 'regression' and 
each parameter needs to have the appropriate name followed by two underscores and then the model 
parameter. For example:
<pre><code>
{
  "regression": {
    "lr__penalty": ["l1", "l2"],
    "lr__C": [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01]
  },
   "svm": {
    "svm__C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    "svm__coef0": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 2.50, 5.0, 7.50, 10.0, 15.0, 20.0]
  }
}
</code></pre>
If the model chosen is an svm, a minimum total of 144 model (432 because of a default 3 cross fold validation)
will be trained. More will be trained, if multiple values for PCA are chosen.

