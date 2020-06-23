# CellO: *Cell O*ntology-based classification &nbsp; <img src="https://raw.githubusercontent.com/deweylab/CellO/master/cello.png" alt="alt text" width="70px" height="70px">

## About

CellO (Cell Ontology-based classification) is a Python package for performing cell type classification of RNA-seq data. CellO makes hierarchical predictions against the [Cell Ontology](http://www.obofoundry.org/ontology/cl.html). These classifiers were trained on nearly all of the primary cell, bulk RNA-seq data in the [Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra). 

## Setup 

CellO requires some resources to run out-of-the-box. These resources can be downloaded with the following command:

``bash download_resources.sh``

This command will download and upack a ``resources`` directory that will be stored in the ``cello`` Python package.

Next, wet the PYTHON path to point to all packages in this repository (specifically, ``cell``, ``graph_lib`` and ``onto_lib_py3``):

``export PYTHONPATH=$(pwd):$PYTHONPATH``

The Python package dependencies are described in ``requirements.txt``. These dependencies can be installed within a Python virtual environment in one fell swoop with the following commands:

``
python -m venv cello_env
source cello_env/bin/activate
pip install -r requirements.txt 
`` 

## Running CellO

CellO uses a supervised machine learning classifier to classify the cell types within a dataset. Notably, the input expression data's 
genes must match the genes expected by the trained classifier.  We provide pre-trained classifiers; however, in the event that your
data's genes do not match the genes expected by these classifiers, you can train CellO to operate specifically on your dataset.

Given an expression matrix (stored either as a TSV/CSV file, HDF5 file, or 10x formatted directory), you can train CellO to operate
on this datasets genes using the ``cello_train_model.py`` command. In the ``example_input`` directory, we have a gene expression matrix
from []() to provide an example. Specifically, to train a model on this dataset's genes, we would run the following command:

``python cell_train_model.py example_input/``.


This package uses [Kallisto](https://pachterlab.github.io/kallisto/)


To download the pre-built Kallisto reference and pre-trained classifiers, run the command: 

``bash download_resources.sh`` 

Finally, make sure that  ``onto_lib``, ``graph_lib``, and ``machine_learning`` are accessible via your ``PYTHONPATH`` environment variable:

``cd CellO``

``export PYTHONPATH:($pwd):$PYTHONPATH``

## Build a feature vector from a set of raw RNA-seq reads

To generate a feature vector from the raw reads stored in a set of FASTQ files, run the command: 

``python cellpredict/generate_feat_vec.py <comma-separated paths to FASTQ files> <path to directory in which temporary outputs are stored> -o <path to output file>``

## Run the classifier 

To run the classifier on a feature vector: 

``python cellpredict/predict.py <output file from generate_feat_vec.py>``

This command will create two files: ``predictions.tsv`` stores the binarized predictions and ``prediction_scores.tsv`` stores the raw probability scores for each cell type.


=======
# cell-type-classification-paper
