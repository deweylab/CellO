# CellO: *Cell O*ntology-based classification &nbsp; <img src="https://raw.githubusercontent.com/deweylab/CellO/master/cello.png" alt="alt text" width="70px" height="70px">

## About

CellO (Cell Ontology-based classification) is a Python package for performing cell type classification of RNA-seq data. CellO makes hierarchical predictions against the [Cell Ontology](http://www.obofoundry.org/ontology/cl.html). These classifiers were trained on nearly all of the primary cell, bulk RNA-seq data in the [Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra). 

## Setup 

CellO requires some resources to run out-of-the-box. These resources can be downloaded with the following command:

``bash download_resources.sh``

This command will download and upack a ``resources`` directory that will be stored in the ``cello`` Python package.  Next, we set the PYTHON path to point to all packages in this repository:

``export PYTHONPATH=$(pwd):$PYTHONPATH``

The Python package dependencies are described in ``requirements.txt``. These dependencies can be installed within a Python virtual environment in one fell swoop with the following commands:

``
python -m venv cello_env
source cello_env/bin/activate
pip install -r requirements.txt 
`` 

## Running CellO

CellO uses a supervised machine learning classifier to classify the cell types within a dataset. CellO takes as input a gene expression matrix, which can be in multiple formats: a TSV/CSV file, HDF5 file, or 10x formatted directory.  

Notably, the input expression data's genes must match the genes expected by the trained classifier.  If the genes match, then CellO will use a pre-trained classifier to classify the expression profiles (i.e. cells) in the input dataset. 

To provide an example, here is how you would run CellO on an example dataset stored in ``example_input/``. This dataset is a set of XXXXXX monocytes distributed by Chromium 10x.  To run CellO on this dataset, run this command:

``python cello_predict.py``

If the genes in the input file do not match the genes on which the model was trained, then CellO will output an error message:

To circumvent this issue, CellO can be told to train a classifier with only those genes included in the given input dataset.  


Note that the ``-t`` flag tells CellO to train a fresh classifier on the genes contained in the input file.  The parameter ``--X XXXXXXX`` tells CellO to write the trained model to the file ``XXXXXXX``. Training CellO usually takes under an hour.

To run CellO on a custom, pre-trained model, run CellO as follows:

``python cell_predict.py -m ``

Note that ``-m XXXXXX`` tells CellO to use the model stored in ``XXXXXXXX``.


## Quantifying reads with Kallisto to match CellO's pre-trained models

We provide a script for quantifying raw reads with [Kallisto](https://pachterlab.github.io/kallisto/). Note that to run this script, Kallisto must be installed and available in your ``PATH`` environment variable.  This script will output an expression profile that includes all of the genes that CellO is expecting and thus, expression profiles created with this script are automatically compatible with CellO.

This script requires a preprocessed kallisto reference.  To download the pre-built Kallisto reference that is compatible with CellO, run the command:

``bash download_kallisto_reference.sh``

This command will download a directory called ``kallisto_refernce`` in the current directory.
