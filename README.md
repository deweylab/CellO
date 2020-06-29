# CellO: *Cell O*ntology-based classification &nbsp; <img src="https://raw.githubusercontent.com/deweylab/CellO/master/cello.png" alt="alt text" width="70px" height="70px">

## About

CellO (Cell Ontology-based classification) is a Python package for performing cell type classification of RNA-seq data. CellO makes hierarchical predictions against the [Cell Ontology](http://www.obofoundry.org/ontology/cl.html). These classifiers were trained on nearly all of the primary cell, bulk RNA-seq data in the [Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra). 

## Dependencies

The Python package dependencies are described in ``requirements.txt``. These dependencies can be installed within a Python virtual environment in one fell swoop with the following commands:

``
python -m venv cello_env
source cello_env/bin/activate
pip install -r requirements.txt 
`` 

## Setup 

CellO requires some resources to run out-of-the-box. These resources can be downloaded with the following command:

``bash download_resources.sh``

This command will download and upack a ``resources`` directory that will be stored in the ``cello`` Python package.  Next, we set the PYTHON path to point to all packages in this repository:

``export PYTHONPATH=$(pwd):$PYTHONPATH``

## Running CellO

### Overview

CellO uses a supervised machine learning classifier to classify the cell types within a dataset. CellO takes as input a gene expression matrix. CellO accepts data in multiple formats:
* TSV
* CSV
* HDF5
* 10x formatted directory 

CellO outputs two tables: a NxM classification probability table of N cells and M cell types where element (i,j) is a probability value that describes CellO's confidence that cell i is of cell type j.  CellO also outputs a binary-decision matrix where element (i,j) is 1 if CellO predicts cell i to be of cell j and is 0 otherwise.

### Running CellO with a pre-trained model

Notably, the input expression data's genes must match the genes expected by the trained classifier.  If the genes match, then CellO will use a pre-trained classifier to classify the expression profiles (i.e. cells) in the input dataset. 

To provide an example, here is how you would run CellO on a toy dataset stored in ``example_input/Zheng_PBMC_10x``. This dataset is a set of 1,000 cells subsampled from the [Zheng et al. (2017)](https://www.nature.com/articles/ncomms14049) dataset.  To run CellO on this dataset, run this command:

``python cello_predict.py -d 10x -u COUNTS -s 3_PRIME example_input/Zheng_PBMC_10x -o output/test``

Note that ``-o test`` specifies the all output files will have the prefix "test". The ``-d`` specifies the input format, ``-u`` specifies the units of the expression matrix, and ``-s`` specifies the assay-type.  For a full list of available formats, units, assay-types, run:

``python cello_predity.py -h``

### Running CellO with a gene set that is incompatible with a pre-trained model

If the genes in the input file do not match the genes on which the model was trained, CellO can be told to train a classifier with only those genes included in the given input dataset by using the ``-t`` flag.  The trained model will be saved to a file named ``<output_prefix>.model.dill`` where ``<output_prefix>`` is the output-prefix argument provided via the ``-o`` option.  Training CellO usually takes under an hour. 

For example, to train a model and run CellO on the file ``example_input/LX653_tumor.tsv``, run the command:

``python cello_predict.py -u COUNTS -s 3_PRIME -t -o test example_input/LX653_tumor.tsv``

Along with the classification results, this command will output a file ``test.model.dill``.

### Running CellO with a custom model

Training a model on a new gene set needs only to be done once (see previous section). For example, to run CellO on ``example_input/LX653_tumor.tsv`` using a specific model stored in a file, run:

``python cell_predict.py -u COUNTS -s 3_PRIME -m test.model.dill -o test example_input/LX653_tumor.tsv``

Note that ``-m test.model.dill`` tells CellO to use the model computed in the previous example.

## Quantifying reads with Kallisto to match CellO's pre-trained models

We provide a script for quantifying raw reads with [Kallisto](https://pachterlab.github.io/kallisto/). Note that to run this script, Kallisto must be installed and available in your ``PATH`` environment variable.  This script will output an expression profile that includes all of the genes that CellO is expecting and thus, expression profiles created with this script are automatically compatible with CellO.

This script requires a preprocessed kallisto reference.  To download the pre-built Kallisto reference that is compatible with CellO, run the command:

``bash download_kallisto_reference.sh``

This command will download a directory called ``kallisto_refernce`` in the current directory.
