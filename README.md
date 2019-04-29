# CellO: *Cell O*ntology-based classification &nbsp; <img src="https://raw.githubusercontent.com/deweylab/CellO/master/cello.png" alt="alt text" width="70px" height="70px">

## About

CellO (Cell Ontology-based classification) is a Python package for performing cell type classification of RNA-seq data. CellO makes hierarchical predictions against the [Cell Ontology](http://www.obofoundry.org/ontology/cl.html). These classifiers were trained on nearly all of the primary cell, bulk RNA-seq data in the [Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra). 

## Setup 

The Python package dependencies are described in ``requirements.txt``. This package uses [kallisto](https://pachterlab.github.io/kallisto/) to create feature vectors from a raw RNA-seq reads and must also be installed.  

To download the pre-built kallisto reference and pre-trained classifiers, run the command: 

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


