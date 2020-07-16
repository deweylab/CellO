# CellO: *Cell O*ntology-based classification &nbsp; <img src="https://raw.githubusercontent.com/deweylab/CellO/master/cello.png" alt="alt text" width="70px" height="70px">

## About

CellO (Cell Ontology-based classification) is a Python package for performing cell type classification of human RNA-seq data. CellO makes hierarchical predictions against the [Cell Ontology](http://www.obofoundry.org/ontology/cl.html). These classifiers were trained on nearly all of the human primary cell, bulk RNA-seq data in the [Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra). 

## Dependencies

The Python package dependencies are described in ``requirements.txt``. These dependencies can be installed within a Python virtual environment in one fell swoop with the following commands:

```
python3 -m venv cello_env 
source cello_env/bin/activate
pip install -r requirements.txt  
``` 

## Setup 

CellO requires some resources to run out-of-the-box. These resources can be downloaded with the following command:

``bash download_resources.sh``

Next, we set the PYTHONPATH to point to all packages in this repository:

``export PYTHONPATH=$(pwd):$PYTHONPATH``

## Running CellO

### Overview

CellO uses a supervised machine learning classifier to classify the cell types within a dataset. CellO takes as input a gene expression matrix. CellO accepts data in multiple formats:
* TSV: tab-separated value 
* CSV: comma-separated value
* HDF5: a database in HDF5 format that includes three datasets: a dataset storing the expression matrix, a dataset storing the list of gene-names (i.e. rows), and a gene-set storing the list of cell ID's (i.e. columns)
* 10x formatted directory: a directory in the 10x format including three files: ``matrix.mtx``, ``genes.tsv``, and ``barcodes.tsv``

Given an output-prefix provided to CellO (this can include the path to the output), CellO outputs three tables formatted as tab-separated-value files: 
* ``<output_prefix>.probability.tsv``: a NxM classification probability table of N cells and M cell types where element (i,j) is a probability value that describes CellO's confidence that cell i is of cell type j  
* ``<output_prefix>.binary.tsv``: a NxM binary-decision matrix where element (i,j) is 1 if CellO predicts cell i to be of cell type j and is 0 otherwise.
* ``<output_prefix>.most_specific.tsv``: a table mapping each cell to the most-specific predicted cell 

### Running CellO with a pre-trained model

Notably, the input expression data's genes must match the genes expected by the trained classifier.  If the genes match, then CellO will use a pre-trained classifier to classify the expression profiles (i.e. cells) in the input dataset. 

To provide an example, here is how you would run CellO on a toy dataset stored in ``example_input/Zheng_PBMC_10x``. This dataset is a set of 1,000 cells subsampled from the [Zheng et al. (2017)](https://www.nature.com/articles/ncomms14049) dataset.  To run CellO on this dataset, run this command:

``python cello_predict.py -d 10x -u COUNTS -s 3_PRIME example_input/Zheng_PBMC_10x -o test``

Note that ``-o test`` specifies the all output files will have the prefix "test". The ``-d`` specifies the input format, ``-u`` specifies the units of the expression matrix, and ``-s`` specifies the assay-type.  For a full list of available formats, units, assay-types, run:

``python cello_predict.py -h``


### Running CellO with a gene set that is incompatible with a pre-trained model

If the genes in the input file do not match the genes on which the model was trained, CellO can be told to train a classifier with only those genes included in the given input dataset by using the ``-t`` flag.  The trained model will be saved to a file named ``<output_prefix>.model.dill`` where ``<output_prefix>`` is the output-prefix argument provided via the ``-o`` option.  Training CellO usually takes under an hour. 

For example, to train a model and run CellO on the file ``example_input/LX653_tumor.tsv``, run the command:

``python cello_predict.py -u COUNTS -s 3_PRIME -t -o test example_input/LX653_tumor.tsv``

Along with the classification results, this command will output a file ``test.model.dill``.

### Running CellO with a custom model

Training a model on a new gene set needs only to be done once (see previous section). For example, to run CellO on ``example_input/LX653_tumor.tsv`` using a specific model stored in a file, run:

``python cello_predict.py -u COUNTS -s 3_PRIME -m test.model.dill -o test example_input/LX653_tumor.tsv``

Note that ``-m test.model.dill`` tells CellO to use the model computed in the previous example.

## Quantifying reads with Kallisto to match CellO's pre-trained models

We provide a script for quantifying raw reads with [Kallisto](https://pachterlab.github.io/kallisto/). Note that to run this script, Kallisto must be installed and available in your ``PATH`` environment variable.  This script will output an expression profile that includes all of the genes that CellO is expecting and thus, expression profiles created with this script are automatically compatible with CellO.

This script requires a preprocessed kallisto reference.  To download the pre-built Kallisto reference that is compatible with CellO, run the command:

``bash download_kallisto_reference.sh``

This command will download a directory called ``kallisto_reference`` in the current directory. To run Kallisto on a set of FASTQ files, run the command

``python run_kallisto.py <comma_dilimited_fastq_files> <tmp_dir> -o <kallisto_output_file>``

where ``<comma_delimited_fastq_files>`` is a comma-delimited set of FASTQ files containing all of the reads for a single RNA-seq sample and ``<tmp_dir>`` is the location where Kallisto will store it's output files.  The file ``<kallisto_output_file>`` is a tab-separated-value table of the log(TPM+1) values that can be fed directly to CellO.  To run CellO on this output file, run:

``python cell_predict.py -u LOG1_TPM -s FULL_LENGTH <kallisto_output_file> -o <cell_output_prefix>``

Note that the above command assumes that the assay is a full-length assay (meaning reads can originate from the full-length of the transcript).  If this is a 3-prime assay (reads originate from only the 3'-end of the transcript), the ``-s FULL_LENGTH`` should be replaced with ``-s 3_PRIME`` in the above command.

## Trouble-shooting

If upon running the command, `pip install -r requirements.txt`, and you receive an error installing Cython, that looks like:

```
ERROR: Command errored out with exit status 1:
     command: /scratch/cdewey/test_cello/CellO-master/cello_env/bin/python3 -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-wo2dj5q7/quadprog/setup.py'"'"'; __file__='"'"'/tmp/pip-install-wo2dj5q7/quadprog/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' egg_info --egg-base pip-egg-info
         cwd: /tmp/pip-install-wo2dj5q7/quadprog/
    Complete output (5 lines):
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/tmp/pip-install-wo2dj5q7/quadprog/setup.py", line 17, in <module>
        from Cython.Build import cythonize
    ModuleNotFoundError: No module named 'Cython'
    ----------------------------------------
ERROR: Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.
```

then you may try upgrading to the latest version of pip and Cython by running:

```
python -m pip install --upgrade pip
pip install --upgrade cython
```
