#######################################################################################################
# Download the Kallisto reference files for running Kallisto with a gene set that is compatible
# with CellO
########################################################################################################

curl http://deweylab.biostat.wisc.edu/cell_type_classification/kallisto_reference.tar.gz > kallisto_reference.tar.gz
tar -zxf kallisto_reference.tar.gz
rm kallisto_reference.tar.gz
