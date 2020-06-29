#######################################################################################################
# Download the pre-trained models and other data, such as gene-symbol mappings, for running CellO
########################################################################################################

curl http://deweylab.biostat.wisc.edu/cell_type_classification/resources.tar.gz > resources.tar.gz
tar -zxf resources.tar.gz
rm resources.tar.gz
