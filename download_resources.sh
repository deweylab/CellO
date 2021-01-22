#######################################################################################################
# Download the pre-trained models and other data, such as gene-symbol mappings, for running CellO
########################################################################################################

curl http://deweylab.biostat.wisc.edu/cell_type_classification/resources_v2.0.0.tar.gz > resources_v2.0.0.tar.gz
tar -zxf resources_v2.0.0.tar.gz
rm resources_v2.0.0.tar.gz
