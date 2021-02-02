"""
Download the kallisto reference files for performing gene expression
quantification in order to produce expression profiles that are 
compatible with CellO's pre-trained models.

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

import subprocess
from os.path import join

def download(dest):
    cmds = [
        'curl http://deweylab.biostat.wisc.edu/cell_type_classification/kallisto_reference.tar.gz > {}'.format(
            join(dest, 'kallisto_reference.tar.gz')
        ),
        'tar -C {} -zxf kallisto_reference.tar.gz'.format(dest),
        'rm {}'.format(join(dest, 'kallisto_reference.tar.gz'))
    ]
    for cmd in cmds:
        print('Running command: {}'.format(cmd))
        subprocess.run(cmd, shell=True)

