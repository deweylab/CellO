"""
Download the kallisto reference files for performing gene expression
quantification in order to produce expression profiles that are 
compatible with CellO's pre-trained models.

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

import subprocess
from os.path import join
from shutil import which

def download(dest):
    if which('curl') is None:
        sys.exit(
            """
            Error. Could not find command, 'curl'. Please make sure that 
            curl is installed and available via the 'PATH' variable. For 
            details, see https://curl.se.
            """
        )
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

