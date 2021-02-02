"""
Download CellO's resources files. These files include CellO's pre-trained
models, gene ID-to-symbol mappings, and training sets for training CellO's
models on new gene sets.

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

import subprocess
from os.path import join

def download(dest):
    cmds = [
        'curl http://deweylab.biostat.wisc.edu/cell_type_classification/resources_v2.0.0.tar.gz > {}'.format(
            join(dest, 'resources_v2.0.0.tar.gz')
        ),
        'tar -C {} -zxf resources_v2.0.0.tar.gz'.format(dest),
        'rm {}'.format(join(dest, 'resources_v2.0.0.tar.gz'))
    ]
    for cmd in cmds:
        print('Running command: {}'.format(cmd))
        subprocess.run(cmd, shell=True)

