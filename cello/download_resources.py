import subprocess
from os.path import join

def download(dest):
    cmds = [
        'curl http://deweylab.biostat.wisc.edu/cell_type_classification/resources_v1.1.0.tar.gz > {}'.format(
            join(dest, 'resources_v1.1.0.tar.gz')
        ),
        'tar -C {} -zxf resources_v1.1.0.tar.gz'.format(dest),
        'rm {}'.format(join(dest, 'resources_v1.1.0.tar.gz'))
    ]
    for cmd in cmds:
        print('Running command: {}'.format(cmd))
        subprocess.run(cmd, shell=True)

