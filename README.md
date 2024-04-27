# Python Version: Python 3.11.5

# Libraries:
 * numpy
 * matplotlib
 * pickle
 * scikit-learn
 * Graphviz: https://github.com/xflr6/graphviz
 * Alignment: https://github.com/eseraygun/python-alignment

# How to Run

Source files are found in ``src/`` and launcher files in ``launchers/``. The launcher files activate SLURM batch files because we used MSU's High-Powered Computing Cluster. Each launcher file runs 7 problems and 50 trials per problem. You will need to change the directory paths to match your system.
Running ``analysis.py`` in the ``src/`` folder. All outputs go to the ``output/`` folder which is organized by operator->problem->plot/file, and all high-level aggregations are output to simply the ``output/`` folder.

# Selection

Running the various selection operators requires switching to the selection branch.

# Mutation

Mutation code is contained in the mutation.zip file
