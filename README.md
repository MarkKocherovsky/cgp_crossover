# Tutorial for Running and Configuring the Code for the Destructive Effect of Crossover Operators on Cartesian Genetic Programming

**Team Members:**  
Mark Kocherovsky  
Marzieh Kianinejad  
Elijah Smith

**Repo Link:**  
[https://github.com/MarkKocherovsky/cgp_crossover](https://github.com/MarkKocherovsky/cgp_crossover)

## Environment
- **Python Version:** 3.11.5
- **Recommended:** Using Anaconda environments
- **Required Libraries:**
  - numpy
  - matplotlib
  - pickle
  - scikit-learn
  - Graphviz: [https://github.com/xflr6/graphviz](https://github.com/xflr6/graphviz)
  - Alignment: [https://github.com/eseraygun/python-alignment](https://github.com/eseraygun/python-alignment)

## Crossover

While on the `main` branch, the `src/` and `launchers/` folders will be of most interest. `src/` contains all actual program files. Distinct crossover operators are stored in `cgp_xover.py` and can be plugged into any program as necessary, specifically in the relevant field in each Python code. The programs running each individual method are kept separate for the time being. Running jobs is triggered by shell scripts in the `launchers/` folder, which themselves trigger sbatch scripts that submit the jobs according to the parameters given in the `launchers/` file.

The source codes correspond to the algorithms as follows:
- `cgp.py`: baseline cgp(1+4)
- `cgp_40.py`: cgp(40+40) no crossover
- `cgp_1x.py`: cgp(40+40) one point crossover
- `cgp_2x.py`: cgp(40+40) two point crossover
- `cgp_sgx.py`: cgp(40+40) subgraph crossover
- `lgp_1x.py`: lgp(40+40) one point crossover
- `lgp_2x.py`: lgp(40+40) two point crossover
- `lgp.py`: lgp(40+40) uniform crossover

The shell script and batch files should match these terminologies. Each script has customizable parameters before you submit jobs; these are labeled in the script itself. Other Python scripts are mainly modules for running CGP.

All outputs are placed in the `output/` folder, sorted by crossover algorithm, problem, and output type (which graph/log file, etc.). To aggregate the results, you only need to run the `analysis.py` file without any parameters in the `src/` folder. All final outputs will be in the top level of the `output/` folder.

## Selection

All files necessary to perform the tests described in the selection portion of the report may be found in the selection branch of the GitHub repo.

To perform tests for a specific selection method, first modify `src/cgp_1x.py` and `src/cgp_2x.py` so that the variable `select` is assigned to the corresponding selection method from `src/selection_methods.py`. Then, using HPCC, run `launchers/cgp.sh`, `launchers/cgp_1x.sh`, and `launchers/cgp_2x.sh`. The script `launchers/cgp.sh` will only need to be run once, but it will be necessary to re-modify `src/cgp_1x.py` and `src/cgp_2x.py` and rerun `launchers/cgp_1x.sh` and `launchers/cgp_2x.sh` once for every desired selection method. Make sure to move output files into named directories after each test so that they are not overwritten by new tests.

Once test results for CGP, CGP1X, and CGP2X have been collected and stored in the `selection_output` directory, visualizations may be generated using `src/visualization.py`. To generate selection visualizations, input your chosen test ("Koza 1" or "Koza 2") and CGP implementation ("cgp_1x" or "cgp_2x") into the fields named `test_name` and `setting_name`, respectively, at the top of `visualization.py` and run the file. Make sure the directory structure of `selection_output` matches the structure defined in `src/visualization.py`.

## Mutation

**Environment:**  
Python Version: 3.11.5

Each algorithm was run 50 times for each function of the seven functions (350 trials per algorithm). All the functions are in `Function.py`.

**Libraries:**
- Numpy
- Pickle
- MatplotLib

Each trial was run on MSU’s High-Powered Computing Cluster. We set these parameters in our algorithms:
- `Pop_size`: the population size (default 80).
- `Generations`: number of generations (default 10000)
- `Instructions`: number of instructions for each individual (default 64)
- `Inputs`: number of inputs (operands) for each individual (default 2)
- `Output`: number of outputs for each individual (default 1)
- `Mutation_prob`: mutation probability (default 0.025)
- `Crossover_prob`: crossover probability (default 0.5)
- `Input_nodes`: constant numbers [0-9] (`np.arange(0, 101)`)
- `Nodes`: the number of input_nodes and each input(x) (default 11)

**In the .sh file:**
- `F`: the number of functions (default 7)
- `T`: the number of trials (default 50)

The `.sh` file containing the loop for seven functions and then 50 trials triggers the `.sb` file to start each trial. All the outputs will be stored in the `output/` folder: `output/algorithm’s_name/function’s_name/log/output_t`.

## Visualization

To visualize the results for each algorithm, we have two files: the code (`.py` file) and the launcher (`.sb` file). The `.sb` file triggers the `.py` code for visualization and doesn’t need any parameters. Finally, all the figures will be stored in the subfolder `figures` in the `output/` folder: `output/figures`.

## Appendix

**Full List of Libraries Installed in Our Anaconda Environment:**

_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
alignment                 1.0.10             pyh5e36f6f_0    bioconda
atk-1.0                   2.36.0               ha1a6a79_0  
blas                      1.0                         mkl  
boost-cpp                 1.82.0               hdb19cb5_2  
bottleneck                1.3.5           py311hbed6279_0  
brotli                    1.0.9                h5eee18b_7  
brotli-bin                1.0.9                h5eee18b_7  
bzip2                     1.0.8                h7b6447c_0  
c-ares                    1.19.1               h5eee18b_0  
ca-certificates           2023.12.12           h06a4308_0  
cairo                     1.16.0               hb05425b_5  
contourpy                 1.2.0           py311hdb19cb5_0  
cycler                    0.11.0             pyhd3eb1b0_0  
cyrus-sasl                2.1.28               h52b45da_1  
dbus                      1.13.18              hb2f20db_0  
expat                     2.5.0                h6a678d5_0  
font-ttf-dejavu-sans-mono 2.37                 hd3eb1b0_0  
font-ttf-inconsolata      2.001                hcb22688_0  
font-ttf-source-code-pro  2.030                hd3eb1b0_0  
font-ttf-ubuntu           0.83                 h8b1ccd4_0  
fontconfig                2.14.1               h4c34cd2_2  
fonts-anaconda            1                    h8fa9717_0  
fonts-conda-ecosystem     1                    hd3eb1b0_0  
fonttools                 4.25.0             pyhd3eb1b0_0  
freetype                  2.12.1               h4a9f257_0  
fribidi                   1.0.10               h7b6447c_0  
gdk-pixbuf                2.42.10              h5eee18b_0  
giflib                    5.2.1                h5eee18b_3  
glib                      2.69.1               he621ea3_2  
gobject-introspection     1.72.0          py311hbb6d50b_2  
graphite2                 1.3.14               h295c915_1  
graphviz                  2.50.0               h1b29801_1  
gst-plugins-base          1.14.1               h6a678d5_1  
gstreamer                 1.14.1               h5eee18b_1  
gtk2                      2.24.33              h73c1081_2  
gts                       0.7.6                hb67d8dd_3  
harfbuzz                  4.3.0                hf52aaf7_2  
icu                       73.1                 h6a678d5_0  
intel-openmp              2023.1.0         hdb19cb5_46306  
jpeg                      9e                   h5eee18b_1  
kiwisolver                1.4.4           py311h6a678d5_0  
krb5                      1.20.1               h143b758_1  
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libboost                  1.82.0               h109eef0_2  
libbrotlicommon           1.0.9                h5eee18b_7  
libbrotlidec              1.0.9                h5eee18b_7  
libbrotlienc              1.0.9                h5eee18b_7  
libclang                  14.0.6          default_hc6dbbc7_1  
libclang13                14.0.6          default_he11475f_1  
libcups                   2.4.2                h2d74bed_1  
libcurl                   8.4.0                h251f7ec_1  
libdeflate                1.17                 h5eee18b_1  
libedit                   3.1.20221030         h5eee18b_0  
libev                     4.33                 h7f8727e_1  
libffi                    3.4.4                h6a678d5_0  
libgcc-ng                 11.2.0               h1234567_1  
libgd                     2.3.3                h695aa2c_1  
libgfortran-ng            11.2.0               h00389a5_1  
libgfortran5              11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libiconv                  1.16                 h7f8727e_2  
libllvm14                 14.0.6               hdb19cb5_3  
libnghttp2                1.57.0               h2d74bed_0  
libpng                    1.6.39               h5eee18b_0  
libpq                     12.15                hdbd6064_1  
librsvg                   2.54.4               h36cc946_3  
libssh2                   1.10.0               hdbd6064_2  
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.5.1                h6a678d5_0  
libtool                   2.4.6             h6a678d5_1009  
libuuid                   1.41.5               h5eee18b_0  
libwebp                   1.3.2                h11a3e52_0  
libwebp-base              1.3.2                h5eee18b_0  
libxcb                    1.15                 h7f8727e_0  
libxkbcommon              1.0.1                h5eee18b_1  
libxml2                   2.10.4               hf1b16e4_1  
lz4-c                     1.9.4                h6a678d5_0  
matplotlib                3.8.0           py311h06a4308_0  
matplotlib-base           3.8.0           py311ha02d727_0  
mkl                       2023.1.0         h213fc3f_46344  
mkl-service               2.4.0           py311h5eee18b_1  
mkl_fft                   1.3.8           py311h5eee18b_0  
mkl_random                1.2.4           py311hdb19cb5_0  
munkres                   1.1.4                      py_0  
mysql                     5.7.24               h721c034_2  
ncurses                   6.4                  h6a678d5_0  
networkx                  3.1             py311h06a4308_0  
ninja                     1.10.2               h06a4308_5  
ninja-base                1.10.2               hd09550d_5  
nspr                      4.35                 h6a678d5_0  
nss                       3.89.1               h6a678d5_0  
numexpr                   2.8.7           py311h65dcdc2_0  
numpy                     1.26.0          py311h08b1b3b_0  
numpy-base                1.26.0          py311hf175353_0  
openjpeg                  2.4.0                h3ad879b_0  
openssl                   3.0.13               h7f8727e_0  
packaging                 23.1            py311h06a4308_0  
pandas                    2.1.1           py311ha02d727_0  
pango                     1.50.7               h05da053_0  
pathlib                   1.0.1              pyhd3eb1b0_1  
pcre                      8.45                 h295c915_0  
pillow                    10.0.1          py311ha6cbd5a_0  
pip                       23.3.1          py311h06a4308_0  
pixman                    0.40.0               h7f8727e_1  
ply                       3.11            py311h06a4308_0  
poppler                   22.12.0              h9614445_3  
poppler-data              0.4.11               h06a4308_1  
pyparsing                 3.0.9           py311h06a4308_0  
pyqt                      5.15.10         py311h6a678d5_0  
pyqt5-sip                 12.13.0         py311h5eee18b_0  
python                    3.11.5               h955ad1f_0  
python-dateutil           2.8.2              pyhd3eb1b0_0  
python-graphviz           0.20.1             pyh22cad53_0    conda-forge
python-tzdata             2023.3             pyhd3eb1b0_0  
pytz                      2023.3.post1    py311h06a4308_0  
qt-main                   5.15.2              h53bd1ea_10  
readline                  8.2                  h5eee18b_0  
scipy                     1.11.4          py311h08b1b3b_0  
setuptools                68.0.0          py311h06a4308_0  
sip                       6.7.12          py311h6a678d5_0  
six                       1.16.0             pyhd3eb1b0_1  
sqlite                    3.41.2               h5eee18b_0  
tbb                       2021.8.0             hdb19cb5_0  
tk                        8.6.12               h1ccaba5_0  
tornado                   6.3.3           py311h5eee18b_0  
tzdata                    2023c                h04d1e81_0  
wheel                     0.41.2          py311h06a4308_0  
xz                        5.4.2                h5eee18b_0  
zlib                      1.2.13               h5eee18b_0  
zstd                      1.5.5                hc292b87_0 

