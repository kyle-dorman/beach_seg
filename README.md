# Segment Sandy Dunes

Repo for Segmenting Planet Dove sandy dune areas.

## Initial Setup

Some one time repo initialization work. 

### Install Conda
You can install miniconda if on Mac or Linux. On Windows install Anaconda.

#### Miniconda
Follow directions [HERE](https://docs.anaconda.com/miniconda/install/)

For Mac, perfer to install miniconda using brew. 
```bash
brew doctor
brew update
brew upgrade
brew upgrade --cask --greedy
brew install --cask miniconda
```

#### Anaconda
Follow directions [HERE](https://docs.anaconda.com/anaconda/install/)

### Open Terminal
On Windows, open an Andaconda Terminal, on Linux/Mac open a regular terminal. 

### Install Git
Check if git is installed. It will return something e.g. `/usr/bin/git` if git is installed. 
```bash
# Linux/Mac
which git
# Windows
where git
```

If git is not installed, install it. 
```bash
# Windows
conda install git
# Mac
brew install git
```

### Clone repo
This command will create a new folder `beach_seg` in your terminal's current directory. If you want it installed somewhere specific, move to that folder first (`cd SOMEWHERE/ELSE`)
```bash
git clone git@github.com:kyledorman/beach_seg.git
```

After cloning the repo, enter the folder
```bash
cd beach_seg
```

### Create conda environment
```bash
conda env create -f environment.yml
```

### Activate Jupyter Widgets
I am not sure if this is required or not. Depends on some package versions.
```bash
conda activate beach_seg
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter nbextension enable --py widgetsnbextension
```

## Inspect Results
You can inspect the results using an included jupyter notebook. 

Launch jupyter notebook
```bash
jupyter notebook --notebook-dir=notebooks --port=8893
```
Run the notebook `beach_seg.ipynb` 

## Format code
```bash
conda activate beach_seg
./lint.sh
```

## Update dependencies
After changing the environment.yml file, run
```bash
conda activate beach_seg
conda env update --file environment.yml --prune
conda activate beach_seg
```