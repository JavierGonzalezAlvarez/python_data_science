# data science - machine learning

virtualenv entorno_virtual -p python3

## create an virtual environment from $
conda create --name base python=3
conda create --name entorno_virtual python=3

## activate v.e.
$ conda activate base
$ conda activate entorno_virtual

## list all v.e.
conda info --envs

## install tensorflow in the v.e.
pip install --upgrade pip
pip install --upgrade tensorflow

## run anaconda
$ anaconda-navigator

# desactivate anaconda
(base) $ conda deactivate

- drop() returns the data frame without the specified column
- loc (index)
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html

-The method np.array converts the selected columns into an array.

