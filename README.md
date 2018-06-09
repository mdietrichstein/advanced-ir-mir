# advanced-ir-mir
Music similarity and feature extraction experiments

## Pitfalls

**Missing Dependencies**
`requirements.txt` does not contain the following required dependencies: matplotlib, numpy, pandas, jupyter, notebook, progressbar.

You have to install them via conda or pip.

**Wrong scikit-learn version**

 `requirements.txt` lists the wrong scikit-learn version.
The notebook uses `sklearn.model_selection` which does not exist in scikit-learn 0.17. See http://scikit-learn.org/0.17/modules/classes.html.

**Python 3 is not supported**

`rp_extract.py` does not work with Python 3. You'll have to use Python 2.7.
