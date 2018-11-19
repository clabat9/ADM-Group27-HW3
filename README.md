# ADM-Group27-HW3
# Find the best place to live in Texas!

The proposed scripts allow you to search for houses in Texas listed on AirBnb Database.

NB : we suggest to download the notebook file because html doesn't work well with the rendering of some latex formulas and pandas DataFrames. 

## Get data to analyze 

The data in usage are available on [Kaggle AirBnb Data](https://www.kaggle.com/PromptCloudHQ/airbnb-property-data-from-texas). 
They're a .csv file well organized with so many interesting features. 

## Script and Other files descriptions

1. __`Hw3_lib.py`__: 
	This script contains all the useful functions to get the proposed analysis deeply commented to have a clear view of the logic behind.

2. __`Example_map.html`__: 
	This html files contains a Texas map that higlights the houses searched by the user in an example query.
3. __`Useful_data.rar`__: 
	This folder contains some fundamental files in order to let the user launching the function without recreating the file.
  This files are :
  - The inverted index of the corpus ({term : [docs where the term appears]})
  - The inverted index with TFIDF of the corpus ({term : [(doc where the term appears,TFID(term,doc)), ...]})
  - The vocabulary of the corpus : ({term : term_id})
  
## `IPython Notebook: Homework3_final.ipynb (contained in .rar file)`
The goal of the `Notebook` is to provide a storytelling friendly format that shows how to use the implemented code and to carry out the reesults. It provides explanations and examples.
We tried to organize it in a way the reader can follow our logic and conlcusions.
