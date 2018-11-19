
# coding: utf-8

# In[11]:


# First of all, we import all the necessary libs

import nltk
import re
import unicodedata
import string
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import pandas as pd
import inflect
import pickle
import math
from scipy.spatial import distance
import heapq
from geopy import geocoders  
import numpy as np
from geopy import distance as geodis
from IPython.display import clear_output
from termcolor import colored
from IPython.display import Markdown
import matplotlib.pyplot as plt
import folium
# This strings open a connection to GeoPy Database in order to get cities and addresses coordinates knowing their name
gn = geocoders.GeoNames(username = "clabat9") 
gl = geocoders.Nominatim( user_agent = "clabat9")




# ---------- SECTION 1 : DOCUMENTS PREPROCESSING ----------
    
# F1 : This function removes stop words from list of tokenized words

def remove_stopwords(wrd):
    new_wrd = [] #List of updated words
    
    for word in wrd:
        if word not in stopwords.words('english'): # If the current word is not a stopword (ckeck using nltk)
            new_wrd.append(word)                   #appends it to the list
  
    return new_wrd




# F2 : This function removes punctuation from list of tokenized words

def remove_punctuation(wrd):
    new_wrds = []  #List of updated words
    
    for word in wrd:
        new_wrd = re.sub(r'[^\w\s]', '', word) # Replaces all punctuation word with "" using RegEx
        if new_wrd != '':
            new_wrds.append(new_wrd)           #And then appends all words different from "" to the list 
    
    return new_wrds



# F3 : This function stems words in a list of tokenized words

def stem_words(wrd):
    stemmer = LancasterStemmer() # Selects the stemmmer from nltk
    stems = [] # List of updated words
    
    for word in wrd:
        stem = stemmer.stem(word) # Stems the word
        stems.append(stem)        # and appends it to the list
        
    return stems




# F4 : This functions removes non ascii chars from a list of tokenized words

def remove_non_ascii(wrd):
    new_wrds = [] # List of updated words
    
    for word in wrd:
        new_wrd = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') # Filters non ascii chars
        new_wrds.append(new_wrd) # Appends the word to the list
    
    return new_wrds



# F5 : This function converts all characters to lowercase from a list of tokenized words

def to_lowercase(wrd):
    new_wrds = [] # List of updated words
    
    for word in wrd:
        new_wrd = word.lower()   # Converts the current word to lower case
        new_wrds.append(new_wrd) # And append it to the list
        
    return new_wrds




# F5 : This function replaces all integers occurences in list of tokenized words with textual representation

def replace_numbers(wrd):
    d = inflect.engine() # Libs inflect contains what we need
    new_wrds = [] # List of updated words
    
    for word in wrd:
        if word.isdigit(): # If the current word is a number
            new_wrd = d.number_to_words(word) # Converts it to its textual representation
            new_wrds.append(new_wrd) # And appends it to the list
        else:
            new_wrds.append(word) # If the current word is not a number appends it to the list
            
    return new_wrds



# The following function takes a record of a dataFrame containg our docs and preprocesses it's title and description 
# with all the previous functions

def preProcessing (x):
    x.fillna("*", inplace = True) # fills NA with "*"
    xt = x["title"] # Takes title and description
    xd = x["description"]
    
    if xt != "*":
        xt = nltk.word_tokenize(xt) # Tokenizes title using nltk
        
    if xd != "*":
        xd = nltk.word_tokenize(xd) # Tokenizes description using nltk
     
    # Uses previous functions
    xt = replace_numbers(xt)
    xd = replace_numbers(xd)
    xt = remove_stopwords(xt)
    xd = remove_stopwords(xd)
    xt = remove_punctuation(xt)
    xd = remove_punctuation(xd)
    xt = stem_words(xt)
    xd = stem_words(xd)
    xt = remove_non_ascii(xt)
    xd = remove_non_ascii(xd)
    xt = to_lowercase(xt)
    xd = to_lowercase(xd)
   
    x["title"] = xt
    x["description"] = xd
    
    return x # Returns the preprocessed doc




# This function takes the query and preprocecesses it with all the previous methods

def query_preProcessing (x):
    xt = nltk.word_tokenize(x) # Tokenizes query using nltk
    
    # Uses previous functions    
    xt = replace_numbers(xt)
    xt = remove_stopwords(xt)
    xt = remove_punctuation(xt)
    xt = stem_words(xt)
    xt = remove_non_ascii(xt)
    xt = to_lowercase(xt)
   
    return xt









# ---------- SECTION 2 : SPLITTING ----------


# F1 : This function takes a path of a DataFrame or the DataFrame it's self and exports each one of its rows as a .tsv file
# Important : if the function receives both the path and the DataFrame, it will use the DataFrame option.
# For our purposes it is not fundamental to guarantee that the file is in the specified path or that the df is consistent, 
# but it's clear that in a more general context will be useful to insert some simple code to catch errors.

def csv_to_tsv(path_of_the_doc_to_convert = None ,pdata = pd.DataFrame()):
   
    if not pdata.empty : # If it receives a DataFrame
        pdata.to_csv("processed_data.tsv",encoding = "utf-8", sep = "\t") # Saves it as a .tsv
        f = open("processed_data.tsv","r", encoding = "utf-8") # Loads it
        leng = 0 # Counter of the number of documents
        for line in f: # For each record (document)
            with open(r"D:\Claudio\Uni\M 1° anno Sapienza\AMDS\Homeworks\Hw3\ptsv\doc_"+str(leng)+".tsv", "w", encoding = "utf-8" ) as ftmp:
                ftmp.write(line) # Saves the record as .tsv
            leng += 1 # Update counter
            
        return leng # Returns the number of documents
        
    else:    # If it receives a path
        data = open(path_of_the_doc_to_convert,"r", encoding = "utf-8") # Opens the data in the specified path
        leng = 0 # And makes the same procedure above
        for line in data:
            with open(r"D:\Claudio\Uni\M 1° anno Sapienza\AMDS\Homeworks\Hw3\tsv\doc_"+str(leng)+".tsv", "w", encoding = "utf-8" ) as ftmp:
                ftmp.write(re.sub(r",","\t",line))
            leng += 1
            
        return leng
    
    
    
    
    
    
    
    
    
# ----------SECTION 3 : CREATION OF VOCABULARY, INVERTED INDECES AND SCORE FUNCTIONS----------

#  This function takes the path where (preprocessed) documents are saved and their total number 
# and returns the vocabulary of the indicated corpus

def create_vocabulary(number_of_docs, path):
    vocabulary = {} # The vocabulary is a dictionary of the form "Word : word_id"
    wid = 0 # word_id
    for idx in range(1,number_of_docs): # for every document..
        with open(path+"doc_"+str(idx)+".tsv", "r", encoding = "utf-8" ) as ftmp:
            first_line = ftmp.readline() # It opens the doc and reads the first line (in our case docs are made by only one line)
            
            desc = (first_line.split(sep = "\t"))[6] # Takes in account only title and description of the record
            title = (first_line.split(sep = "\t"))[9]
            
            # Following lines clean up some unuseful chars
            desc = desc.split("'")
            title = title.split("'")
            foo = ["]","[",", "]
            desc = list(filter(lambda x: not x in foo, desc))
            title = list(filter(lambda x: not x in foo, title))
            
            for word in title+desc: # For every word in title +  description
                if not word in list(vocabulary.keys()) : # if the word is not in the dic
                    vocabulary[word] = wid # adds it
                    wid += 1 # Update word_id
                    
    with open("vocabulary", "wb") as f :
            pickle.dump(vocabulary, f) # Saves the vocabulary as a pickle
            
    return vocabulary # Returns the vocabulary




# This function create the first inverted index we need in the form "word (key) : [list of docs that contain word] (value)".
# It takes the number of (preprocessed) docs  and the path where they are saved and returns the reverted index as a dictionary.

def create_inverted_index(number_of_docs, path):
    inverted_index = {} # Initializes the inverted index, in our case a dic
    
    for idx in range(1,number_of_docs+1): # for every document
        
        # Opens the doc, cleans it and extracts title and description as the previous function
        with open(path+"doc_"+str(idx)+".tsv", "r", encoding = "utf-8" ) as ftmp:
            first_line = ftmp.readline()
            desc = (first_line.split(sep = "\t"))[6]
            title = (first_line.split(sep = "\t"))[9]
            desc = desc.split("'")
            title = title.split("'")
            foo = ["]","[",", "]
            desc = list(filter(lambda x: not x in foo, desc))
            title = list(filter(lambda x: not x in foo, title))
            
            for word in title+desc: # for every word in title + description
                if word in list(inverted_index.keys()) : # if the word is in the inverted index
                    inverted_index[word] = inverted_index[word] + ["doc_"+str(idx)] # adds the current doc to the list of docs that contain the word
                else :
                    inverted_index[word] = ["doc_"+str(idx)] # else creates a record in the dic for the current word and doc

    with open("inverted_index", "wb") as f :
            pickle.dump(inverted_index, f) # Saves the inverted index as a pickle
            
    return inverted_index # returns the inverted index




# This function takes a term, a riverted index and the total number of docs in the corpus to compute the IDF of the term
        
def IDFi(term, reverted_index, number_of_docs):
    return math.log10(number_of_docs/len(reverted_index[term]))




# This function create the second inverted index we need in the form "word (key) : [(doc that contain the word, TFID of the term in the doc),....]"
# It takes the number of (preprocessed) docs, the path where they are saved, the vocabulary and a list containig all the idfs and returns the reverted index as a dictionary.

def create_inverted_index_with_TFIDF(number_of_docs, path, vocabulary, idfi):
    inverted_index2 = {} # Initializes the inverted index, in our case a dic
    
    for idx in range(1, number_of_docs+1): # for every document
        
        # Opens the doc, cleans it and extracts title and description as the previous function
        with open(path+"doc_"+str(idx)+".tsv", "r", encoding = "utf-8" ) as ftmp:
            first_line = ftmp.readline()
            desc = (first_line.split(sep = "\t"))[6]
            title = (first_line.split(sep = "\t"))[9]
            desc = desc.split("'")
            title = title.split("'")
            foo = ["]","[",", "]
            desc = list(filter(lambda x: not x in foo, desc))
            title = list(filter(lambda x: not x in foo, title))
            
            for word in title+desc: # for every word in title + description
                if word in list(inverted_index2.keys()) : # if the word is inthe inverted index
                    
                    # adds to the index line of the current word a tuple that contains the current doc and its TFID for the current word. It uses the vocabulary to get the index of the word
                    # in the IDF list.
                    inverted_index2[word] = inverted_index2[word] + [("doc_"+str(idx),((title+desc).count(word)/len(title+desc))*idfi[vocabulary[word]])] # Just applying the def
                else :
                    # Makes the same initializing the index line of the current word
                    inverted_index2[word] = [("doc_"+str(idx),((title+desc).count(word)/len(title+desc))*idfi[vocabulary[word]])]

    with open("inverted_index2", "wb") as f : # Saves the inverted index as a pickle
            pickle.dump(inverted_index2, f)
            
            
            
            
# This function takes the two inverted indices , the (processed) query, the document the query has to be compared to and the vocabulary
# and returns the cosine similarity between them

def score(pquery, document, inverted_index, inverted_index_with_TFIDF, vocabulary, idfi):
    #the first vector is made by the all the tfid of the words in thw query. To build it we use a simple list comprehension
    # that computes the tfid for all the words in set(query) in order to not process the same word more times
    v1 = [((pquery.count(word)/len(pquery))*idfi[vocabulary[word]])  if word in vocabulary.keys() else 0 for word in set(pquery)]
    v2 = []
    
    # We don't need to work on vectors in R^(number of distinct words in query+document) becouse, in that case, all elements that 
    # are not simultaneously non zero will give a 0 contribute in the computation of the similarity, 
    # so we just need to work in R^(number of different words in query).
    #(the optimal solution will be to work in R^(dim of intersection of different words in query+ different words in document)) . 
    # In the end, to build the vector associated to the doc:
    for word in set(pquery) : # for every distinc word in the query
        if word in vocabulary.keys(): # if the word is in the corpus vocabulary
            if document in inverted_index[word]: # if the document contains the word
                idx = inverted_index[word].index(document) # gets the index of the doc in the second inverted index using the first inverted index
                                                           # order will be the same
                v2.append(inverted_index_with_TFIDF[word][idx][1]) # appends the tfid of the current word for the selected doc
                                                                   # gettin it from the second inverted index
            else: # if the doc doesnt contain the word the associated component is 0 
                v2.append(0)
        else: # if the word is not in the vocabulary the associated component of the doc vectror is 0
            v2.append(0)
    if not all(v == 0 for v in v2): # if at least one word is in common
        return (1 - distance.cosine(v1, v2)) # returns the cosine similarity
    else: # if the query and the doc haven't nothing in common their similarity is 0
        return 0
    
    
    
    
# This function implements our score function explained in the notebook. It takes the max rate user prefers to spend, the number of 
# bed user prefers to have in it's holiday house, the city user prefers to stay in and one of the doc that match his query and returns it's score.
def score_function(max_cash, pref_bed, pref_city, coords, result):
    score = 0
    max_distance = 1298 # Normalization factor for the distances computed on the two farthest points of the Texas
    cash = float(result["average_rate_per_night"].split("$")[1])
    try :
        bed = int(result["bedrooms_count"])
    except :
        bed = 0.5
        
    if (cash < max_cash) & (cash > 0) :
        score += (5)*math.exp(-cash/max_cash)
    score += (4)*min(bed/pref_bed, pref_bed/bed)
    coord = (result.loc["latitude"], result.loc["longitude"])
    score += 3*(1 - geodis.distance(coords,coord).km/1298)
    return (100/12)*score









# ----------SECTION 4: SEARCH ENGINES----------

# This function implements a search engine that returns the docs containing ALL the words in the query.
# It takes the path where (preprocessed) docs are saved and the inverted index above
# and returns the list of the names of the docs containing all the world of the query, a df containing all features of this docs
# (useful later) and a df containing only the requested features.
# We tried to have some fun inserting code that allows the user to retry the search if it returns no results.


def first_search_engine(inverted_index, path):
    check = "y" # This var controls the logic of the multiple attempts
    while check == "y": # while it is "y"
        print(colored("Insert your query:", "blue", attrs =["bold","underline"]))#(Markdown('<span style="color: #800080">Insert your query: </span>'))  # Get users query (asking in a nice colored way :) )  
        query = input()
    
        pq = query_preProcessing(query) #Preprocesses the query
    
        l = set() # our results are in a set
        not_first_word = 0 # Var to know if it's the first word of the query
    
        for el in pq: # for every word in the query 
            if el in list(inverted_index.keys()): # if the word is in at least one document
                if not_first_word == 0: # If it's the first word of the query
                    l = set(inverted_index[el]) # Adds all the docs that contain the word to results
                    not_first_word += 1 # The next word is not the first
                else : # If it isn't the first word
                    l = l.intersection(set(inverted_index[el])) # Takes the docs that contain the word in a set and intersect it with
                                                                # the set of the results
            else: # if a word is not in the corpus there will be no results for this kind of search.
                l = [] # empty list
                break #exit the loop
            
        if len(l) == 0: # If there are no results
                print(colored("Your search did not bring results. Do you wanna try again? [y/n]", "red", attrs = ["underline"]))  # Get users query (asking in a nice colored way :) )  
                check = input() # asks the user if he wanna make another search
                while (check != "y")&(check !="n") : # force the user to give an acceptable answer
                    print(colored("You can choose [y/n] !","red", attrs = ["underline"]))  # Get users query (asking in a nice colored way :) )  
                    check = input() 
                # If the user wants to retry, the while statement loops again
                
                if check == "n": # If the user prefers to not retry
                    return [],l,pd.DataFrame(), pd.DataFrame() # returns empty data structures
        else: # If there are results
            res = []
            for doc in l : # for every doc of the results creates a df ( a column for each feature, as in the original one)
                res.append(pd.read_csv(path +doc+ ".tsv", sep = "\t", engine = "python", names = ["id","average_rate_per_night","bedrooms_count","city","date_of_listing","description","latitude","longitude","title","url"]))
            complete_results =  pd.concat(res).reset_index() # And then concatenates all of them
            
            # Takes only requested features and makes some operations to have a better visualization and clean up some junks
            results = complete_results.loc[:,["title","description","city","url"]]
            results.columns = map(str.upper, results.columns) 
            results["TITLE"] = results["TITLE"].apply(lambda x : re.sub(r"\\n"," ",x))
            results["DESCRIPTION"] = results["DESCRIPTION"].apply(lambda x : re.sub(r"\\n"," ",x))
            
            return pq,l,complete_results,results # returns results (and the query, useful later)
        
        
        
        
        
# This function implements a search engine that returns the first k  documents with the highest cosine similiraty 
# within respect the query. It takes the two inverted indices, the vocabulary, the number of (preprocessed) docs, the paths where 
# they are saved and k and returns a df containig the results.

def second_search_engine(inverted_index, inverted_index_with_TFIDF, vocabulary, path, idfi, k = 10):
    # Use the first search engine to get the results we need to compute the scores
    pq1,docs,_,_ = first_search_engine(inverted_index, path)
    scores = [] # Initializes the list containing the scores
    for doct in docs: # for every documents that matches the query
        
        # Appends to "scores" the cosine similarity between the doc and the query, and the name of the doc as a tuple
        scores.append((score(pq1,doct,inverted_index, inverted_index_with_TFIDF, vocabulary,idfi),doct))
    
    
    heapq._heapify_max(scores) # Creates a max heap based on the scores in "scores"
    res =  heapq.nlargest(k, scores, key=lambda x: x[0]) # Get the first k highest score elements of "scores"
    
    # The following codes simply build up the presentation of the results, similiar to the first one but with a column "SCORES"
    out = []
    for doc in res : 
            out.append(pd.read_csv(path+str(doc[1])+ ".tsv", sep = "\t", engine = "python", names = ["id","average_rate_per_night","bedrooms_count","city","date_of_listing","description","latitude","longitude","title","url"]))
    results = pd.concat(out).reset_index().loc[:,["title","description","city","url"]]
    only_scores = [res[i][0] for i in range(len(res))]
    results.insert(0, "SCORES", only_scores)
    results.columns = map(str.upper, results.columns)
    if not all(v == 0 for v in only_scores): # If the scores aren't all 0, presents the nonzero elements in the first k scores
            results["TITLE"] = results["TITLE"].apply(lambda x : re.sub(r"\\n"," ",x))
            results["DESCRIPTION"] = results["DESCRIPTION"].apply(lambda x : re.sub(r"\\n"," ",x))
            return results.loc[results["SCORES"] !=0].style.hide_index() #Return only the nnz results
    else: # If the scores are all zero the search has no matching
            print(colored("Your search did not bring results!", "red", attrs = ["underline"]))  
            
            
            
            
def our_search_engine(inverted_index, path, k = 10):
    _,docs,complete_results, results = first_search_engine(inverted_index, path) # Searches for all the docs that match the query with
                                                                                # first search engine
    if not complete_results.empty: # If there are resuls, asks for more information
        print(colored("How much would you like to spend at most?", "blue",attrs =["bold","underline"])) 
        max_cash = float(input())
        print(colored("How many bedrooms do you prefer? ", "blue",attrs =["bold","underline"]))  
        pref_bed = int(input())
        print(colored("In which city would you rather be? ", "blue",attrs =["bold","underline"]))  
        pref_city = input()
        
        # This code is useful to prevent possibly time out of the service used to get the city coordinates
        check = True
        while check :
            try:
                _,coords = gn.geocode(pref_city + ", TX", timeout = 10)
                check = False
            except:
                print(colored("There is no city in Texas with this name or the service timed out, reinsert please:", "red",attrs =["bold","underline"]))  
                pref_city = input()
        # Now we add to the results a column "SCORE" that contains the score obtained by each document using our function
        results.insert(0,"SCORE",complete_results.apply(lambda x : score_function(max_cash, pref_bed, pref_city,coords,x), axis = 1))
        
        #In the end we need to sort the results on the score column. As before, we need the highest k and we must use heap sort to get them
        # so we can use the same procedure of the second search engine working on the column "SCORE" of the results df
        scores = [(results["SCORE"].tolist()[i],i) for i in range(len(results["SCORE"].tolist()))] # Our "scores" in this case is a list
                                                                        # containig the scores and the indices of the row of the df associated to that score
        heapq._heapify_max(scores) # Creates a max heap based on the scores in "scores"
        res =  heapq.nlargest(k, scores, key=lambda x: x[0]) # Get the first k highest score elements of "scores"
        return results.iloc[[res[j][1] for j in range(len(res))]].style.hide_index()
    
    
    
    
    
    
    
    
    
#  SECTION 5 : DRAWING FUNCTION (just to visualize oyur score function without fill the ipynb with unuseful things)

def g_fun(s_max):
    x = np.arange(-2.0, s_max+20, 0.1)
    g = []
    for y in x:
        if (y <= s_max) & (y >= 0):
            g.append(math.exp(-y/s_max))
        else :
            g.append(0)
    plt.grid(True)
    plt.plot(x,g, linewidth = 4, color = "orange")
    plt.ylabel("g(s)")
    plt.xlabel("s")
    plt.show()
    
    
    
    
def h_fun(b_max):
    x = np.arange(0.000000000001, b_max+20, 1)
    g = []
    for y in x:
        if y>=0:
            g.append(min(y/b_max,b_max/y))
    plt.grid(True)
    plt.plot(x,g,"ro", linewidth = 4, color = "orange")
    plt.ylabel("h(b)")
    plt.xlabel("b")
    plt.show()
    
    
    
    
def u_fun():
    x = np.arange(0.0, 1, 0.1)
    g = []
    for y in x:
        if y>=0:
            g.append(1-y)
    plt.grid(True)
    plt.plot(x,g, linewidth = 4, color = "orange")
    plt.ylabel("u(d(c,c_tilde))")
    plt.xlabel("d(c,c_tilde)")
    plt.show()
    
    
    
    
    
    
    
    
    
# SECTION 6 : VISUALIZE THE HOUSES!

def fmap(data):
    print(colored("Insert the address you're interested in :", "blue", attrs = ["underline","bold"]))
    add = input() # Gets address
    check = True
    while check : # Controls if it's possible to get the address coordinates and, incase it isn't, asks the user to reinsert the address
        try:
            coords = gl.geocode(add)[1]
            check = False
        except:
            print(colored("There is no address with this name or the service timed out, reinsert please:", "red",attrs =["bold","underline"]))  
            add = input()
    print(colored("Enter the maximum distance (in km) within which you want to view the available houses:", "blue", attrs = ["underline", "bold"]))
    dist = float(input()) # Get the circle radius
    
    # If there are resuls, asks for more information
    print(colored("How much would you like to spend at most?", "blue",attrs =["bold","underline"])) 
    max_cash = float(input())
    print(colored("How many bedrooms do you want? ", "blue",attrs =["bold","underline"]))  
    pref_bed = int(input())
   
    
    # This strings clean up some weird data and filters on the user preferences
    data = data.dropna(axis=0)
    data = data.loc[data["bedrooms_count"] != "Studio"]
    cash_data = data.apply(lambda x : float(x["average_rate_per_night"].split("$")[1]), axis = 1)
    data = data.loc[( np.less_equal(cash_data.values,max_cash))  & (data.apply(lambda x : int(x["bedrooms_count"]), axis = 1) <= pref_bed)] 
    data.reset_index(inplace = True)
    
     
    # Gets coordinates and urls of the records     
    tups = data.apply(lambda x : (x["latitude"],x["longitude"]), axis = 1).tolist()
    links = data.apply(lambda x : x["url"], axis = 1).tolist()
    
    # This list contains houses that are in the selected radius
    spots_in_distance=[[item,links[idx]] for idx,item in enumerate(tups) if geodis.distance(coords,item).km <= dist ] 
    
    # Now it builds the map with the houses of interest, just a bit of folium
    mp = folium.Map(
    location = coords,
    zoom_start = 12)
    folium.Circle(
    location = coords,
    radius = dist*1000,
    color = "darkgreen",
    fill_color = "blue"
    ).add_to(mp)
    folium.Marker(coords,icon=folium.Icon(color='green')
    ).add_to(mp)
    iconm = folium.features.CustomIcon('http://www.pngall.com/wp-content/uploads/2016/05/Iron-Man.png', icon_size=(50,50))
    folium.Marker(coords,
                  icon = folium.Icon(color='darkblue', icon_color='white', icon='male', prefix='fa')
            ).add_to(mp)
          
    for i in range(len(spots_in_distance)):
        tooltip = 'Click for link!'
        link = folium.Popup('<a href='+str(spots_in_distance[i][1])+'target="_blank">'+ str(spots_in_distance[i][1])+' </a>')
        folium.Marker(spots_in_distance[i][0],icon=folium.Icon(color='red', icon = "home"),tooltip = tooltip,popup = link 
            ).add_to(mp)
    
    return mp
    
       
           
   
    

