# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 17:40:20 2017
@author: Edward Preble
"""
###############################
#Task: Exploration and analysis of NTSB Aviation Accident dataset
#
#Major Code Sections:
#    Script Setup
#    ETL
#        XML import
#        JSON import
#    Exploration & Summary Stats
#    Basic Visualizations
#    Text Mining
#        NLTK Setup and Preprocessing
#        TF-IDF
#        Kmeans
#    Misc
#        Unused/Uninteresting Code
#        Ngram Frequency Analysis
###############################


###############################
#Script Setup
###############################
#REQUIRED FILE PATHS:
path_xml = "C:\Users\Edward\Desktop\Github\data-scientist-exercise02\data\AviationData.xml"
path_json = "C:\Users\Edward\Desktop\Github\data-scientist-exercise02\data"
kmeans_model_output_directory = 'C:\Users\Edward\Desktop\Github\data-scientist-exercise02'

#REQUIRED PACKAGES:
import pandas as pd
import xml.etree.ElementTree as ET
import json
from os import listdir
from os.path import isfile, join
from datetime import datetime
import nltk
from nltk.util import ngrams
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

###############################
#ETL
#   XML Import
###############################
tree = ET.parse(path_xml)
root = tree.getroot()
len(root[0])                    #How many records?  77257
len(root[0][0].attrib.keys())   #How many columns in each record?   31
aviationdata_column_names = root[0][0].attrib.keys()    #Grab list of column names
aviationdata_df = pd.DataFrame(columns=aviationdata_column_names)   #Create empty dataframe

#Load xml into dataframe, iterfind iterates through each xml tree branch finding keys/values
data = []
inner = {}
for el in tree.iterfind('./*'):
    for i in el.iterfind('*'):
        for k, v in i.items():            
            inner[k] = v
        data.append(inner)
        inner = {}
aviationdata_df = pd.DataFrame(data)
#Cleanup
del data, inner, i, k, v
###############################
#ETL
#   JSON Import
###############################
# Get list of JSONs in path_json, omitting AviationData.xml from the list
ListOfDataFiles = [f for f in listdir(path_json) if (isfile(join(path_json, f))) & (f!='AviationData.xml')]

#Load all files in directory, parse into a list of dicts, then load them into dataframe
i=0
for f in ListOfDataFiles:
    print i, f
    CurrentFileName = path_json + '\\' + f
    input_json = open( CurrentFileName, 'rb' )
    content = input_json.read()
    input_json.close()
    parsed_json = json.loads( content )         #results in a list of dictionaries
    del content                                 #This line is critical, memory bogs down incredibly if content isn't deleted each time
    json_df = pd.DataFrame(parsed_json['data']) #3 column dataframe results from each json

    #Concatenate new dataframe to the first one
    if i>0:
        merge_list = [all_jsons_df, json_df]
        all_jsons_df = pd.concat(merge_list)
    else:
        all_jsons_df = json_df.copy()
    i+=1

basic_stats = all_jsons_df.describe()   #All the EventId's are unique!
#Cleanup
del parsed_json, i, f, merge_list, json_df
###############################
#ETL
#   Make some conversions
#   Then join text data with structured data, then slice out small and large plane data
###############################
#Some basic stats/checks
basic_stats = aviationdata_df.describe()
    #NOTE: EventId not all unique, but AccidentNumber are...
ids = aviationdata_df["EventId"]
dups_df = aviationdata_df[ids.isin(ids[ids.duplicated()])]
    #UPDATE: This is ok, one EventID can involve 2 aircraft, which is two AccidentNumbers

#Convert various string columns into numeric
aviationdata_df.dtypes
aviationdata_df['TotalFatalInjuries'] = pd.to_numeric(aviationdata_df['TotalFatalInjuries'], errors='coerce')
aviationdata_df['TotalSeriousInjuries'] = pd.to_numeric(aviationdata_df['TotalSeriousInjuries'], errors='coerce')
aviationdata_df['TotalMinorInjuries'] = pd.to_numeric(aviationdata_df['TotalMinorInjuries'], errors='coerce')
aviationdata_df['TotalUninjured'] = pd.to_numeric(aviationdata_df['TotalUninjured'], errors='coerce')
aviationdata_df['NumberOfEngines'] = pd.to_numeric(aviationdata_df['NumberOfEngines'], errors='coerce')
aviationdata_df.dtypes

#Convert accident EventDate to datetime
aviationdata_df['datetimes']=""
aviationdata_df['datetimes'] = pd.to_datetime(aviationdata_df['EventDate'], format="%m/%d/%Y", errors='coerce')

#Calculate number of passengers on board (if known)
aviationdata_df['total_passengers']=""
aviationdata_df['total_passengers'] = aviationdata_df['TotalFatalInjuries'].fillna(0) + aviationdata_df['TotalSeriousInjuries'].fillna(0) + aviationdata_df['TotalMinorInjuries'].fillna(0) + aviationdata_df['TotalUninjured'].fillna(0)

#Outer Join the json data to the aviationdata
all_data_df = pd.merge(aviationdata_df, all_jsons_df, on='EventId', how='outer')

#OPTIONAL (save memory)
#If not mining the narratives, remove all rows with empty probable cause statements
all_data_df['probable_cause'].replace('', np.nan, inplace=True)
all_data_df.dropna(subset=['probable_cause'], inplace=True)

###############################
#Exploration & Summary Stats
###############################
#Find Accident Date Range
min_date = aviationdata_df['datetimes'].min()
max_date = aviationdata_df['datetimes'].max()

#Find number of incidents based on number of passengers on board
print aviationdata_df[(aviationdata_df["total_passengers"]>0) & (aviationdata_df["total_passengers"]<5)].count()["AccidentNumber"]
print aviationdata_df[(aviationdata_df["total_passengers"]>=5) & (aviationdata_df["total_passengers"]<20)].count()["AccidentNumber"]
print aviationdata_df[aviationdata_df["total_passengers"]>=20].count()["AccidentNumber"]
print aviationdata_df[aviationdata_df["total_passengers"]==0].count()["AccidentNumber"]
#70148 accidents with <5 passengers
#4022 accidents with 5-19 passengers
#2395 accidents with >19 passengers
#692 accidents with unknown number of passengers

#Find number of fatal incidents based on number of passengers on board
print aviationdata_df[aviationdata_df["TotalFatalInjuries"]>0].count()["AccidentNumber"]
print aviationdata_df[(aviationdata_df["TotalFatalInjuries"]>0) & (aviationdata_df["total_passengers"]>0) & (aviationdata_df["total_passengers"]<5)].count()["AccidentNumber"]
print aviationdata_df[(aviationdata_df["TotalFatalInjuries"]>0) & (aviationdata_df["total_passengers"]>=5) & (aviationdata_df["total_passengers"]<20)].count()["AccidentNumber"]
print aviationdata_df[(aviationdata_df["TotalFatalInjuries"]>0) & (aviationdata_df["total_passengers"]>=20)].count()["AccidentNumber"]
print aviationdata_df[(aviationdata_df["TotalFatalInjuries"]>0) & (aviationdata_df["total_passengers"]==0)].count()["AccidentNumber"]
#13988 fatal accidents with <5 passengers
#1217 fatal accidents with 5-19 passengers
#223 fatal accidents with >19 passengers
#0 fatal accidents with unknown number of passengers

###############################
#    Basic Visualizations
###############################
 
#Break down the accidents and fatal accidents by year
aviationdata_df['year']=pd.DatetimeIndex(aviationdata_df['datetimes']).year
aviationdata_df['year'] = aviationdata_df['year'].fillna(0.0).astype(int)

#Very little data before 1982, just use 1982-2015
accidents_over_time_column_names = range(1982,2016)
accidents_over_time_index = ['Accidents_Small','Accidents_Medium','Accidents_Large','Accidents_Unknown']
accidents_over_time_df = pd.DataFrame(columns=accidents_over_time_column_names, index=accidents_over_time_index)
fatal_accidents_over_time_index = ['Fatal_Accidents_Small','Fatal_Accidents_Medium','Fatal_Accidents_Large']
fatal_accidents_over_time_df = pd.DataFrame(columns=accidents_over_time_column_names, index=fatal_accidents_over_time_index)

#Find total number of large flights with fatalities for last 10 years
recent_large_flight_fatalities_df = fatal_accidents_over_time_df.copy()
recent_large_flight_fatalities_df = recent_large_flight_fatalities_df.loc[:,'2006':]
recent_large_flight_fatalities_df['Total'] = recent_large_flight_fatalities_df.sum(axis=1)
#62 large flights with fatalities

#Determine number of accidents and fatal accidents for small, medium, large, and unknown size planes, by year
for year in accidents_over_time_column_names:
#    print year, aviationdata_df[aviationdata_df["year"]==year].count()["AccidentNumber"]
    accidents_over_time_df.iloc[accidents_over_time_df.index.get_loc('Accidents_Small'), accidents_over_time_df.columns.get_loc(year)] = aviationdata_df[(aviationdata_df["year"]==year) & (aviationdata_df["total_passengers"]>0) & (aviationdata_df["total_passengers"]<5)].count()["AccidentNumber"]
    accidents_over_time_df.iloc[accidents_over_time_df.index.get_loc('Accidents_Medium'), accidents_over_time_df.columns.get_loc(year)] = aviationdata_df[(aviationdata_df["year"]==year) & (aviationdata_df["total_passengers"]>=5) & (aviationdata_df["total_passengers"]<20)].count()["AccidentNumber"]
    accidents_over_time_df.iloc[accidents_over_time_df.index.get_loc('Accidents_Large'), accidents_over_time_df.columns.get_loc(year)] = aviationdata_df[(aviationdata_df["year"]==year) & (aviationdata_df["total_passengers"]>=20)].count()["AccidentNumber"]
    accidents_over_time_df.iloc[accidents_over_time_df.index.get_loc('Accidents_Unknown'), accidents_over_time_df.columns.get_loc(year)] = aviationdata_df[(aviationdata_df["year"]==year) & (aviationdata_df["total_passengers"]==0)].count()["AccidentNumber"]
    fatal_accidents_over_time_df.iloc[fatal_accidents_over_time_df.index.get_loc('Fatal_Accidents_Small'), fatal_accidents_over_time_df.columns.get_loc(year)] = aviationdata_df[(aviationdata_df["year"]==year) & (aviationdata_df["TotalFatalInjuries"]>0) & (aviationdata_df["total_passengers"]>0) & (aviationdata_df["total_passengers"]<5)].count()["AccidentNumber"]
    fatal_accidents_over_time_df.iloc[fatal_accidents_over_time_df.index.get_loc('Fatal_Accidents_Medium'), fatal_accidents_over_time_df.columns.get_loc(year)] = aviationdata_df[(aviationdata_df["year"]==year) & (aviationdata_df["TotalFatalInjuries"]>0) & (aviationdata_df["total_passengers"]>=5) & (aviationdata_df["total_passengers"]<20)].count()["AccidentNumber"]
    fatal_accidents_over_time_df.iloc[fatal_accidents_over_time_df.index.get_loc('Fatal_Accidents_Large'), fatal_accidents_over_time_df.columns.get_loc(year)] = aviationdata_df[(aviationdata_df["year"]==year) & (aviationdata_df["TotalFatalInjuries"]>0) & (aviationdata_df["total_passengers"]>=20)].count()["AccidentNumber"]

#Transpose the two dataframes for easy plotting
accidents_over_time_transpose_df = accidents_over_time_df.transpose()
fatal_accidents_over_time_transpose_df = fatal_accidents_over_time_df.transpose()

#Plot accidents over time for difference size aircraft
#Plot.ly Syntax Reference: https://plot.ly/javascript-graphing-library/reference/
import plotly
import plotly.graph_objs

plotly.offline.plot({
    'data': [
        plotly.graph_objs.Scatter(
            x = fatal_accidents_over_time_transpose_df.index,
            y = fatal_accidents_over_time_transpose_df['Fatal_Accidents_Small'],
            mode = 'lines+markers',
            name = '1-4 Passenger Flights'
        ),
        plotly.graph_objs.Scatter(
            x = fatal_accidents_over_time_transpose_df.index,
            y = fatal_accidents_over_time_transpose_df['Fatal_Accidents_Medium'],
            mode = 'lines+markers',
            name = '5-19 Passenger Flights'
        ),
        plotly.graph_objs.Scatter(
            x = fatal_accidents_over_time_transpose_df.index,
            y = fatal_accidents_over_time_transpose_df['Fatal_Accidents_Large'],
            mode = 'lines+markers',
            name = '20+ Passenger Flights'
        )
    ],
    'layout':
        plotly.graph_objs.Layout(
            showlegend=True,
            legend=dict(
                x=0.6,
                y=0.98,
                traceorder='normal',
                font=dict(
                    family='Calibri',
                    size=24,
                    color='black'
                    )
                ),
            xaxis=dict(
                title='Year',
                titlefont=dict(
                    family='Calibri',
                    size=30,
                    color='black'
                    ),
                tickfont=dict(
                    family='Calibri',
                    size=24,
                    color='black'
                    )
                ),
            yaxis=dict(
                title='Accidents Per Year',
                titlefont=dict(
                    family='Calibri',
                    size=30,
                    color='black'
                    ),
                tickfont=dict(
                    family='Calibri',
                    size=24,
                    color='black'
                    )
                ),            
            height=600,
            width=600
        )
}, show_link=False,image='png')

###############################
#    Text Mining
#        NLTK Setup and Preprocessing
###############################
#OPTIONAL - Can drop all_json_df and aviationdata_df at this point to save memory
del all_jsons_df
del aviationdata_df

stop_words = nltk.corpus.stopwords.words( 'english' )
#Add more stop words that are not accident related (mostly investigation related)
stop_words.append('faa')
stop_words.append('federal')
stop_words.append('aviation')
stop_words.append('administration')
stop_words.append('investigators')
stop_words.append('investigation')
stop_words.append('ntsb')
stop_words.append('contributing')

# Use regular expressions to get rid of non-a-zA-Z chars
# leaves lots of spaces and makes two words out of "pilot's", but better than making one word out of "pilot.nextsentence"
def remove_nonletters(s):
    letters_only = re.sub("[^a-zA-Z]",      # The pattern to search for
                          " ",              # The pattern to replace it with
                          s )               # The text to search
    return letters_only

#Tokenize the text
def tokenize_field(s):
    term_vec = [ ]
    s=s.lower()
    term_vec.append( nltk.word_tokenize( s ) )
    return term_vec

#Remove stop words from pre-tokenized list, which outputs a list
def remove_stops(s):
    s = [[word for word in text if word not in stop_words] for text in s]
    return s

#Function for TfidfVectorizer
porter = nltk.stem.porter.PorterStemmer()
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [porter.stem(t) for t in filtered_tokens]
    return stems

#Function for TfidfVectorizer
def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#Slice Data of Interest from main dataframe
#SF (small fatal)
all_small_fatal_data_df = all_data_df[(
    all_data_df["total_passengers"]>0) & (
    all_data_df["total_passengers"]<5) & (
    all_data_df["TotalFatalInjuries"]>0)].copy()

#SNF (small non-fatal)
all_small_nonfatal_data_df = all_data_df[(
    all_data_df["total_passengers"]>0) & (
    all_data_df["total_passengers"]<5) & (
    all_data_df["TotalFatalInjuries"]==0)].copy()

#LF (large fatal)
all_large_fatal_data_df = all_data_df[(
    all_data_df["total_passengers"]>=20) & (
    all_data_df["TotalFatalInjuries"]>0)].copy()

#LNF (large non-fatal)
all_large_nonfatal_data_df = all_data_df[(
    all_data_df["total_passengers"]>=20) & (
    all_data_df["TotalFatalInjuries"]==0)].copy()

#Copy data into corpus for text pre-processing
corpus_SF  = all_small_fatal_data_df['probable_cause'].copy()
corpus_SNF = all_small_nonfatal_data_df['probable_cause'].copy()
corpus_LF  = all_large_fatal_data_df['probable_cause'].copy()
corpus_LNF = all_large_nonfatal_data_df['probable_cause'].copy()

#Remove punctuation and numbers etc, leave only letters behind
corpus_SF  = corpus_SF.apply(remove_nonletters)
corpus_SNF = corpus_SNF.apply(remove_nonletters)
corpus_LF  = corpus_LF.apply(remove_nonletters)
corpus_LNF = corpus_LNF.apply(remove_nonletters)

#Remove stop_words from each dataset
i=0
for cause in corpus_SF.iloc[0:len(corpus_SF)]:
    filtered_line = " ".join([word for word in cause.lower().split() if word not in stop_words])
    corpus_SF.iloc[i] = filtered_line
    i+=1
i=0
for cause in corpus_SNF.iloc[0:len(corpus_SNF)]:
    filtered_line = " ".join([word for word in cause.lower().split() if word not in stop_words])
    corpus_SNF.iloc[i] = filtered_line
    i+=1
i=0
for cause in corpus_LF.iloc[0:len(corpus_LF)]:
    filtered_line = " ".join([word for word in cause.lower().split() if word not in stop_words])
    corpus_LF.iloc[i] = filtered_line
    i+=1
i=0
for cause in corpus_LNF.iloc[0:len(corpus_LNF)]:
    filtered_line = " ".join([word for word in cause.lower().split() if word not in stop_words])
    corpus_LNF.iloc[i] = filtered_line
    i+=1

#NOTE!!!
#CHOOSE WHICH CORPUS TO USE, THEN REST OF CODE MUST BE RERUN MANUALLY FOR EACH CORPUS

corpus = corpus_SF.copy()

#Or
corpus = corpus_SNF.copy()

#Or
corpus = corpus_LF.copy()

#Or
corpus = corpus_LNF.copy()




#Create vocabulary reference list, which will be used later to look up clusters topics
totalvocab_stemmed = []
totalvocab_tokenized = []
for wordlist in corpus.iloc[0:len(corpus)]:
    allwords_stemmed = tokenize_and_stem(wordlist) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed)    #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(wordlist)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

###############################
#    Text Mining
#        TFIDF
###############################
#Lots of super helpful info here:http://brandonrose.org/clustering
#Saved as Document Clustering with Python.pdf

#Setup TFIDF Vectorizer
#Lots of optimization still possible with min/max/ngram/tokenizing variation
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=200000,
                                 min_df=2,
                                 use_idf=True, tokenizer=tokenize_only, ngram_range=(4,7))
#Create TFIDF Matrices
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus) #fit the vectorizer to text
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
features = tfidf_vectorizer.get_feature_names()
#OPTIONAL
#Cosine Similarity Matrix
#dist = 1 - cosine_similarity(tfidf_matrix)

###############################
#    Text Mining
#        KMeans
###############################
num_clusters = 5
km = KMeans(n_clusters=num_clusters)

#Create New KMeans Model
%time km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

#OPTIONAL TO RELOAD MODEL
#km = joblib.load('doc_cluster.pkl')
#clusters = km.labels_.tolist()

#NOTE!!!
#CHOOSE CORRECT ACCIDENTS LIST, DEPENDING ON DATABASE CHOSEN ABOVE
joblib.dump(km,  kmeans_model_output_directory + '\\kmeans_SF.pkl')
accidents = { 'Accident': all_small_fatal_data_df['AccidentNumber'].tolist()
            , 'EventID': all_small_fatal_data_df['EventId'].tolist()
            , 'cluster': clusters}

joblib.dump(km,  kmeans_model_output_directory + '\\kmeans_SNF.pkl')
accidents = { 'Accident': all_small_nonfatal_data_df['AccidentNumber'].tolist()
            , 'EventID': all_small_nonfatal_data_df['EventId'].tolist()
            , 'cluster': clusters}

joblib.dump(km,  kmeans_model_output_directory + '\\kmeans_LF.pkl')
accidents = { 'Accident': all_large_fatal_data_df['AccidentNumber'].tolist()
            , 'EventID': all_large_fatal_data_df['EventId'].tolist()
            , 'cluster': clusters}

joblib.dump(km,  kmeans_model_output_directory + '\\kmeans_LNF.pkl')
accidents = { 'Accident': all_large_nonfatal_data_df['AccidentNumber'].tolist()
            , 'EventID': all_large_nonfatal_data_df['EventId'].tolist()
            , 'cluster': clusters}

#Kmeans Model Summary
frame = pd.DataFrame(accidents, index = [clusters] , columns = ['Accident', 'EventID', 'cluster'])
frame['cluster'].value_counts()

#Print out cluster topics
from __future__ import print_function  #FYI, above print statements won't work after this import
print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("\nCluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print (terms[ind],sep=" ",end=",")
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['Accident'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace

#SF Results
#1    7555
#0     519
#3     357
#2     238
#4     132

#SNF Results
#2    18629
#1      687
#4      488
#0      333
#3      264

#LF Results
#0    58
#2     4
#4     2
#3     2
#1     2

#LNF Results
#0    700
#3     13
#4      2
#2      2
#1      2


###############################
#    Misc - unused/uninteresting
###############################

##This may also work for XML import, and be faster, but it is already super fast for this data, so not tested.
#for el in tree.iterfind('./*'):
#    for i in el.iterfind('*'):
#        data.append(dict(i.items()))


##Types of Aircraft
#    #60737 values are blank, so not very interesting
#aviationdata_df["AircraftCategory"].unique()
#aviationdata_df["AircraftCategory"].value_counts()

##Types of Flights (commercial, general, etc)
#    #60592 values are blank, so not very interesting
#aviationdata_df["FARDescription"].unique()
#aviationdata_df["FARDescription"].value_counts()

##Removing punctuation from unicode text via translate also needs an encode statement
#all_data_df['narrative'][5].encode('utf-8').translate(None, string.punctuation)

##Remove Punctuation
#def remove_punctuation(s):
#    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
#    return s

#practice_df['narrative_punctuation_removed'] = practice_df['narrative'].apply(remove_punctuation)
#practice_df['cause_punctuation_removed'] = practice_df['probable_cause'].apply(remove_punctuation)

#def is_number(s):
#    try:
#        float(s)
#        return True
#    except ValueError:
#        return False

#def remove_numbers(s):
#    term_vec = [ ]
#    for x in s:
#        for y in x:
##            print type(y)
#            if not is_number(y):
#                term_vec.append( y )
##    s = [x for x in s if not is_number(x)]
#    return term_vec

#CREATE SMALL PRACTICE DATASET
#practice_df = all_data_df[0:500].copy()
#
#practice_df['narrative_numbers_removed'] = practice_df['narrative_stopwords_removed'].apply(remove_numbers)
#practice_df['cause_numbers_removed'] = practice_df['cause_stopwords_removed'].apply(remove_numbers)
#
#Or, remove stop words from non-tokenized string, which outputs a string
#def remove_stops_two(s):
#    list_of_words = s.split()
#    resultwords  = [word for word in list_of_words if word.lower() not in stop_words]
#    result = ' '.join(resultwords)
#    return result
#
#practice_df['narrative_stopwords_removed'] = practice_df['narrative_punctuation_removed'].apply(remove_stops_two)
#practice_df['cause_stopwords_removed'] = practice_df['cause_punctuation_removed'].apply(remove_stops_two)


##Combine all the text and run the vectorizer below to see what top words are:
#list_of_all_words_cause = []
#for all_lists in all_data_df['cause_stopwords_removed']:
#    for second_list in all_lists:
#        for all_words in second_list:
#            list_of_all_words_cause.append(all_words)

##column wise df processing of text, memory intense and superseded
#all_data_df['narrative_nonletters_removed'] = all_data_df['narrative'].apply(remove_nonletters)
#all_data_df['cause_nonletters_removed'] = all_data_df['probable_cause'].apply(remove_nonletters)
##Remove original data to save resources
#all_data_df.drop('narrative', axis=1, inplace=True)
#all_data_df.drop('probable_cause', axis=1, inplace=True)
#
#all_data_df['narrative_tokenized'] = all_data_df['narrative_nonletters_removed'].apply(tokenize_field)
#all_data_df['cause_tokenized'] = all_data_df['cause_nonletters_removed'].apply(tokenize_field)
##Remove previous step data to save resources
#all_data_df.drop('narrative_nonletters_removed', axis=1, inplace=True)
#all_data_df.drop('cause_nonletters_removed', axis=1, inplace=True)
#
#all_data_df['narrative_stopwords_removed'] = all_data_df['narrative_tokenized'].apply(remove_stops)
#all_data_df['cause_stopwords_removed'] = all_data_df['cause_tokenized'].apply(remove_stops)
##Remove previous step data to save resources
#all_data_df.drop('narrative_tokenized', axis=1, inplace=True)
#all_data_df.drop('cause_tokenized', axis=1, inplace=True)
#
################################
##    Text Mining
##        Ngram Frequency Analysis
################################
##Add columns for narrative and probable cause text processing below
#all_data_df['narrative_nonletters_removed']=""
#all_data_df['cause_nonletters_removed']=""
#all_data_df['narrative_tokenized']=""
#all_data_df['cause_tokenized']=""
#all_data_df['narrative_stopwords_removed']=""
#all_data_df['cause_stopwords_removed']=""
#
##Determine main topics of each set of data (4 of them) with ngram frequency analysis
##Process each set of data into a final dataframe, then drop interim data to save memory
##Steps spelled out for first set, and repeat for rest
##Final sorted ngram results put into this dataframe
#cause_results_df = pd.DataFrame(columns=['small_fatal','small_nonfatal','large_fatal','large_nonfatal'])
#
##Slice off subsets of small and large planes, fatal and non-fatal accidents
#all_small_fatal_data_df = all_data_df[(
#    all_data_df["total_passengers"]>0) & (
#    all_data_df["total_passengers"]<5) & (
#    all_data_df["TotalFatalInjuries"]>0)].copy()
#list_of_all_words_small_fatal_cause = []
##merge all words for analysis
#for all_lists in all_small_fatal_data_df['cause_stopwords_removed']:
#    for second_list in all_lists:
#        for all_words in second_list:
#            list_of_all_words_small_fatal_cause.append(all_words)
##remove df to save memory
#del all_small_fatal_data_df
##Create lists of ngrams (3-7 words) for each category
#list_of_ngrams_small_fatal_cause = []
#for i in range(3,7):
#    ngrams_list = []
#    ngrams_list = list(ngrams(list_of_all_words_small_fatal_cause, i))
#    list_of_ngrams_small_fatal_cause = list_of_ngrams_small_fatal_cause + ngrams_list
##remove large list to save memory
#del list_of_all_words_small_fatal_cause
##Make frequency counts of the ngrams to see which are most common
#counts_small_fatal_cause    = Counter(list_of_ngrams_small_fatal_cause)
##remove large list to save memory
#del list_of_ngrams_small_fatal_cause
#cause_results_df['small_fatal']     =counts_small_fatal_cause.most_common(50)
##remove large list to save memory
#del counts_small_fatal_cause
#
##REPEAT ABOVE PROCEDURE FOR SMALL-NONFATAL ACCIDENTS
#all_small_nonfatal_data_df = all_data_df[(
#    all_data_df["total_passengers"]>0) & (
#    all_data_df["total_passengers"]<5) & (
#    all_data_df["TotalFatalInjuries"]==0)].copy()
#list_of_all_words_small_nonfatal_cause = []
#for all_lists in all_small_nonfatal_data_df['cause_stopwords_removed']:
#    for second_list in all_lists:
#        for all_words in second_list:
#            list_of_all_words_small_nonfatal_cause.append(all_words)
#del all_small_nonfatal_data_df
#list_of_ngrams_small_nonfatal_cause = []
#for i in range(3,7):
#    ngrams_list = []
#    ngrams_list = list(ngrams(list_of_all_words_small_nonfatal_cause, i))
#    list_of_ngrams_small_nonfatal_cause = list_of_ngrams_small_nonfatal_cause + ngrams_list
#del list_of_all_words_small_nonfatal_cause
#counts_small_nonfatal_cause = Counter(list_of_ngrams_small_nonfatal_cause)
#del list_of_ngrams_small_nonfatal_cause
#cause_results_df['small_nonfatal']  =counts_small_nonfatal_cause.most_common(50)
#del counts_small_nonfatal_cause
#
##REPEAT ABOVE PROCEDURE FOR LARGE-FATAL ACCIDENTS
#all_large_fatal_data_df = all_data_df[(
#    all_data_df["total_passengers"]>=20) & (
#    all_data_df["TotalFatalInjuries"]>0)].copy()
##OPTIONAL
##remove 3 of 4 9/11 attack rows, repetition floods analysis
#all_large_fatal_data_df = all_large_fatal_data_df[all_large_fatal_data_df['EventId'] != '20020123X00106']
#all_large_fatal_data_df = all_large_fatal_data_df[all_large_fatal_data_df['EventId'] != '20020123X00105']
#all_large_fatal_data_df = all_large_fatal_data_df[all_large_fatal_data_df['EventId'] != '20020123X00104']
#
#list_of_all_words_large_fatal_cause = []
#for all_lists in all_large_fatal_data_df['cause_stopwords_removed']:
#    for second_list in all_lists:
#        for all_words in second_list:
#            list_of_all_words_large_fatal_cause.append(all_words)
#del all_large_fatal_data_df
#list_of_ngrams_large_fatal_cause = []
#for i in range(3,7):
#    ngrams_list = []
#    ngrams_list = list(ngrams(list_of_all_words_large_fatal_cause, i))
#    list_of_ngrams_large_fatal_cause = list_of_ngrams_large_fatal_cause + ngrams_list
#del list_of_all_words_large_fatal_cause
#counts_large_fatal_cause    = Counter(list_of_ngrams_large_fatal_cause)
#del list_of_ngrams_large_fatal_cause
#cause_results_df['large_fatal']     =counts_large_fatal_cause.most_common(50)
#del counts_large_fatal_cause
#
##REPEAT ABOVE PROCEDURE FOR LARGE-NONFATAL ACCIDENTS
#all_large_nonfatal_data_df = all_data_df[(
#    all_data_df["total_passengers"]>=20) & (
#    all_data_df["TotalFatalInjuries"]==0)].copy()
#list_of_all_words_large_nonfatal_cause = []
#for all_lists in all_large_nonfatal_data_df['cause_stopwords_removed']:
#    for second_list in all_lists:
#        for all_words in second_list:
#            list_of_all_words_large_nonfatal_cause.append(all_words)
#del all_large_nonfatal_data_df
#list_of_ngrams_large_nonfatal_cause = []
#for i in range(3,7):
#    ngrams_list = []
#    ngrams_list = list(ngrams(list_of_all_words_large_nonfatal_cause, i))
#    list_of_ngrams_large_nonfatal_cause = list_of_ngrams_large_nonfatal_cause + ngrams_list
#del list_of_all_words_large_nonfatal_cause
#counts_large_nonfatal_cause = Counter(list_of_ngrams_large_nonfatal_cause)
#del list_of_ngrams_large_nonfatal_cause
#cause_results_df['large_nonfatal']  =counts_large_nonfatal_cause.most_common(50)
#del counts_large_nonfatal_cause
#
##Notes on results
#    #Lots of deduplication still needed for top causes, many ngrams overlap somewhat