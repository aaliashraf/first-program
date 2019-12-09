import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
df=pd.read_csv("movie_dataset.csv")
movie_feature=['cast','genres','director']
index_feature=['index']
#df2=df[index_feature]

def unite_feature(row):
    return row['cast']+" "+row['genres']+" "+row['director']
for feature in movie_feature:
    df[feature]=df[feature].fillna('')
df["unite_feature"]=df.apply(unite_feature,axis=1)
df1=df["unite_feature"]

def get_title(index):
    return df[df.index == index]["title"].values[0]
def get_index(title):
    return df[df.title==title]["index"].values[0]
def get_genres(index):
    return df[df.index==index]["genres"].values[0]
def get_director(index):
    return df[df.index==index]["director"].values[0]
def get_cast(index):
    return df[df.index==index]["cast"].values[0]
# user taking values
user_movie=input("Search For Movie :")
pre_movie=user_movie.capitalize()



movie_index=get_index(pre_movie)
movie_genres=get_genres(movie_index)
movie_director=get_director(movie_index)
movie_cast=get_cast(movie_index)
#print("the index is :",movie_index)
""" Now we used the technique of Bag of Words for converting text data to numerical values 
 it has 4 steps """
#print(df1)

""" # Step 1:  convert text to lower case
letter=[]
total=0
for i in df1:
    letter.append(i.lower())
  
#print(letter)
  # Step 2: split the string
processed=[]
for j in letter:
    processed.append(j.split(' '))
#print(processed)
# Step 3: count the frequency of occurences
frequency=[]

for l in processed:
    frequency.append(Counter(l))

"""
cv = CountVectorizer()
all_vectors = cv.fit_transform(df["unite_feature"])
#print(all_vectors)


#print("frequency index is\n:",movie_index) at what index the movie is 
#print("all the indexes are\n:",frequency) complete dictionary with keys and values
################
""" there values of index and frequency are the
 dictionary so perform mathematical operation according to them"""

#    vector.append(frequency[i].values())
# to get all in vectors

#user_vector=np.array(movie_vector)
#print("vector at movie index \n",user_vector)

"""#a=all_vectors[0]
#b=all_vectors[3])
#### Here the Formula will apply on vectors
print("at specific index\n",x)
print("at specific index\n",y)
a = np.array(x)
b = np.array(y)
dot = np.dot(a, b)
norma = np.linalg.norm(a)
normb = np.linalg.norm(b)
cos = dot / (norma * normb)
"""

#### formula using cos_similarity library
cosine_sim = cosine_similarity(all_vectors)
# enumerate used for listing with the indexs 
similair_movie=list(enumerate(cosine_sim[movie_index]))

#print("sorted here",similair_movie.sort())
# lambda fun. used for the sorted in order of ascending
sorted_similar_movies = sorted(similair_movie,key=lambda x:x[1],reverse=True)[1:]
#print("similar",similair_movie)
i=0
print("Top 5 similar movies to "+user_movie+" are:\n")
for index,element in enumerate(sorted_similar_movies):
    # print with index
    #print("Movie Name in Order\n {0} ={1} ".format(index," :\n"+get_title(element[0])+"\n Genres :"+get_genres(element[0])+"\nDirector :"+get_director(element[0])+"\n Cast :"+get_cast(element[0])))
    print("Movie: \n"+get_title(element[0])+"\n   Genres :"+get_genres(element[0])+" \n   Director :"+get_director(element[0])+"\n   Cast :"+get_cast(element[0]))
    i=i+1
    if i>=5:
        break

