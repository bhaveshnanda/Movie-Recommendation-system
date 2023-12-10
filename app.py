import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3

movies_data = pd.read_csv('movies.csv')

selected_features = ['genres','keywords','tagline','cast','director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

vectorizer = TfidfVectorizer()

feature_vector  = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vector)

list_of_all_titles = movies_data['title'].tolist()
st.set_page_config(
        page_title="Movie-Recommendation-System",
)
st.title('Movie Recommendation System')


selected_movie_name = st.selectbox(
    'Tell me your favourite movie',
    list_of_all_titles    
)

find_close_match = difflib.get_close_matches(selected_movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

silmilarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(silmilarity_score, key = lambda x:x[1], reverse = True)

if st.button('Recommend'):
    st.write('Your selected movie was ' + selected_movie_name)
    st.write('')
    st.write('Recommended movies are:')
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index==index]['title'].values[0]
        if (i<11):
            st.write(i,'.',title_from_index)
            i+=1
