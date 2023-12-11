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
list_of_all_genres = movies_data['genres'].tolist()
st.set_page_config(
        page_title="Movie-Recommendation-System",
)

# st.title('🎥 Movie Recommendation System')
st.markdown("<h1 style='text-align: center;'>🎥 Movie Recommendation System</h1>", unsafe_allow_html=True)

column1, column2 = st.columns([1,1])

with column1:
    selected_movie_name = st.selectbox(
        '🎬 - Tell me your favourite movie',
        list_of_all_titles    
    )

with column2:
    selected_genres_name = st.selectbox(
        '🎞️ - Tell me genres which you like the most',
        list_of_all_genres    
    )

find_close_match = difflib.get_close_matches(selected_movie_name, list_of_all_titles)
find_close_match_genres = difflib.get_close_matches(selected_genres_name, list_of_all_genres)

close_match = find_close_match[0]
close_match_genres = find_close_match_genres[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
index_of_the_movie_genres = movies_data[movies_data.genres == close_match_genres]['index'].values[0]

silmilarity_score = list(enumerate(similarity[index_of_the_movie]))
silmilarity_score_genres = list(enumerate(similarity[index_of_the_movie_genres]))

sorted_similar_movies = sorted(silmilarity_score, key = lambda x:x[1], reverse = True)
sorted_similar_movies_genres = sorted(silmilarity_score_genres, key = lambda x:x[1], reverse = True)

col1, col2, = st.columns([1,1])

with col1:
    if st.button('Recommend based on movie'):
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
                speak = pyttsx3.init()
                voices = speak.getProperty('voices')
                speak.setProperty('voice', voices[1].id)
                speak.setProperty('rate', 210)
                speak.setProperty('pause', 0.30)
                speak.say(title_from_index)
                speak.runAndWait()
                
                

with col2:
    if st.button('Recommend based on genres'):
        st.write('Your selected genres was ' + selected_genres_name)
        st.write('')
        st.write('Recommended movies are:')
        i = 1
        for movie in sorted_similar_movies_genres:
            index = movie[0]
            title_from_index_genres = movies_data[movies_data.index==index]['title'].values[0]
            if (i<11):
                st.write(i,'.',title_from_index_genres)
                i+=1
                speak = pyttsx3.init()
                voices = speak.getProperty('voices')
                speak.setProperty('voice', voices[1].id)
                speak.setProperty('rate', 210)
                speak.setProperty('pause', 0.30)
                speak.say(title_from_index_genres)
                speak.runAndWait()
