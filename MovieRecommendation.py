import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3
import speech_recognition as sr

r = sr.Recognizer()


movies_data = pd.read_csv('movies.csv')

""" print(6,movies_data.head())
print(movies_data.shape) """


selected_features = ['genres', 'id','keywords','tagline','cast','director']
""" print(selected_features) """

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
# print(combined_features)


vectorizer = TfidfVectorizer()
feature_vector  = vectorizer.fit_transform(combined_features)
""" print(feature_vector) """


similarity = cosine_similarity(feature_vector)
""" print(similarity)
print(similarity.shape) """


with sr.Microphone(device_index=0, sample_rate=48000) as source:
    print('Tell me your favorite movie')
    speak = pyttsx3.init()
    voices = speak.getProperty('voices')
    speak.setProperty('voice', voices[1].id)
    speak.setProperty('rate', 180)
    speak.setProperty('pause', 0.4)
    speak.say("Tell me your Favorite Movie")
    speak.runAndWait()
    print("listening...")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source, phrase_time_limit=5) 
    movie_name = r.recognize_google(audio)
""" movie_name = input('\nEnter your favorite movie name: ') """

list_of_all_titles = movies_data['title'].tolist()
""" print(list_of_all_titles) """


find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
""" print(find_close_match) """


close_match = find_close_match[0]
""" print(close_match) """

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
""" print(index_of_the_movie) """


silmilarity_score = list(enumerate(similarity[index_of_the_movie]))
""" print(silmilarity_score) """
""" print(len(silmilarity_score)) """


sorted_similar_movies = sorted(silmilarity_score, key = lambda x:x[1], reverse = True)
""" print(sorted_similar_movies) """

print('\nMovies suggested for you: \n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if (i<11):
        print(i,'.',title_from_index)
        i+=1
        speak = pyttsx3.init()
        voices = speak.getProperty('voices')
        speak.setProperty('voice', voices[1].id)
        speak.setProperty('rate', 210)
        speak.setProperty('pause', 0.4)
        speak.say(title_from_index)
        speak.runAndWait()
        
        
        import pickle
        pickle.dump(list_of_all_titles,open('movies.pkl','wb'))