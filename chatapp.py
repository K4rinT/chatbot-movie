import os
import time
import json
import spacy
import pickle
import random
import requests
import numpy as np
import pandas as pd

from tensorflow import keras
from gensim.models import FastText
from chatmodel import ChatModel as chatmodel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# [G]: Global Variables
model = "all-mpnet-base-v2"
sbert = SentenceTransformer(model)
nlp = spacy.load('en_core_web_sm')
movie_master = []                   # Follow_up question (Note: Might change to store value inside class)

class ChatApp:
    def __init__(self):
        self._chatmodel = chatmodel()
        self._train_x, self._train_y = self._chatmodel.get_train_x(), self._chatmodel.get_train_y()
        self._responses = self._chatmodel.get_responses()
        self._intents = self._chatmodel.get_intents()
        self._corpus, self._similarities = self._chatmodel.get_corpus(), self._chatmodel.get_similarities()
        self._corpus_list = self.corpus_list(self._corpus)
        self._corpus_emb = pickle.load(open('movie_embeddings.pkl','rb'))
        self._ft_model = FastText.load('ft_movie_corpus.model')
        # self._model = keras.models.load_model('chatbot_model.keras')      # Not use this time

    @staticmethod
    def sentence_embedding(sentence):
        """ Sentence BERT's Embedding """
        sbert_embedding = sbert.encode(sentence)

        return sbert_embedding
    
    def predict_class(self, sentence, error_thres=0.75):
        """ Predicting Intention (Label) """
        # [1]: Get movie's name by using NER
        doc = nlp(sentence)
        movie_input = [ent.text for ent in doc.ents]

        # [2]: Extracted movie name and intention sentence
        if movie_input:
            # [2.1]: Movie Embedded -> Retrieve the correct name
            print('Movie Input!')
            input_emb = self._ft_model.wv[movie_input[0]]
            sim = {movie: cosine_similarity([input_emb], [embedding])[0][0] for movie, embedding in self._corpus_emb.items()}

            # [2.2]: Pass to movie_master (Global Variable)     (Note: Might change to store value inside class)
            filtered_sim = {movie: similarity for movie, similarity in sim.items()}
            movie_input = max(filtered_sim, key=filtered_sim.get)
            movie_master.append(movie_input)

            # [2.3]: Extract intention only!
            extracted_text = ""
            for token in doc:
                if token.pos_ == "PROPN":
                    break
                extracted_text += token.text + " "

            extracted_text = extracted_text.strip()
            sentence = extracted_text

        # [3]: Sentence Embedding
        embedded = ChatApp.sentence_embedding(sentence)
        embedded = np.array(embedded).reshape(1,-1)

        # [4]: Type Checking: Before Semantic Similarity
        if isinstance(self._train_x, pd.Series):
            self._train_x = np.stack(self._train_x.values)
        embedded = embedded.astype(np.float64)
        self._train_x = self._train_x.astype(np.float64)

        # [5]: Semantic Similarity
        print('Predict Start!')         # Backend Check
        sim_sbert = cosine_similarity(embedded, self._train_x)
        results = [[class_, prob] for class_, prob in zip(self._train_y, sim_sbert[0]) if prob>error_thres]
        results.sort(key=lambda x: x[1], reverse=True)
        results_list = [{'intent': result[0], 'probability': result[1]} for result in results]

        # [5]: Keras Classifier predicted               # Note: Not use, if use this code need recheck!
        # clf = KerasClassifier(self._model)
        # clf.initialize(self._train_x, self._train_y)
        # predicted_classes = clf.classes_
        # predicted_probability = clf.predict_proba(embedded.reshape(1,-1)).reshape(-1)
        # results = [[predict_class, prob] for predict_class, prob in zip(predicted_classes, predicted_probability) if prob>error_threshold]
        # results_list = [{'intent': result[0], 'probability': result[1]} for result in results]
        
        return results_list, movie_input

    def format_movie_detail(self, response_template, detail_value):
        """ Retrieve the random responses based on tags """
        detail = f"{random.choice(response_template)}: "
        detail = detail.format(detail_value)

        return detail
    
    def rand_response(self, ints, movie_input, file_name='movies_data.json'):
        """ Random Response following its tag (Label) """
        # [1]: Retrieve Responses
        responses_dict = self._responses
        
        # [2]: Intention detected as 'noanswer'
        if not ints:
            return random.choice(responses_dict['noanswer'])
        
        # [3]: Retrieve tag (Label) & All responses of its tag
        tag = ints[0]['intent']
        responses_all = responses_dict[tag]
        
        # [4]: Main intentions using movie_input
        if tag in ['specific_movie', 'movie_rating', 'movie_runtime', 'movie_genre', 'movie_release']:
            # [4.1]: Follow up question (User already asked movie name)
            if movie_master:
                movie_input = movie_master[-1]
                movie_corpus = pd.DataFrame(self._corpus)
                movie_index = movie_corpus.loc[movie_corpus['title']==movie_input, 'movie_id'].iloc[0]
                
                if os.path.exists(file_name):           # File checked
                    data = json.load(open(file_name, 'r'))
                    if movie_input not in data[0]:
                        data = self.fetch_movie_data(movie_index)
                        self.save_data_to_file(movie_input, data)
                        data = json.load(open(file_name, 'r'))
                else:                                   # No file created (movie_detail)
                    data = self.fetch_movie_data(movie_index)
                    self.save_data_to_file(movie_input, data)
                    data = json.load(open(file_name, 'r'))

                if tag == 'specific_movie':
                    detail = f"{random.choice(responses_dict[tag])}: "
                    detail += f"\n{data[0][movie_input]['overview']}"      
                    return detail
                    
                elif tag == 'movie_rating':
                    return self.format_movie_detail(responses_dict[tag], data[0][movie_input]['vote_average'])
                    
                elif tag == 'movie_runtime':
                    return self.format_movie_detail(responses_dict[tag], data[0][movie_input]['runtime'])
                
                elif tag == 'movie_genre':
                    genres = [genre['name'] for genre in data[0][movie_input]['genres']]
                    genre_list = ', '.join(genres)
                    return self.format_movie_detail(responses_dict[tag], genre_list)
                
                else:       # tag == 'movie_release'
                    return self.format_movie_detail(responses_dict[tag], data[0][movie_input]['release_date'])
        
            else:           # Answer as 'noanswer'
                return random.choice(responses_dict['noanswer'])

        # [5]: Random movie name only
        elif tag in ['recommend_movie']:
            random_movies = random.sample(self._corpus_list, 5)
            recommend = f"{random.choice(responses_dict['recommend_movie'])}\n"
            recommend += '\n'.join([f"{idx}. {movie}" for idx, movie in enumerate(random_movies, start=1)])
            
            return recommend
        
        # [6]: Random response from the detected tag 'greeting', 'goodbye', 'thanks'
        else:
            return random.choice(responses_all)

    def chatbot_response(self, text):
        """ Get Response """
        ints, movie_input = self.predict_class(text)            # Note: movie_input might use when the value storing inside class) 
        response = self.rand_response(ints, movie_master)
        
        return response
    
    def fetch_movie_data(self, movie_idx, delay_time=10):
        """ Retreive movie data from API """
        # [1]: Parameter Init
        api_key = '474a7f15ed8defb46291a4ce150b2a44'
        base_url = 'https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US'

        # [2]: Retrieve response
        time.sleep(delay_time)
        response = requests.get(base_url.format(movie_idx, api_key))
        
        return response.json()

    def save_data_to_file(self, movie_input, new_data, file_name='movies_data.json'):
        """ Save data from API to local """
        try:
            # [1]: File Checked
            if os.path.exists(file_name):
                with open(file_name, 'r') as file:
                    try:
                        existing_data = json.load(file)
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []

            # [2]: Update data into existing data
            if existing_data:
                existing_data[0].update({movie_input: new_data})
            else:
                existing_data = [{movie_input: new_data}]

            # [3]: File saved
            with open(file_name, 'w') as file:
                json.dump(existing_data, file, indent=4)
        
        except (OSError, IOError, Exception) as e:
            print(f"An error occurred while processing the file: {e}")

    def get_movie_input(self):
        """ Get movie input """
        return self._movie_input
    
    def corpus_list(self, corpus):
        """ Get movie corpus """
        movie_corpus = [movie for _,movie in corpus['title'].items()]
        return movie_corpus