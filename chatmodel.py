import json
import pickle
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from gensim.models import FastText
from scikeras.wrappers import KerasClassifier
from sentence_transformers import SentenceTransformer


# [G]: Global Variables
model = "all-mpnet-base-v2"
sbert = SentenceTransformer(model)

class ChatModel:
    def __init__(self):
        self._responses, self._intents = self.intents_data('intents.json')
        self._train_x, self._train_y = self.training_data(self._intents)
        self._corpus, self._similarities = self.fasttext_train('movies.pkl', 'similarity.pkl')
        # self._model = self.keras_training_model(self._train_x, self._train_y)             # Not use this time

    @staticmethod
    def sentence_embedding(sentence):
        """ Sentence BERT's Embedding """
        sbert_embedding = sbert.encode(sentence)

        return sbert_embedding
    
    @staticmethod
    def get_clf(meta, hidden_layer_sizes, dropout):             # Not use this time
        """ Keras Classifier Initialization """
        n_features_in_ = meta["n_features_in_"]
        n_classes_ = meta["n_classes_"]
        keras_model = keras.models.Sequential()
        keras_model.add(keras.layers.Input(shape=(n_features_in_,)))

        for hidden_layer_size in hidden_layer_sizes:
            keras_model.add(keras.layers.Dense(hidden_layer_size, activation="relu"))
            keras_model.add(keras.layers.Dropout(dropout))
        keras_model.add(keras.layers.Dense(n_classes_, activation="softmax"))

        return keras_model
    
    def intents_data(self, url):
        """ Get Intents(Label) & Responses """
        # [1]: Variable Initialization
        responses = {}
        intents_list = []

        # [2]: Load Intents (Label)
        intents_r = json.load(open(url, 'r'))

        # [3]: Get Responses & Intent (Label)
        for intent in intents_r['intents']:
            tag = intent['tag']
            responses_current = intent['responses']
            responses.setdefault(tag, responses_current)

            for pattern in intent['patterns']:
                intents_list.append({'pattern': pattern, 'tag': tag})

        return responses, intents_list
    
    def training_data(self, data):
        """ Get train_x & train_y """
        # [1]: Intents's Initialize & Embedding
        main_intents = pd.DataFrame(data)
        main_intents['emb'] = main_intents.apply(lambda col: ChatModel.sentence_embedding(col.iloc[0]), axis=1)     # 'pattern' access

        # [2]: Get train_x and train_y variable
        train_x, train_y = main_intents.loc[:, 'emb'], main_intents.loc[:, 'tag']
        print('Training Data Created!')     # Backend Check

        return train_x, train_y
    
    def fasttext_train(self, corpus_f, similarity_f):
        """ FastText Init & Retrieve movie_corpus, similarities """
        # [1]: Load Corpus and Movie Similarities
        movie_corpus = pickle.load(open(corpus_f, 'rb'))
        similarities = pickle.load(open(similarity_f,'rb'))

        # [2]: Declare FastText parameters
        kwargs = {
            'sentences': movie_corpus,
            'vector_size': 150,
            'window': 5,
            'min_count': 1,
            'workers': 4,
            'sg': 1,
            'epochs': 10,
            'negative': 10,
            'min_n': 3,
            'max_n': 6
        }

        # [3]: FastText Init & Corpus Embedding
        fasttext_model = FastText(**kwargs)
        movie_embeddings = {movie: fasttext_model.wv[movie] for _,movie in movie_corpus['title'].items()}

        # [4]: Save FastText's model & Corpus Embedding
        fasttext_model.save('ft_movie_corpus.model')
        pickle.dump(movie_embeddings, open('movie_embeddings.pkl', 'wb'))

        return movie_corpus, similarities       # similarities contained movie similarity (might use in the future)

    def keras_training_model(self, train_x, train_y):           # Not use this time
        """ Keras Classifier Training """
        # [1]: Declare Keras parameters
        kwargs = dict(
            model = ChatModel.get_clf,
            loss = 'sparse_categorical_crossentropy',
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01),
            hidden_layer_sizes = (100,),
            metrics = ['accuracy'],
            random_state = 42,
            dropout = 0.5,
            batch_size = 5,
            epochs = 200,
            verbose = 1,
        )

        # [2]: Init Classifier & Train
        clf = KerasClassifier(**kwargs)
        hist = clf.fit(train_x, train_y)

        # [3]: Save classifier
        clf.model_.save('chatbot_model.keras')
        print('Chatbot Model Created!')     # Backend Check

        return clf

    def get_train_x(self):
        """ Get train_x """
        return self._train_x

    def get_train_y(self):
        """ Get train_y """
        return self._train_y

    def get_model(self):
        """ Get Keras Classifier model """
        return self._model
    
    def get_intents(self):
        """ Get Intents (Label) """
        return self._intents
    
    def get_responses(self):
        """ Get Responses (Answer) """
        return self._responses
    
    def get_corpus(self):
        """ Get Movies Corpus """
        return self._corpus
    
    def get_similarities(self):             # Not use this time (Might use in the future)
        """ Get Movies Similarities """
        return self._similarities