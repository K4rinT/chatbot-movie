# Chatbot Movie Recommendation

## 1. Overview and Use Cases

### Overview
When a user inputs a question, the system uses NER to extract the `movie_name` from the sentence. Embedding technique will use as follows:
1. Sentence embedding using Sentence BERT for both the patterns in `intents.json` (training data) and the user's input.
2. `movie_name` embedding using FastText.

Cosine similarity is then used to check the semantic similarity between the user_input sentence and the patterns in `intents.json` (training data). The response is randomly selected from the tag with the highest semantic similarity. The UI is created using PyQt5 to interact with the user.

### Use Cases
- **Example 1: (greeting case)**
  - **Input:** Hi 
  - **Output:** Hi! How can I help you with movies today? 

- **Example 2: (goodbye case)**
  - **Input:** bye 
  - **Output:** Goodbye! Enjoy your movie! 

- **Example 3: (thanks case)**
  - **Input:** Thank you 
  - **Output:** My pleasure 

- **Example 4: (recommend movie case)**
  - **Input:** Can you recommend me a movie (recommend_movie tag)
  - **Output:** Certainly! Here are the five films I recommend: 1. F.I.S.T., 2. Slumdog Millionaire, 3. Bran Nue Dae, 4. Rang De Basanti, 5. The Red Violin 

- **Example 5: (ask movie detail case)**
  - **Input:** Tell me about Spider-Man
  - **Output:** Providing the requested details for your review: After being bitten by a genetically altered spider at Oscorp, nerdy but endearing high school student Peter Parker is endowed with amazing powers to become the superhero known as Spider-Man. 
  - **Follow-up Questions:**
    - **Input:** What's the rating of the movie? 
    - **Output:** The movie is rated 7.297.
    - **Input:** What's the runtime of the movie? 
    - **Output:** The runtime of the movie is 121 mins.
    - **Input:** What's the genre of the movie? 
    - **Output:** It is categorized under Action, Science Fiction. 
    - **Input:** When was the movie released? 
    - **Output:** Its release date is 2002-05-01 

- **Example 6: (typo case)**
  - **Input:** Tell me about Spider-Mna 
  - **Output:** Providing the requested details for your review: After being bitten by a genetically altered spider at Oscorp, nerdy but endearing high school student Peter Parker is endowed with amazing powers to become the superhero known as Spider-Man. 

- **Example 7: (no answer case)**
  - **Input:** Tell me about D 
  - **Output:** Please give me more info 

### File Details
- **chatmodel.py:** Contains code related to the model (e.g., FastText) and intent data.
- **chatapp.py:** Contains code related to the responses when the user interacts.
- **chatgui.py:** Contains code for the GUI to interact with the user.

## 2. Techniques and Tools Used (Core Libraries)
- **PyQt5:** For creating the chatbot interface.
- **Spacy:** For NER to extract `movie_name`.
- **FastText (gensim):** For embedding `movie_name`.
- **Sentence Transformer:** For embedding intention sentences.
- **requests:** For API calls.
- **sklearn:** For cosine similarity.

## 3. Embedding Explanation
- **FastText:** Used to embed `movie_name` input by the user.
  - **Reason:** To handle cases where the user misspells the `movie_name`.
- **Sentence BERT:** Used to embed both the user's input sentences and the patterns in `intention.json`.
  - **Reason:** Provides contextual embeddings that can handle sentences with the same meaning but different vocabulary.

## 4. Hugging Face Component Explanation
- **Sentence Transformer:** Used for embedding intention sentences.

## 5. API Usage Details
- Used for fetching movie information based on the user's input. (The API was used in `chatapp.py` in `fetch_movie_data` function)

## 6. Instructions for Running and Testing the MVP				
To set up and run the MVP, follow these steps:
1. **Create a new Conda environment**:
   This command creates a new Conda environment named `your_environment_name` with Python 3.10.6. You can replace `your_environment_name` with whatever name you prefer for your environment.
```bash
conda create --name your_environment_name python=3.10.6
```
2. **Activate the Conda environment:**
This command activates the environment you just created. Make sure to replace your_environment_name with the name you used in the previous step.
```bash
conda activate your_environment_name
```
3. **Install the required packages:**
This command installs all the necessary packages listed in requirements.txt.
```bash
pip install -r requirements.txt
```
4. **Run the application:**
This command starts the application. Ensure that you are in the directory where chatgui.py is located before running this command.
```bash
python chatgui.py
```
