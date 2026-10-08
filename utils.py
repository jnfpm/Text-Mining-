#File used to store relevant functions that we may need
import re
import nltk
import requests
import os
from dotenv import load_dotenv
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer
import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)   # override=True: re-read .env even if the variable was already loaded earlier in this kernel
token = os.getenv('GENIUS_API_KEY')

#Basic functions that will probably be needed to clean/preprocess the Lyrics

def removing_stopwords(lyrics):
    '''Remove stopwords from the lyrics'''
    stopwords = nltk.corpus.stopwords.words('english') #defines a list of stopwords in english
    lyrics = [word for word in lyrics if word not in stopwords]
    return lyrics

def clean_lyrics(lyrics):
    '''Remove unwanted characters and patterns from the lyrics
    Not removing words between parentheses because they usually represent choruses example: "I need you (need you)".'''
    
    lyrics = unidecode(lyrics)  # Convert to ASCII (basically takes accentuation)
    lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove text within brackets
    lyrics = lyrics.lower() #ower case all the lyrics
    lyrics = re.sub(r'[^a-zA-Z0-9\s]', '', lyrics)  # Remove special characters
    lyrics = re.sub(r'\s+', ' ', lyrics)  # Replace multiple spaces with a single space
    return lyrics.strip()  # Remove leading and trailing whitespace

def tokenize_lyrics(lyrics):
    '''Tokenize the lyrics and remove unwanted characters like abreviations and contractions'''
    tokenized_text = nltk.tokenize.word_tokenize(clean_lyrics(lyrics)) #separate the lyrics into tokens

    tokenized_text = [re.sub("'m","am",token) for token in tokenized_text]
    tokenized_text = [re.sub("n't","not",token) for token in tokenized_text]
    tokenized_text = [re.sub("'s","is",token) for token in tokenized_text]
    return tokenized_text

#turning tokens into lemmas (using literally the function from class notebooks)
def lemmatize_all(token, list_pos=["n","v","a","r","s"]):
    """Apply WordNet lemmatization for each requested part-of-speech tag.

    Inputs are one token and an ordered list of WordNet POS tags. The function
    repeatedly updates the token with each lemmatization pass and returns the
    final lemma.
    """
    wordnet_lem = nltk.stem.WordNetLemmatizer() 

    return [wordnet_lem.lemmatize(token, arg_1) for arg_1 in list_pos] #only difference: comprehension instead of for loop, but same result

#in case we need steemming instead of lemmatization
def stemming_all(token):
    """Apply Porter stemming to a token."""
    porter_stemmer = nltk.stem.PorterStemmer()
    return [[porter_stemmer.stem(token) for token in token]]

# chamada a API do genius que retorna os atributos da musica
def get_song_info(song_id):
    '''
    Faz uma chamada(request) à API, dá o id da música e recebe(get) o id, o titulo, o artista, e a data de acordo 
    com a base de dados do Genius
    '''
    r = requests.get(f'https://api.genius.com/songs/{song_id}', #site do genius para fazer calls a API
                     headers={'Authorization': f'Bearer {token}'}, timeout=20) #usa a API key no .env para ser auturizado a fazer a call
    
    r.raise_for_status() # verefica se pedido correu bem
    
    s = r.json()['response']['song'] # transforma a resposta que vem em json em um dict, vai para a informação util do pedido
    
    # devolve um dicionario com a id,titulo, artista e year tirado do json
    
    return {
        'id': song_id,
        'title': s['title'],
        'artist': s['primary_artist']['name'],
        'release_date': s.get('release_date'),
    }

def get_info_batch(id : pd.DataFrame or list[int],title = False,artist = False,release_date = False) : 
    ''' 
    Recebe uma lista ou dataframe de song_id e transforma em um dataframe com os argumentos que queremos saídos do Genius
    '''
    
    # recebe todos os id e transforma-os em um dataframe
    songs = pd.DataFrame([get_song_info(i) for i in id]).set_index('id')
    
    # boolean mask apenas mete no daframe final os argumentos que são true
    filtered_songs = songs.loc[:,[title,artist,release_date]]
    print(filtered_songs)


    
