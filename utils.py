# file used to store relevant functions that we may need
# In a notebook, install dependencies with:
#!pip install pandas requests python-dotenv unidecode nltk
# and then:
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

import re
import os
import time
import nltk
import pandas as pd
import requests
from dotenv import load_dotenv
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer


def ensure_nltk_data():
    """Download the corpora used by the text-preprocessing functions if missing."""
    required = ['punkt', 'stopwords', 'wordnet']
    for resource in required:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)


ensure_nltk_data()

load_dotenv(override=True)   # override=True: re-read .env even if the variable was already loaded earlier in this kernel
token = os.getenv('GENIUS_API_KEY')

# basic functions that will probably be needed to clean/preprocess the Lyrics

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
    lyrics = re.sub(r'\s+', ' ', lyrics)  # replace multiple spaces with a single space
    return lyrics.strip()  # remove leading and trailing whitespace

def tokenize_lyrics(lyrics):
    '''Tokenize the lyrics and remove unwanted characters like abreviations and contractions'''
    tokenized_text = nltk.tokenize.word_tokenize(clean_lyrics(lyrics)) #separate the lyrics into tokens

    tokenized_text = [re.sub("'m","am",token) for token in tokenized_text]
    tokenized_text = [re.sub("n't","not",token) for token in tokenized_text]
    tokenized_text = [re.sub("'s","is",token) for token in tokenized_text]
    return tokenized_text

# turning tokens into lemmas (using literally the function from class notebooks)
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

def get_song_info(song_id):
    '''
    Faz uma chamada(request) à API, dá o id da música e recebe(get) o id, o titulo, o artista, e a data de acordo 
    com a base de dados do Genius
    '''
    r = requests.get(f'https://api.genius.com/songs/{song_id}', #site do genius para fazer calls a API
                     headers={'Authorization': f'Bearer {token}'}, timeout=20) #usa a API key no .env para ser auturizado a fazer a call
    
    r.raise_for_status() # verefica se pedido correu bem
    
    s = r.json()['response']['song'] # transforma a resposta que vem em json em um dict, vai para a informação util do pedido
    
    # nas traduções o primary_artist é o tradutor; o artista original vem em 'translation_of'
    artist = s['primary_artist']['name']
    
    # devolve um dicionario com a id,titulo, artista e year tirado do json
    for rel in s['song_relationships']:
        if rel['relationship_type'] == 'translation_of' and rel['songs']:
            artist = rel['songs'][0]['primary_artist']['name']
    
    return {
        'id': song_id,
        'title': s['title'],
        'artist': artist,
        'release_date': s.get('release_date'),
    }
    
def get_info_batch(ids : pd.DataFrame or list[int],title = False,artist = False,release_date = False) : 
    ''' 
    Recebe uma lista ou dataframe de song_id e transforma em um dataframe com os argumentos que queremos saídos do Genius
    '''
    rows = []
    
    for song_id in ids:
        try:
            rows.append(get_song_info(song_id))
        except requests.RequestException:   # 404 (música apagada), timeout, erro na rede
            rows.append({'id': song_id})    # fica só o id, o resto NaN
        time.sleep(0.25)
    # recebe todos os id e transforma-os em um dataframe
    
    songs = pd.DataFrame(rows).set_index('id')

    wanted = [name for name, keep in [('title', title), ('artist', artist), ('release_date', release_date)] if keep]
    return songs.reindex(columns=wanted)

def artist_from_title(title):
    # Strip common translation suffixes before trying to extract the artist
    t = re.sub(
        r'\s*(english\s+)?translations?\s*$'
        r'|\s*traduc\w*\s*$'           # Spanish/Portuguese: traducción, tradução
        r'|\s*traduction\s*$'          # French
        r'|\s*\(.*?translat\w*\)\s*$', # e.g. "(Official English Translation)"
        '', title, flags=re.I
    )
    parts = re.split(r'\s[-–—]\s', t, maxsplit=1)
    return parts[0].strip() if len(parts) == 2 else None

def fix_translation_artists(df):
    '''
    Recebe um dataframe com as colunas id e title, já só com as linhas cujo
    artista é uma conta da plataforma Genius (e.g. "Genius English Translations",
    "Rap Genius") e devolve um novo dataframe com id e o artista real extraído
    do título. Devolve None quando o título não segue o formato "Artista - Título".
    '''
    result = df[['id']].copy()
    result['artist'] = df['title'].fillna('').map(artist_from_title)

    return result

def fix_all_translation_artists(df):
    """Return a copy with Genius translation-account artists inferred from titles.

    Rows whose artist cannot be extracted from the title are left unchanged.
    """
    result = df.copy()
    translation_artist_mask = result['artist'].astype('string').str.contains(
        r'\bgenius\b.*\b(?:translat\w*|trad\w*|romaniz\w*|romanis\w*|transliter\w*)\b',
        case=False,
        na=False,
        regex=True,
    )
    inferred_artists = result.loc[translation_artist_mask, 'title'].fillna('').map(
        artist_from_title
    )
    replace_mask = inferred_artists.notna() & inferred_artists.ne('')
    result.loc[inferred_artists.index[replace_mask], 'artist'] = inferred_artists.loc[
        replace_mask
    ]

    return result



def correct_pre_years(treshold, data, split_name, suspicious_year_threshold=1800):
    """Correct pre-threshold years and classify Genius lookup results by ID.

    Years below ``suspicious_year_threshold`` and rows with no usable API year
    are set to ``pd.NA``. Suspicious and not-found IDs are returned separately.
    """
    data = data.copy()
    numeric_year = pd.to_numeric(data['year'], errors='coerce')
    mask = numeric_year < treshold
    ids = data.loc[mask, 'id']

    print(f"{split_name}: {len(ids)} rows with years before {treshold}")
    if ids.empty:
        return data, [], [], []

    info = get_info_batch(ids, release_date=True)
    fetched_years = pd.to_numeric(
        info['release_date'].astype('string').str.extract(r'(\d{4})', expand=False),
        errors='coerce'
    )

    replacements = data.loc[mask, 'id'].map(fetched_years)
    has_year = replacements.notna()
    suspicious = has_year & replacements.lt(suspicious_year_threshold)
    resolved = has_year & ~suspicious
    not_found = ~has_year

    data.loc[replacements.index[resolved], 'year'] = replacements[resolved].astype(int)
    clear_year = suspicious | not_found
    if clear_year.any():
        data['year'] = data['year'].astype('object')
        data.loc[replacements.index[clear_year], 'year'] = pd.NA

    resolved_ids = data.loc[replacements.index[resolved], 'id'].tolist()
    suspicious_ids = data.loc[replacements.index[suspicious], 'id'].tolist()
    not_found_ids = data.loc[replacements.index[not_found], 'id'].tolist()

    print(f"{split_name}: accepted {resolved.sum()} years; "
          f"{suspicious.sum()} suspicious; {not_found.sum()} not found")
    return data, resolved_ids, suspicious_ids, not_found_ids

def summarize_years(frame, ids, split, status):
    years = pd.to_numeric(
        frame.loc[frame['id'].isin(ids), 'year'],
        errors='coerce'
    ).dropna()

    return {
        'split': split,
        'status': status,
        'rows_with_year': len(years),
        'minimum_year': years.min() if not years.empty else None,
        'years_below_1900': int((years < 1900).sum()),
        'maximum_year': years.max() if not years.empty else None,
    }




def padding_lyrics(tokenized_lyrics_list, max_len=None, padding_token="0"):
    """
    Applies padding to a list of tokenized lyrics so they all have the same length.
    """
    # 1. If no len is provided, automatically find the length of the longest song
    if max_len is None:
        max_len = max(len(lyrics) for lyrics in tokenized_lyrics_list)
    
    padded_dataset = []
    
    for lyrics in tokenized_lyrics_list:
        if len(lyrics) < max_len:
            # If the song is shorter, append the padding_token at the END (Post-padding)
            num_padding_needed = max_len - len(lyrics)
            padded_lyrics = lyrics + [padding_token] * num_padding_needed
        else:
            # If the song is longer than max_len, truncate it
            padded_lyrics = lyrics[:max_len]
            
        padded_dataset.append(padded_lyrics)
        
    return padded_dataset
