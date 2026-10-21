#File used to store relevant functions that we may need
import re
import nltk
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import unicodedata
from text_quality import audit_lyrics, normalize_lyrics
from nltk.tokenize.treebank import TreebankWordDetokenizer
import time

load_dotenv(override=True)   # override=True: re-read .env even if the variable was already loaded earlier in this kernel
token = os.getenv('GENIUS_API_KEY')

#Basic functions that will probably be needed to clean/preprocess the Lyrics

def removing_stopwords(lyrics):
    '''Remove stopwords from the lyrics'''
    stopwords = nltk.corpus.stopwords.words('english') #defines a list of stopwords in english
    lyrics = [word for word in lyrics if word not in stopwords]
    return lyrics

def clean_lyrics(lyrics):
    '''Prepare Unicode lyrics for word models; keep accents and contractions.

    Bracketed section labels are removed. Parenthetical sung words remain.
    Punctuation becomes a boundary so "love/hate" does not become "lovehate".
    For a typography-only copy preserving case/structure, use normalize_lyrics.
    '''
    lyrics = normalize_lyrics(lyrics, preserve_lines=False)
    if not isinstance(lyrics, str):
        return lyrics
    lyrics = re.sub(r'\[.*?\]', ' ', lyrics)
    lyrics = lyrics.lower()
    lyrics = ''.join(
        c if c.isalnum() or c.isspace() or c in "'\u200c\u200d" or unicodedata.category(c).startswith('M')
        else ' ' for c in lyrics
    )
    return re.sub(r'\s+', ' ', lyrics).strip()

def tokenize_lyrics(lyrics):
    '''Tokenize Unicode lyrics and expand only unambiguous contraction tokens.'''
    tokenized_text = nltk.tokenize.word_tokenize(clean_lyrics(lyrics)) #separate the lyrics into tokens
    # 's can mean a possessive, "is", or "has"; preserve that ambiguity.
    contractions = {"'m": "am", "n't": "not"}
    return [contractions.get(token, token) for token in tokenized_text]

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
    
    #songs = pd.DataFrame([get_song_info(i) for i in id]).set_index('id')
    
    # boolean mask apenas mete no daframe final os argumentos que são true
    #filtered_songs = songs.loc[:,[title,artist,release_date]]
    #print(filtered_songs)


def artist_from_title(title):
    t = re.sub(r'\s*english\s+translation\s*$', '', title, flags=re.I)
    parts = re.split(r'\s[-–—]\s', t, maxsplit=1)
    return parts[0].strip() if len(parts) == 2 else None

def fix_translation_artists(df):
    '''
    Recebe um dataframe com as colunas id e title, já só com as de artista Genius English Translation 
    e devolve um novo com o id,artista, correto
    '''
    
    result = df[['id']].copy()
    result['artist'] = df['title'].fillna('').map(artist_from_title)

    return result

def normalize_lyrics_unicode(
    text,
    *,
    repair_mixed_script=False,
    preserve_lines=True,
):
    """Normalize Unicode lyrics without removing accents or other languages."""

    # Preserve missing values rather than turning them into the word "nan".
    if not isinstance(text, str):
        if pd.api.types.is_scalar(text) and pd.isna(text):
            return text
        raise TypeError("Expected a string or a missing value.")

    # Compose equivalent characters: e + combining accent becomes é.
    text = ud.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Recover recognizable Windows-1252 punctuation stored as control codes.
    def recover_control(match):
        char = match.group()
        if char == "\x85":
            return "\n"  # Unicode NEXT LINE; legacy ellipsis usage is ambiguous.
        try:
            return bytes([ord(char)]).decode("cp1252")
        except UnicodeDecodeError:
            return " "

    text = re.sub(r"[\x80-\x9f]", recover_control, text)

    # Remove zero-width scraping artifacts inside Latin words.
    # Example: "T\u200bhe" -> "The".
    def remove_inside_word(match):
        left = match.group(1)
        right = match.string[match.end()]
        if all("LATIN" in ud.name(c, "") for c in (left, right)):
            return left
        return left + " "

    text = re.sub(
        r"([^\W\d_])\u200b(?=[^\W\d_])",
        remove_inside_word,
        text,
    )

    replacements = {
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
        "…": "...",
        "\u200b": " ",   # Zero-width space outside Latin words.
        "\ufeff": "",    # Embedded byte-order mark.
        "\u00ad": "",    # Discretionary soft hyphen.
        "\u2060": "",    # Word joiner.
    }

    # Remove directional formatting marks, preserving linguistic ZWJ/ZWNJ.
    directional_marks = (
        0x200E, 0x200F,
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
    )
    replacements.update({chr(cp): "" for cp in directional_marks})
    text = text.translate(str.maketrans(replacements))

    # Optional correction for the corpus-specific Cyrillic е/Е artifact.
    if repair_mixed_script and ("е" in text or "Е" in text):
        letters = [c for c in text if c.isalpha()]
        latin_count = sum("LATIN" in ud.name(c, "") for c in letters)

        if latin_count >= 0.9 * len(letters):
            def repair_word(match):
                word = match.group()
                latin = sum("LATIN" in ud.name(c, "") for c in word)
                allowed = all(
                    "LATIN" in ud.name(c, "") or c in "еЕ"
                    for c in word
                )

                if latin and latin * 2 >= len(word) and allowed:
                    return word.translate(str.maketrans({"е": "e", "Е": "E"}))
                return word

            text = re.sub(r"[^\W\d_]+", repair_word, text)

    # Standardize spaces and line endings without joining separate words.
    spacing = {}
    for char in set(text):
        if char in "\n\u2028\u2029":
            spacing[ord(char)] = "\n" if preserve_lines else " "
        elif char.isspace() or ud.category(char) == "Cc" or char == "\ufffc":
            spacing[ord(char)] = " "

    text = text.translate(spacing)
    text = re.sub(r" +", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    return ud.normalize("NFC", text.strip())