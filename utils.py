#File used to store relevant functions that we may need
import re
import nltk
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer

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

