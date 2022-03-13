import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import dill

from typing import List, Tuple

import warnings
warnings.filterwarnings('ignore')

# Relevant compounds for using nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Defining english stopwords
STOPWORDS = stopwords.words('english')
# Defining lemmatizer
LEMMATIZER = WordNetLemmatizer()

# Path to model objects
model_path = os.path.join('..', '..', 'models', 'final_model.pkl')
le_path = os.path.join('..', '..', 'models', 'le.pkl')

# Importing model
with open(model_path, 'rb') as f:
    model = dill.load(f)

# Importing label encoder
with open(le_path, 'rb') as f:
    le = dill.load(f)


def map_pos_wordnet(token: Tuple[str, str]) -> str:
    """
    function maps pos of token to particular wordnet object

    Args:
        token: tuple where first element represents word and the second the pos tag of word

    Returns:
        pos tag str representation by wordnet
    """
    if token[0].startswith('J'):
        return wordnet.ADJ
    elif token[0].startswith('V'):
        return wordnet.VERB
    elif token[0].startswith('N'):
        return wordnet.NOUN
    elif token[0].startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def transform(text: str) -> List[str]:
    """
    Function transform raw text into form suitable for model

    Args:
        text: should be lyrics of english song

    Returns:
        final_text: preprocessed text
    """
    # Striping text from errors
    text = text.rstrip('EmbedShare URLCopyEmbedCopy')

    # removing some kinds of numbers
    text = re.sub(r'\d{,}$', '', text)
    # removing not letter sequence of chars
    text = re.sub(r'\W+', ' ', text)

    # Tokenizing
    text_tokens = word_tokenize(text)
    # Removing stopwords
    text_tokens = [el for el in text_tokens if el not in STOPWORDS]

    # Creating POS tags
    pos_tokens = nltk.pos_tag(text_tokens)
    # Mapping POS tags according to wordnet convention
    pos_tokens_wordnet = [(el[0], map_pos_wordnet(el[1])) for el in pos_tokens]

    # Lemmatizing words
    tokens_lemma = [LEMMATIZER.lemmatize(token[0], pos=token[1]) for token in pos_tokens_wordnet]
    # Lowering strings
    tokens_lemma = [i.lower() for i in tokens_lemma]
    # Turning list into raw text
    final_text = ' '.join(tokens_lemma)
    return [final_text]


def predict(text: str) -> str:
    """
    Function predicts music genre from song with lyrics=`text`

    Args:
        text: lyrics of some english song

    Returns:
        capitalized music genre. {'Blues', 'Country', 'Disco', 'Hip-hop', 'Jazz', 'Pop', 'Punk',
       'R-n-b', 'Rock'}
    """
    # Transforming text into suitable form
    x = transform(text)
    # Predicting with model
    preds = model.predict(x)
    # Encoding prediction
    genre = le.inverse_transform(preds)
    return genre[0].capitalize()
