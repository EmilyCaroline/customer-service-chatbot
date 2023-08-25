import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """
    breaking down phrases into smaller units called words/tokens
    It can be a word or punctuation character, or number etc
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = breaking down phrases into smaller units 
    examples:
    words = ["reads", "reading"]
    words = [stem(w) for w in words]
    -> ["read", "read"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    Assign value as 1 if the word exists in the sentence, assign value as 0 if it doesn't exists
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "see", "you"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0,        1]
    """
    # stemming
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
