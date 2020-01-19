import re
import nltk
from nltk.corpus import wordnet
from bs4 import BeautifulSoup
from bs4.element import Comment


# Lemmatizes text and returns string of the lemmatized words
def lemmatize(body_text):
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    words = nltk.word_tokenize(body_text)
    tagged = nltk.pos_tag(words)
    lem_words = list()
    # Filters words that are not matched in regex filter
    for word, tag in tagged:
        # Filters out non alphanumerical words
        if not bool(re.match('^[a-zA-Z0-9]+$', word)):
            words.remove(word)
        else:
            wntag = get_wordnet_pos(tag)
            if wntag is None:
                lemma = wordnet_lemmatizer.lemmatize(word.lower())
            else:
                lemma = wordnet_lemmatizer.lemmatize(word.lower(), pos=wntag)
            lem_words.append(lemma)
    return " ".join(map(str, lem_words))


# parses out only visible text from the website and returns a string of it
def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)


# Returns a bool to tell if an element is not in a certain tag
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

# Tags parts of speech so the lemmatizer knows how to properly lemmatize a given word
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None