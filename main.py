import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
from string import punctuation

from fibonacci import FibonacciHeap

def data_representation(filename):
    with open(filename, "r") as f:
        text = f.read()

    doc = nlp(text)
    sentence = list(doc.sents)[8]

    with open("part_of_speech_tagging.html", "w") as f:
        f.write(displacy.render(sentence, style="dep"))

    with open("entity_recognition.html", "w") as f:
        f.write(displacy.render(doc, style="ent"))
    
    return 

def open_file(): 
    with open("input.txt", "r") as f:
        return f.read()


def most_used_words(doc, n_words): 
    """
    Input: 
        doc: nlp item containing the text
        n_words: the nth first most used words
    Return:
        output: List[(word: string, frequency: int)] (sorted)
    """
    if not doc: 
        raise TypeError("The doc is null.")
    
    if len(list(doc)) < n_words:
        message = "There is not enough words in the document. It has " + str(len(list(doc))) +" words but you want to get " + str(n_words) + "."
        raise ValueError(message)
    if n_words <= 0:
        raise ValueError('Please provide a valid number of words (strictly positive)')
    
    keyword = {}
    stopwords = list(STOP_WORDS)
    pos_tag = ["PROPN", "ADJ", "NOUN", "VERB"]

    for token in doc: 
        if (token.text in stopwords or token.text in punctuation):
            continue 
        if (token.pos_ in pos_tag):
            keyword[token.text] = keyword.get(token.text, 0) + 1

    return sorted(keyword.items(), key=lambda tuple: tuple[1], reverse=True)[:n_words]

def normalization(words):
    """
    Input:
        words: List[(word: string, frequency: int)]
    Return: 
        output: Dic[(word: string | normalized_frequency: float)]
    """
    if words is None or words == []:
        raise ValueError("Please provid a non empty list")
    
    max_freq = words[0][1]
    weighted_words = {}

    for (word, freq) in words: 
        weighted_words[word] =  freq/max_freq
    
    return weighted_words

def weighted_sentences(doc, weighted_words, n_sents):
    """
    Input: 
        weighted_words: Dic[((word: string | weight: float))]
        doc:nlp item containing the text
        n_sents: the nth first weighted sentences
    Return:
        output: List[(sentence: spacy.token.span.Span, weight: float)] 

    """

    if not doc: 
        raise TypeError("No doc.")
    
    if not weighted_words: 
        raise ValueError("No weighted words.")
    
    if n_sents > len(list(doc)) : 
        message = "There is not enough words in the document. It has " + str(len(list(doc))) +" words but you want to get " + str(n_sents) + "."
        raise ValueError(message)

    sent_strength = FibonacciHeap()

    for sent in doc.sents:
        sent_weight = get_weight_sentence(sent, weighted_words)
        sent_strength.insertion(sent, sent_weight)
    
    weighted_sentences = []
    while len(weighted_sentences) < n_sents :
        (sent, old_weight) = sent_strength.extract_max()
        new_weight = get_weight_sentence(sent, weighted_words)

        if old_weight == new_weight:
            weighted_sentences.append(sent)
            weighted_words = zero_weighted_words_from_sentence(weighted_words, sent)
        else : 
            sent_strength.insertion(sent, new_weight)

    return weighted_sentences

def zero_weighted_words_from_sentence(weighted_words, sent):
    """
    Input: 
        weighted_words: Dic[word: string, weight: float]
        sent: spacy.token.span.Span
    Return : 
        output: Dic[word: string, weight: float]
    """
    for token in sent: 
        if token.text in weighted_words:
            weighted_words[token.text] = weighted_words[token.text]

    return weighted_words

def get_weight_sentence(sent, weighted_words): 
    """
    Input: 
        sent: spacy.token.span.Span
        weighted_words: Dic[word: string, weight: float]
    Return: 
        output: float
    """
    sent_weight = 0
    for token in sent:
        sent_weight += weighted_words.get(token.text, 0.0)

    return sent_weight


def get_short_version(weighted_sentences): 
    """
    Input: 
        weighted_setences: List[(sentence: spacy.token.span.Span)]
    Return:
        output: string
    """
    if weighted_sentences is None: 
        raise ValueError("No sentences.")
    
    text = ""
    for sentence in weighted_sentences: 
        text += " " + sentence.text

    return text


nlp = spacy.load("en_core_web_md")
text = open_file()
doc = nlp(text)

words = most_used_words(doc, 10)
weighted_words = normalization(words)
weighted_sents = weighted_sentences(doc, weighted_words, 3)
short_version = get_short_version(weighted_sents)

print(short_version)


