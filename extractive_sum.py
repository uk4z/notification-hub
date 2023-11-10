import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

def open_file(): 
    with open("examples/break_up/input.txt", "r") as f:
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
        n_words: the nth first most used words
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

    sent_strength = {}

    for sent in doc.sents:
        sent_weight = 0
        for token in sent:
            sent_weight += weighted_words.get(token.text, 0.0)
        sent_strength[sent] = sent_weight
    
    return sorted(sent_strength.items(), key=lambda tuple: tuple[1], reverse=True)[:n_sents]

def get_short_version(weighted_sentences): 
    if weighted_sentences is None: 
        raise ValueError("No sentences.")
    
    text = ""
    for (sentence, _) in weighted_sentences: 
        text += " " + sentence.text

    return text


nlp = spacy.load("en_core_web_md")
text = open_file()
doc = nlp(text)


# Call the functions here
""" for i in range(1, 60, 5): 
    words = most_used_words(doc, i)
    weighted_words = normalization(words)
    weighted_sents = weighted_sentences(doc, weighted_words, 3)
    short_version = get_short_version(weighted_sents)

    print("Considering " +str(i) + " words")
    print(short_version)
    print(most_used_words(doc, 100)) """