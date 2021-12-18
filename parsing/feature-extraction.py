import sys
import glob
import os
import spacy
from itertools import chain
from nltk.corpus import stopwords

import errno

nlp = spacy.load("en_core_web_sm")

# NLTK Imports
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('stopwords')

try:
   nltk.data.find('tokenizers/punkt')
except:
   nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except:
    nltk.download('averaged_perceptron_tagger')

try:
   nltk.data.find('corpora/wordnet')
   from nltk.corpus import wordnet as wn

except:
   nltk.download('wordnet')


from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


all_stopwords = nlp.Defaults.stop_words

# take out the 'not' stopword from the default list as not is an important stopword that can alter meaning if not included
all_stopwords.remove('not')

# Wordnet Lematizer Initialization
lemmatizer = WordNetLemmatizer()

# Loading Spacy English Model
# nlp = spacy.load('en_core_web_sm')

files_list = glob.glob('articles\*.txt')



def file_read(text):
    f = open(text, encoding="ascii", errors="ignore")
    lines = f.readlines()
    sentences = ''
    for line in lines:
        if line.find('  ') == 0 or '\t' in line:
            sentences = sentences + line.strip() + "\n"
        else:
            sentences = sentences + line

    f.close()

    return sentences



# Words Extracted from a Sentence
def word_tokenizer(sentence):
    return word_tokenize(sentence)

# POS Tags for Words using NLTK POS Taggers
def pos_taggers(word_list):
    return nltk.pos_tag(word_list)

def dependency_parsing(sentence):
    dependency_parsed_tree =[]
    doc = nlp(sentence)
    sent = list(doc.sents)
    for s in sent:
        rootOfSentence = s.root.text
    for token in doc:
        dependency_parsed_tree.append([token.dep_,token.head.text,token.text])
    return dependency_parsed_tree


def wordnet_features(words):
    # Initializtion of the Wordnet Features Dictionaries
    synonymns = {}
    hypernyms = {}
    hyponyms = {}
    meronyms = {}
    holonyms = {}

    # Looping through Words
    for word in words:
        # Initialization for temporary Lists for each Feature
        temp_synonymns = []
        temp_hypernyms = []
        temp_hyponyms = []
        temp_meronyms = []
        temp_holonyms = []

        # Synsets for the Word (WordNet)
        for i, j in enumerate(wn.synsets(word)):
            # Adding the synonymns to the List
            temp_synonymns.extend(wn.synset(j.name()).lemma_names())

            # Adding the hypernymns to the List
            temp_hypernyms.extend(list(chain(*[l.lemma_names() for l in j.hypernyms()])))

            # Adding the hyponymns to the List
            temp_hyponyms.extend(list(chain(*[l.lemma_names() for l in j.hyponyms()])))

            # Adding the meronymns to the List
            temp_meronyms.extend(list(chain(*[l.lemma_names() for l in j.part_meronyms()])))

            # Adding the holonymns to the List
            temp_holonyms.extend(list(chain(*[l.lemma_names() for l in j.part_holonyms()])))

        # Adding to the Dictionary
        synonymns[word] = temp_synonymns
        hypernyms[word] = temp_hypernyms
        hyponyms[word] = temp_hyponyms
        meronyms[word] = temp_meronyms
        holonyms[word] = temp_holonyms

    return synonymns, hypernyms, hyponyms, meronyms, holonyms

def word_stemmatization(words):
    stemmatize_word = {}
    ps = PorterStemmer()
    for word in words:
        stemmatize_word[word] = ps.stem(word)
    return stemmatize_word


def lemmatization(word_tokens):
    # Initializtion of Lemmas based on Word Tokens (NLTK)
    lemmas = {}

    # Looping through the Word Tokens
    for word in word_tokens:
        # Lemmatize the Word
        lemmas[word] = lemmatizer.lemmatize(word)

    return lemmas


def wordNet_pos_tagger(nltk_tag):
    # POS Tag start with J - Replace with ADJ
    if nltk_tag.startswith('J'):
        return wn.ADJ

    # POS Tag start with V - Replace with VERB
    elif nltk_tag.startswith('V'):
        return wn.VERB

    # POS Tag start with N - Replace with NOUN
    elif nltk_tag.startswith('N'):
        return wn.NOUN

    # POS Tag start with R - Replace with ADV
    elif nltk_tag.startswith('R'):
        return wn.ADV

    # Replace tag with None if no match
    else:
        return None


def lemmatization_wordnet(wordnet_tagged):
    # Initializtion of Lemmas basd on Wordnet Tagged Words
    lemmas_wordnet = {}

    # Looping through the Wordnet Tagged Words
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmas_wordnet[word] = []
        else:
            # else use the tag to lemmatize the token
            lemmas_wordnet[word] = lemmatizer.lemmatize(word, tag)

    return lemmas_wordnet


def named_entity_recognition(sentence):
    ner = {}
    doc = nlp(sentence)
    for X in doc.ents:
        key_entities = ''.join(map(str, X.text))
        ner[X] = X.label_

    return ner

def feature_extraction(file_name):
    words_list = []
    pos_tag_list = []
    wordnet_tagged_list = []
    stemmas_list = []
    lemmas_list = []
    lemmas_wordnet_list = []
    synonymns_list = []
    hypernyms_list = []
    hyponyms_list = []
    meronyms_list = []
    holonyms_list = []
    dependency_parse_tree_list = []
    ners_list = []
    base = os.path.basename(file_name)
    text_data = file_read(file_name)
    words =  word_tokenize(text_data)
    # print(words)
    sentences = sent_tokenize(text_data)

    for index,sentence in enumerate(sentences):
        word_tokens = [word for word in word_tokenizer(sentence) if not word in all_stopwords]
        d_parse = dependency_parsing(sentence)
        # Wordnet Features - Synonymns, Hypernymns, Hyponymns, Meronymns, Holonymns
        syn, hyper, hypo, mero, holo = wordnet_features(word_tokens)
        stemmas = word_stemmatization(word_tokens)
        lemmas = lemmatization(word_tokens)
        # Attach POS Tags to the Word List
        pos_tagged = pos_taggers(word_tokens)
        wordnet_tagged = list(map(lambda x: (x[0], wordNet_pos_tagger(x[1])), pos_tagged))
        lemmas_wordnet = lemmatization_wordnet(wordnet_tagged)
        ner = named_entity_recognition(sentence)

        # Attach POS Tags to the Word List
        pos_tagged = pos_taggers(word_tokens)
        # Appending the Features to Individual Lists
        words_list.append(word_tokens)
        pos_tag_list.append(pos_tagged)
        wordnet_tagged_list.append(wordnet_tagged)
        stemmas_list.append(stemmas)
        lemmas_list.append(lemmas)
        lemmas_wordnet_list.append(lemmas_wordnet)
        synonymns_list.append(syn)
        hypernyms_list.append(hyper)
        hyponyms_list.append(hypo)
        meronyms_list.append(mero)
        holonyms_list.append(holo)
        dependency_parse_tree_list.append(d_parse)
        ners_list.append(ner)

    # print(lemmas_list)

    output_file_name = os.path.splitext(base)[0]

    # Create the Features Folder with TextFile Folder
    try:
        os.makedirs('Features/' + output_file_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Printing to Files
    with open('Features/' + output_file_name + '/' + output_file_name + '_Word_Tokens.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in words_list)
    with open('Features/' + output_file_name + '/' + output_file_name + '_POS_Tags.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in pos_tag_list)
    with open('Features/' + output_file_name + '/' + output_file_name + '_Stemmas.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in stemmas_list)
    with open('Features/' + output_file_name + '/' + output_file_name + '_Lemmas.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in lemmas_list)
    with open('Features/' + output_file_name + '/' + output_file_name + '_Lemmas_WordNet.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in lemmas_wordnet_list)
    with open('Features/' + output_file_name + '/' + output_file_name + '_Dependency_Parse_Tree.txt',
              'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in dependency_parse_tree_list)
    with open('Features/' + output_file_name + '/' + output_file_name + '_synonyms.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in synonymns_list)
    with open('Features/' + output_file_name + '/' + output_file_name + '_hypernymns.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in hypernyms_list)
    with open('Features/' + output_file_name + '/' + output_file_name + '_hyponymns.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in hyponyms_list)
    with open('Features/' + output_file_name + '/' + output_file_name + '_meronymns.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in meronyms_list)
    with open('Features/' + output_file_name + '/' + output_file_name + '_holonymns.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in holonyms_list)
    with open('Features/' + output_file_name + '/' + output_file_name + '_NER.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in ners_list)


for file_name in files_list:
    feature_extraction(file_name)
