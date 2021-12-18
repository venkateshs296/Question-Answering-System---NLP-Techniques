from datetime import datetime
import sys
import glob
import os
import spacy

from itertools import chain
from nltk.corpus import stopwords

import json
import csv

nlp = spacy.load("en_core_web_sm")


import requests
from elasticsearch import Elasticsearch, helpers

import errno

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




files_list = glob.glob('articles\*.txt')
# question = "When was Arizona buy by Mexico?"
lemmatizer = WordNetLemmatizer()


max_cosine_val = -1
res_sentence = ""
sw = stopwords.words('english')


def get_dependency_parsing(sentence):
    dependency_parsed_tree =[]
    doc = nlp(sentence)
    sent = list(doc.sents)
    head_set = set()
    for s in sent:
        rootOfSentence = s.root.text
    for token in doc:
        dependency_parsed_tree.append([token.dep_,token.head.text,token.text])
        head_set.add(token.head.text)
    return dependency_parsed_tree, rootOfSentence, head_set


def find_word_tokens(sentence) :
    word_tokens = [word for word in word_tokenize(sentence) if not word in sw]
    return word_tokens


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



def cosine_sim(docids, sentences, question) :

    # remove stop words from the string
    outsentence = ""
    maxi = -1
    out_doc_id = ""
    for index, sentence in enumerate(sentences):
        X_list = word_tokenize(question)
        X_list = {w.lower() for w in X_list}
        X_set = {w for w in X_list if not w in sw} #query token
        Y_list = word_tokenize(sentence)
        Y_list = {w.lower() for w in Y_list}
        Y_set = {w for w in Y_list if not w in sw} #answer_tokens
        rvector = X_set.union(Y_set)
        l1 = [] #ques vect
        l2 = [] #ans vect
        for w in rvector:
            if w in X_set:
                l1.append(1)  # create a vector
            else:
                l1.append(0)
            if w in Y_set:

                l2.append(1)
            else:
                l2.append(0)
        c = 0

        # cosine formula
        for i in range(len(rvector)):
            c += l1[i] * l2[i]
        cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
        if cosine > maxi:
            maxi = cosine
            outsentence = sentence
            out_doc_id = docids[index]

    # print("max cosine is :" + str(maxi))
    # print("out sentence is :" + outsentence)

    return maxi, outsentence, out_doc_id


def is_who_ques(question) :
    return question.find("Who") != -1

def is_when_ques(question) :
    return question.find("When") != -1

def is_what_ques(question) :
    return question.find("What") != -1

def get_cosine_sim(docids, sentences, question) :

    # remove stop words from the string
    outsentence = ""
    maxi_cosine = -1
    out_doc_id = ""
    ques_dep_parsetree, ques_root, ques_heads = get_dependency_parsing(question)

    for index, sentence in enumerate(sentences):
        X_list = word_tokenize(question)
        X_list = {w.lower() for w in X_list}
        que_tokens = {w for w in X_list if not w in sw}  # query token
        Y_list = word_tokenize(sentence)
        Y_list = {w.lower() for w in Y_list}
        sent_tokens = {w for w in Y_list if not w in sw} #answer_tokens
        rvector = que_tokens.union(sent_tokens)

        sent_dep_parsetree, sent_root, sent_heads = get_dependency_parsing(sentence)
        sent_ner_list = get_named_entity_list(sentence)
        ques_vect = [] #ques vect
        sent_vect = [] #ans vect
        for w in rvector:
            if w in que_tokens:
                ques_vect.append(1)  # create a vector
            else:
                ques_vect.append(0)
            if w in sent_tokens:
               if w in sent_ner_list:
                   sent_vect.append(20)
               elif w in que_tokens:
                   sent_vect.append(10)
               else:
                   sent_vect.append(1)
            else:
                sent_vect.append(0)

        if lemmatizer.lemmatize(sent_root) == lemmatizer.lemmatize(ques_root) :
            ques_vect.append(1)
            sent_vect.append(10)
        elif sent_root == ques_root :
            ques_vect.append(1)
            sent_vect.append(10)

        if ques_root in sent_heads :
            ques_vect.append(1)
            sent_vect.append(5)

        que_root_synonyms = get_synonym_for_word(ques_root)
        for que_root_syn in que_root_synonyms :
            if que_root_syn in sent_heads :
                ques_vect.append(1)
                sent_vect.append(3)

        ner_map = {"Who" : "PERSON, ORG", "When" : "DATE, TIME"}
        ner_sent_labels = get_named_entity_labels(sentence)
        actual_labels = []
        if is_who_ques(question):
            actual_labels = ner_map["Who"]
        elif is_when_ques(question):
            actual_labels = ner_map["When"]
        # elif is_what_ques(question):
        #     actual_labels = ner_map["What"]


        if ner_sent_labels:
            for ner_label in ner_sent_labels :
                if ner_label in actual_labels :
                    ques_vect.append(1)
                    sent_vect.append(7)
                    break

        c = 0
        # cosine formula
        for i in range(len(rvector)):
            c += ques_vect[i] * sent_vect[i]
        cur_cosine = c / float((sum(ques_vect) * sum(sent_vect)) ** 0.5)
        if cur_cosine > maxi_cosine:
            maxi_cosine = cur_cosine
            outsentence = sentence
            out_doc_id = docids[index]

    # print("max cosine is :" + str(maxi))
    # print("out sentence is :" + outsentence)

    return maxi_cosine, outsentence, out_doc_id

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

# POS Tags for Words using NLTK POS Taggers
def pos_taggers(word_list):
    return nltk.pos_tag(word_list)


def parse_question() :
    question_word_tokens = find_word_tokens(question)
    # print(question_word_tokens)
    ques_pos_tagged = pos_taggers(question_word_tokens)
    wordnet_tagged = list(map(lambda x: (x[0], wordNet_pos_tagger(x[1])), ques_pos_tagged))
    # print(wordnet_tagged)
    synonyms = {}

    curr_word = question_word_tokens[2]
    syns = wn.synsets(curr_word)
    # print(syns[0].name())
    # print(syns[0].lemmas()[0].name())

    for word in question_word_tokens:
        syns = wn.synsets(word)
        temp_synonymns = []
        for synset in syns:
            for lemma in synset.lemmas():
                temp_synonymns.append(lemma.name())

        synonyms[word] = temp_synonymns


def get_synonym_for_word(word) :
    synonyms = []
    syns = wn.synsets(word)
    for synset in syns :
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return synonyms


def load_json(json_files):

    for file_name in json_files:
        actual_file_name = os.path.basename(file_name)
        # print("filename :" + actual_file_name)
        with open('json/' + actual_file_name, 'r') as open_file:
            doc_dict = json.load(open_file)
            doc_dict['_id'] = actual_file_name.split('.')[0]
            yield  doc_dict


def named_entity_recognition(sentence):
    ner = {}
    doc = nlp(sentence)
    for X in doc.ents:
        key_entities = ''.join(map(str, X.text))
        ner[X] = X.label_

    return ner

def get_named_entity_list(sentence):
    ner = set()
    doc = nlp(sentence)
    for X in doc.ents:
        ner.add(str(X))

    return ner

def get_named_entity_labels(sentence):
    ner_labels = set()
    doc = nlp(sentence)
    for X in doc.ents:
        ner_labels.add(X.label_)

    return ner_labels



def create_artciles_json() :
    for file_name in files_list :
        actual_file_name = os.path.basename(file_name)
        text_data = file_read("articles/" + actual_file_name)
        sentences = sent_tokenize(text_data)
        dict1 = {}
        ct = 1
        for sentence in sentences :
            dict1[ct] = sentence
            ct = ct + 1
        out_file_name = actual_file_name.replace('.txt', '')
        out_file = open("json/" + out_file_name + ".json", "w")
        json.dump(dict1, out_file, indent=2)
        out_file.close()

def do_index_elastic_search():
    json_files = glob.glob('json\*.json')
    res = helpers.bulk(es, load_json(json_files), index=index_name)


def index_elastic_search():
    for file_name in files_list:
        actual_file_name = os.path.basename(file_name)
        text_data = file_read("articles/" + actual_file_name)
        # print(words)
        sentences = sent_tokenize(text_data)
        dict1 = {}
        ct = 1
        for sentence in sentences:
            doc = {
                'text': sentence,
                'author': 'venky',
                'timestamp': datetime.now()
            }
            es.index(index=index_name, id= actual_file_name + "-" + str(ct), document=doc)

            dict1[ct] = sentence
            ct = ct + 1
        out_file_name = actual_file_name.replace('.txt', '')
        out_file = open("json/" + out_file_name + ".json", "w")
        json.dump(dict1, out_file, indent=2)
        out_file.close()



def retrieve_candidate_sentences_es(question):
    ques_word_tokens = word_tokenize(question)
    # filtered_ = {w for w in ques_word_tokens if not w in sw}  # query token
    print(ques_word_tokens)
    filtered_ques = ""
    question_types = ["Who", "When", "What", "?", "\'s"]
    ner_list = get_named_entity_list(question)
    for word in ques_word_tokens :
        if word not in sw and word not in question_types:
            if word in ner_list:
                filtered_ques += word
                filtered_ques += "^100"
            else:
                filtered_ques += word
            filtered_ques += " "

    # print(filtered_ques)
    search_query = {
            "query": {
               "query_string": {
                   "query": filtered_ques,
                    "fields" : [ ]
                }
             },
            "highlight" : {
                "fields" : {
                    "*" : {}
                }
            }
    }
    result = es.search(index = index_name, body = search_query)
    docs = result['hits']['hits']
    # print(docs)
    es_res_dict = {}
    es_res_docids = []
    es_res_sentences = []
    print("docs length:" + str(len(docs)))
    for doc in docs :
        print("doc id" + doc['_id'] )
        doc_id  = doc['_id']
        file_json = open("json/" + doc_id + ".json")
        doc_json = json.load(file_json)
        sent_highlights = doc['highlight']
        file_json.close()
        # print("keys length:")
        # print(str(len(sent_highlights.keys())))
        # print(sent_highlights.keys())
        for key in sent_highlights.keys() :
            sentence = doc_json[str(key)]
            es_res_sentences.append(sentence)
            es_res_docids.append(doc_id + ".txt")

    return es_res_docids, es_res_sentences






def get_questions(file_name):
    text_data = file_read(file_name)
    questions = sent_tokenize(text_data)
    return questions

def write_to_csv(file_name,questions, doc_ids, ans_sentences):
    file = csv.writer(open(file_name, "w", newline=""))
    file.writerow(["question", "document Id", "Answer Sentence"])
    for index, question in enumerate(questions):
        file.writerow([question, doc_ids[index], ans_sentences[index]])


def do_named_entity_recognition(question, doc_ids, sentences) :
    is_who_ques = False  # person
    is_what_ques = False  # org
    is_when_ques = False  # date

    if question.find("Who") != -1:
        is_who_ques = True
    elif question.find("When") != -1:
        is_when_ques = True
    else:
        is_what_ques = True

    filtered_docids = []
    filtered_sentences = []
    removed_sentences = []

    for index, sentence in enumerate(sentences):
        ner = named_entity_recognition(sentence)
        # print(ner)
        isPatternFound = False
        for key in ner:
            if is_who_ques:
                if ner[key] == 'PERSON':
                    isPatternFound = True
                    break
            elif is_when_ques:
                if ner[key] == 'DATE':
                    isPatternFound = True
                    break
            else:
                if ner[key] == 'ORG':
                    isPatternFound = True
                    break
        if isPatternFound:
            filtered_docids.append(doc_ids[index])
            filtered_sentences.append(sentence)
        else:
            removed_sentences.append(sentence)

    print("filtered sentences length:" + str(len(filtered_sentences)))
    print("removed sentences length:" + str(len(removed_sentences)))

    if len(filtered_sentences) == 0 :
        filtered_sentences.extend(sentences)
        filtered_docids.extend(doc_ids)

    return filtered_docids, filtered_sentences

res = requests.get('http://localhost:9200')
es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
index_name = 'nlpproj96'

create_artciles_json()
do_index_elastic_search()
questions = get_questions("questions.txt")
res_docids = []
res_sentences = []
print("question length" + str(len(questions)))
for question in questions:
    doc_ids, sentences = retrieve_candidate_sentences_es(question)
    print("es sentences length:" + str(len(sentences)))
    question = question.replace("\'s", "")
    max_cosine_val, res_sentence, res_doc_id = get_cosine_sim(doc_ids, sentences, question)
    print(res_sentence)
    print(res_doc_id)
    res_docids.append(res_doc_id)
    res_sentences.append(res_sentence)

print("questions len:" + str(len(questions)))
print("docIds len:" + str(len(res_docids)))
print("ans len:" + str(len(res_sentences)))

write_to_csv("qa-res.csv", questions, res_docids, res_sentences)








