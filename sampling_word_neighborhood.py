import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree
import nltk
from transformers import pipeline
pipeline_model = pipeline('fill-mask', model='bert-base-uncased')
# from transformers import *
# model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
# tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")
df=pd.read_csv('IMDB-Dataset.csv')
reviews=[]
sentiments=[]
for i in range(len(df)):
    reviews.append(df['review'][i])
    sentiments.append(df['sentiment'][i])

reviews=reviews[0:10]


def generate_masked_and_verb_perturbations(texts):
    replacements={}
    for i in range(len(texts)):
        replacements[texts[i]]=[]

    for i in range(len(texts)):
        pos_tagged = nltk.pos_tag(texts[i].split())
        verbs = []
        for item in pos_tagged:
            if('VB' in item[1]):
                verbs.append(item[0])
        for v in verbs:
            masked_sentence=texts[i].replace(v, "[MASK]")
            pred = pipeline_model(masked_sentence)
            replacements[texts[i]].append(pred[0]['sequence'])
    return replacements

def generate_masked_and_nouns_perturbations(texts):
    replacements={}
    for i in range(len(texts)):
        replacements[texts[i]]=[]

    for i in range(len(texts)):
        pos_tagged = nltk.pos_tag(texts[i].split())
        nouns = []
        for item in pos_tagged:
            if('NN' in item[1]):
                nouns.append(item[0])
        for v in nouns:
            masked_sentence=texts[i].replace(v, "[MASK]")
            pred = pipeline_model(masked_sentence)
            replacements[texts[i]].append(pred[0]['sequence'])
    return replacements



def generate_masked_and_adjective_perturbations(texts):
    replacements={}
    for i in range(len(texts)):
        replacements[texts[i]]=[]

    for i in range(len(texts)):
        pos_tagged = nltk.pos_tag(texts[i].split())
        nouns = []
        for item in pos_tagged:
            if('JJ' in item[1]):
                nouns.append(item[0])
        for v in nouns:
            masked_sentence=texts[i].replace(v, "[MASK]")
            pred = pipeline_model(masked_sentence)
            replacements[texts[i]].append(pred[0]['sequence'])
    return replacements

v_replacements=generate_masked_and_verb_perturbations(reviews)
n_replacements=generate_masked_and_nouns_perturbations(reviews)
adj_replacements=generate_masked_and_adjective_perturbations(reviews)