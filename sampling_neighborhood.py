import pandas as pd
from transformers import *
model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")
df=pd.read_csv('IMDB-Dataset.csv')
reviews=[]
sentiments=[]
for i in range(len(df)):
    reviews.append(df['review'][i])
    sentiments.append(df['sentiment'][i])

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
  # tokenize the text to be form of a list of token IDs
  inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
  # generate the paraphrased sentences
  outputs = model.generate(
    **inputs,
    num_beams=num_beams,
    num_return_sequences=num_return_sequences,
  )
  # decode the generated sentences using the tokenizer to get them back to text
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def return_paraphrases(reviews):
    paraphrased_reviews={}
    for review in reviews:
        sentence_list=get_paraphrased_sentences(model, tokenizer, review, num_beams=5, num_return_sequences=5)
        paraphrased_reviews[review]=sentence_list
        print(review, sentence_list)
    return paraphrased_reviews


reviews=reviews[0:10]
paraphrased_reviews=return_paraphrases(reviews)