import os
import sys

import requests
from flask import Flask, request

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import adam
import tensorflow as tf
import re
import json
import random

# load dataset from json file which is given in the folder
with open('data1.json',encoding='utf-8') as json_data:
    intents = json.load(json_data)
context = {}

words = []
classes = []
documents = []
ignore_words =['?','ተብሎ','የሚጠራ','ሲሆን','ይህም','ወይም','በሌሎች','ላይ','ነው','በ','የ','ከ','ሊሆኑ','ማለት','ይችላሉ','ወይ','ይችላል','በጣም','ከመሆን','የተነሳ','ወደ','ሙሉ','አይነት','ጨምሮ','የአንድ','አንድ','ግለሰብ','የሆነ','ሁኔታ','ውስጥ','ናቸው','ማለትም','የሚገኙ','በአደገኛ','አደገኛ','ወይንም','በአንድ','እንዲሁም','ሲሆን','በግዜው','ጨምሮ','ያሉ','ሰዎች','በአብዛኛውን','ሰው','ምንም','ሆኖ','ከሰው','ወደ','ከባድ','ምንድ','ምንድን','ምንድነው','ምንድናቸው','ነው','ናቸው','ስንል','ምን','ማለታችን','ማለት','አለው','የምን','የያዘው','እንዴት','የያዘውን','ሊሆን','የምን','በምን','የሚችል','ማን','በተለየ','በአብዛኛው','የቱን','የትኛው','እንዴት','አይነቶች','አለብን','ያለብን','ከመያዛችን','በፊት','የመያዝ','አጋጣሚን','አጋጣሚ','እንዳለብን','አሉት','በሰአቱ','በግዜው','ሁሉ','ሁሉም','ሆነ','ሆኖም','ሁሉንም','ማለት','ማን','ብቻ','ነገር','ነገሮች','ናቸው','አሁን','አለ','እስከ','እንኳን','እስከ','እዚሁ','እና','እንደ','ከ','ወዘተ','ወይም','ዋና','ይህ','ደግሞ','ጋራ','ግን','ጋር','ሆኖም','ማን','ለማን','ማነው','ማንማን','ማንን','ከማንኛው','ማንኛው','በማን','ጥቀስ','ግለፅ','ዘርዝር','ጥራ','ምን','ምንድን','የምን','ለምን','በምን','ወይ','ይሆን','እንደ','እንዴት']

def normalize(sentences):
    text=[i.replace('ኀ','ሀ').replace('ሐ','ሀ').replace('ሃ','ሀ').replace('ኃ','ሀ').replace('ሓ','ሀ').replace('ኁ','ሁ').replace('ሑ','ሁ').replace('ሒ','ሂ').replace('ኂ','ሂ').replace('ኄ','ሄ').replace('ሔ','ሄ').replace('ሕ','ህ').replace('ኅ','ህ').replace('ሖ','ሆ').replace('ኆ','ሆ').replace('ጸ','ፀ').replace('ጹ','ፁ').replace('ጺ','ፂ').replace('ጻ','ፃ').replace('ጼ','ፄ').replace('ጽ','ፅ').replace('ጾ','ፆ').replace('ቸ,','ቼ').replace('ሸ','ሼ').replace('ዬ','የ').replace('ዉ','ው').replace('ሓ','ሀ').replace('ሠ','ሰ').replace('ሡ','ሱ').replace('ሢ','ሲ').replace('ሣ','ሳ').replace('ሤ','ሴ').replace('ሥ','ስ').replace('ሦ','ሶ').replace('ዐ','አ').replace('ዑ','ኡ').replace('ዒ','ኢ').replace('ዓ','አ').replace('ኣ','አ').replace('ዔ','ኤ').replace('ዕ','እ').replace('ዖ','ኦ').replace('መካኒካል','ሜካኒካል').replace('ኢንጂነሪንግ','ምህንድስና').replace('ሰፍትዌር','ሶፍትዌር').replace('ሲስተም','ስይስተም').replace('ከሚካል','ኬሚካል').replace('ደምወዝ','ደሞዝ').replace('ዶሞዝ','ደሞዝ').replace('አርኪተክቸር','አርክተክቸር').replace('ኢሌክትሪካል','ኤሌክትሪካል').replace('ኮምፒተር','ኮምፕዩተር').replace('ሳይነስ','ሳይንስ').replace('ኢንፎርመሽን','ኢንፎርሜሽን').replace('ኢንዱስትርያል','ኢንዳስትሪያል').replace('ኢንዱስትሪያል','ኢንዳስትሪያል').replace('መሃንዲስ','መሀንድስ') for i in sentences]    
    return(text)

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words= normalize(words)
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
from keras.models import load_model
model=load_model('model_FAQ_best_wzz_english_ChatBot.h5')


def clean_up_sentence(sentence):
    #sentence_words= normalize(sentence)
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words if word not in ignore_words]
    sentence_words= normalize(sentence_words)
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    p = bow(sentence, words)
    
    d = len(p)
    f = len(documents)-2
    a = np.zeros([f, d])
    tot = np.vstack((p,a))
    
    results = model.predict(tot)[0]
    
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list
not_answerd=[]
from collections import Counter
import collections
def response(sentence, userID, show_details=False):
    results = classify(sentence)
    #print('Result:',results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return (random.choice(i['responses']))
            results.pop(0)
    else:
        not_answerd.append(sentence)
        c = Counter(not_answerd[1:])  # slice to ignore first value, Counter IS a dict already 
        # Just output counts > 3
        with open('listfile.txt', 'w', encoding='utf-8') as filehandle:
            for k, v in c.items():
                if v > 2:
                    filehandle.write('%s\n' % k)
                    #print(k)
        res="ይቅርታ አልገባኝም በደምብ አስተካክለህ ጠይቀኝ።"
        return res  
       #print (not_answerd)
app = Flask(__name__)


@app.route('/', methods=['GET'])
def verify():
    # when the endpoint is registered as a webhook, it must echo back
    # the 'hub.challenge' value it receives in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == os.environ["VERIFY_TOKEN"]:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200

    return "Hello world", 200


@app.route('/', methods=['POST'])
def webhook():

    # endpoint for processing incoming messaging events
    data = request.get_json()
    print(data)  # you may not want to log every incoming message in production, but it's good for testing

    if data["object"] == "page":

        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:

                if messaging_event.get("message"):  # someone sent us a message

                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID
                    message_text = messaging_event["message"]["text"]  # the message's text

                    responseai = response(message_text, sender_id)
                    send_message(sender_id, responseai)


                if messaging_event.get("delivery"):  # delivery confirmation
                    pass

                if messaging_event.get("optin"):  # optin confirmation
                    pass

                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                    pass

    return "ok", 200


def send_message(recipient_id, message_text):

    print("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {
        "access_token": os.environ["PAGE_ACCESS_TOKEN"]
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    })
    r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=data)
    if r.status_code != 200:
        print(r.status_code)
        print(r.text)


if __name__ == '__main__':
    app.run(debug=True)
