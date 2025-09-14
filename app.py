import flask
from flask import Flask, request, redirect, url_for, render_template, session
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from collections import Counter
import operator
import requests
from bs4 import BeautifulSoup
import pickle
import openpyxl
from Treatment import diseaseDetail
from statistics import mean

warnings.simplefilter("ignore")

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'super secret key'

def synonyms(term):
    synonyms = []
    response = requests.get(f'https://www.thesaurus.com/browse/{term}')
    soup = BeautifulSoup(response.content, "html.parser")
    try:
        container = soup.find('section', {'class': 'MainContentContainer'})
        row = container.find('div', {'class': 'css-191l5o0-ClassicContentCard'}).find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        pass
    for syn in wordnet.synsets(term):
        synonyms += syn.lemma_names()
    return set(synonyms)

def similarity(dataset_symptoms, user_symptoms):
    found_symptoms = set()
    for data_sym in dataset_symptoms:
        data_sym_split = data_sym.split()
        for user_sym in user_symptoms:
            count = sum(1 for symp in data_sym_split if symp in user_sym.split())
            if count / len(data_sym_split) > 0.5:
                found_symptoms.add(data_sym)
    return list(found_symptoms)

def preprocess(user_symptoms):
    df_comb = pd.read_csv(r"C:\Projects\MediCompanion-AI\Dataset\dis_sym_dataset_comb.csv")
    df_norm = pd.read_csv(r"C:\Projects\MediCompanion-AI\Dataset\dis_sym_dataset_norm.csv")
    dataset_symptoms = list(df_comb.columns[1:])
    lemmatizer = WordNetLemmatizer()
    splitter = RegexpTokenizer(r'\w+')
    processed_user_symptoms = [' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym.strip().replace('-', ' ').replace("'", ''))]) for sym in user_symptoms]
    
    user_symptoms = []
    for user_sym in processed_user_symptoms:
        user_sym_words = user_sym.split()
        str_sym = set()
        for comb in range(1, len(user_sym_words) + 1):
            for subset in combinations(user_sym_words, comb):
                str_sym.update(synonyms(' '.join(subset)))
        str_sym.add(' '.join(user_sym_words))
        user_symptoms.append(' '.join(str_sym).replace('_', ' '))
    return user_symptoms

@app.route("/")
@app.route("/index", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    df_comb = pd.read_csv(r"C:\Projects\MediCompanion-AI\Dataset\dis_sym_dataset_comb.csv")
    df_norm = pd.read_csv(r"C:\Projects\MediCompanion-AI\Dataset\dis_sym_dataset_norm.csv")
    dataset_symptoms = list(df_comb.columns[1:])
    
    user_symptoms = request.form.get('symptoms', '').split(',')
    user_symptoms = preprocess(user_symptoms)
    found_symptoms = similarity(dataset_symptoms, user_symptoms)

    select_list = list(range(len(found_symptoms)))
    dis_list = set()
    final_symp = [found_symptoms[i] for i in select_list]
    
    for symp in final_symp:
        dis_list.update(set(df_norm[df_norm[symp] == 1]['label_dis']))

    counter_list = []
    for dis in dis_list:
        row = df_norm[df_norm['label_dis'] == dis].values.tolist()
        if row:
            row[0].pop(0)
            for idx, val in enumerate(row[0]):
                if val != 0 and dataset_symptoms[idx] not in final_symp:
                    counter_list.append(dataset_symptoms[idx])
    
    dict_symp = dict(Counter(counter_list))
    dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1), reverse=True)
    another_symptoms = [tup[0] for tup in dict_symp_tup]

    session['my_var'] = another_symptoms
    session['my_var2'] = final_symp
    session['count'] = len(dict_symp_tup)
    session['tup'] = dict_symp_tup[0] if dict_symp_tup else None

    return render_template("predict.html", found_symptoms=enumerate(found_symptoms), another_symptoms=enumerate(another_symptoms), count=len(dict_symp_tup), dict_symp_tup=len(dict_symp_tup))

@app.route("/next", methods=["POST", "GET"])
def next():
    my_var2 = session.get('my_var2', [])
    final_symptoms = request.form.get('relevance', '').split(',')
    df_comb = pd.read_csv(r"C:\Projects\MediCompanion-AI\Dataset\dis_sym_dataset_comb.csv")
    dataset_symptoms = list(df_comb.columns[1:])
    
    sample_x = [0] * len(dataset_symptoms)
    for sym in final_symptoms:
        my_var2.append(sym)
        if sym in dataset_symptoms:
            sample_x[dataset_symptoms.index(sym)] = 1

    session['sample_x'] = sample_x
    session['my_var2'] = my_var2

    return render_template("next.html", my_var2=enumerate(my_var2))

@app.route("/final", methods=["POST", "GET"])
def final():
    sample_x = session.get('sample_x', [])
    my_var2 = session.get('my_var2', [])

    df_comb = pd.read_csv(r"C:\Projects\MediCompanion-AI\Dataset\dis_sym_dataset_comb.csv")
    df_norm = pd.read_csv(r"C:\Projects\MediCompanion-AI\Dataset\dis_sym_dataset_norm.csv")
    X = df_comb.iloc[:, 1:]
    Y = df_comb.iloc[:, 0:1]
    dataset_symptoms = list(X.columns)

    my_model = pickle.load(open(r"C:\Projects\MediCompanion-AI\model_saved", 'rb'))
    output = my_model.predict_proba([sample_x])
    scores = cross_val_score(my_model, X, Y, cv=10)

    diseases = sorted(set(Y['label_dis']))
    topk = output[0].argsort()[-5:][::-1]

    topk_dict = {}
    for t in topk:
        row = df_norm[df_norm['label_dis'] == diseases[t]].values.tolist()
        match_sym = {dataset_symptoms[idx] for idx, val in enumerate(row[0][1:]) if val != 0} if row else set()
        prob = (len(match_sym.intersection(set(my_var2))) + 1) / (len(set(my_var2)) + 1)
        prob *= mean(scores)
        topk_dict[t] = prob

    topk_sorted = sorted(topk_dict.items(), key=lambda x: x[1], reverse=True)
    arr = [f'Disease name: {diseases[key]}' for key, _ in topk_sorted]

    return render_template("final.html", arr=arr)

@app.route("/treatment", methods=["POST", "GET"])
def treatment():
    treat_dis = request.form.get('dis', 'False')
    workbook = openpyxl.load_workbook(r"C:\Projects\MediCompanion-AI\cure_minor.xlsx")
    worksheet = workbook['Sheet1']
    ans = []

    for row in worksheet.iter_rows(values_only=True):
        if treat_dis in row:
            stri = ''.join(map(str, row[1:]))
            ans = stri.split(',')
            break
    return render_template("treatment.html", ans=ans)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
