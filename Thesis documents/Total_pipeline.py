import os
import re
import json
import wandb
import pickle
import requests
import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1) # to sort conflict with Ludwig
tf.config.threading.set_inter_op_parallelism_threads(1)
print("Num_threads:", tf.config.threading.get_intra_op_parallelism_threads())# to sort conflict with Ludwig
print("Num_threads:", tf.config.threading.get_inter_op_parallelism_threads())# to sort conflict with Ludwig

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
cached_stopwords = stopwords.words("english")
new_stopwords=['a', 'll', 's', 'at', 'd', 't']
cached_stopwords.extend(new_stopwords)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import *
from ludwig.api import LudwigModel
from transformers import BertTokenizer

# from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import Paraphrase_module
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

prediction_threshold=0.5
from lime.lime_text import LimeTextExplainer
LIME_num_samples=1000
label_names = [0,1]  # The label whose explanation is needed
explainer = LimeTextExplainer(class_names=label_names)


os.environ["WANDB_API_KEY"] = "980f6fd5d9bd3d0adbd7b9a244e7b151326bd44c"
wandb.init(project='Total_pipeline', entity='cheriyan', config = {"L1_RO_model": "bert-base-uncased",
                     "L2_RO_model": "bert-base-cased",
                      "L1_num_labels": 1,
                      "L2_num_labels": 3,
                      "batch_size": 16,
                      "add_special_tokens": True,
                      "max_length": 250,
                      "padding": True,
                      "truncation": True,
                      "return_attention_mask": True,
                      "random_seed": 42})
config=wandb.config

def comment_preprocess(comment):
    comment = re.sub(r"[^a-zA-Z]+", ' ', str(comment))
    # Comment = str(Comment).lower()
    comment = ' '.join([word for word in str(comment).split() if word not in cached_stopwords])

    # Lemmatization
    comment = comment.split()
    comment = [lemmatizer.lemmatize(word) for word in comment]
    comment = ' '.join(comment)
    return comment

def regex_check(comment):
    regex_presence = 0
    regex_df = pd.read_csv("H:\\SO Norms Work\\Regex_Updated.csv")
    regexes = regex_df["Regex_Text"]
    for regex in regexes:
        pattern = re.compile(regex)
        if pattern.search(comment):
            print(pattern)
            regex_presence = 1
            break
    print("Regex status: ", regex_presence)
    return regex_presence

def PAPI_check(comment):
    papi_api_key = 'AIzaSyAAhtjSbNwJUFh32JAxtiydqiye87kyfqM'
    PAPI_url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' + '?key=' + papi_api_key)
    data_dict = {'comment':
                {'text': comment},
                 'languages': ['en'],
                 'doNotStore': True,
                 'requestedAttributes': {'TOXICITY': {}}
                 }
    response = requests.post(url=PAPI_url, data=json.dumps(data_dict))
    response_dict = json.loads(response.content)
    PAPI_score = response_dict["attributeScores"]["TOXICITY"]['summaryScore']['value']
    print("PAPI score: ", PAPI_score)
    return PAPI_score

def regex_and_PAPI_check(comment):      # this function returns TRue if both Regex is prsent and PAPI score>=0.7, else False
    return 1 if(regex_check(comment) and PAPI_check(comment)>=0.7) else 0

def get_politeness_features(comment):    # this function accepts the comment and returns a DF of features
    r = robjects.r  # this block finds the politeness features of a comment, makes is DF
    r['source']('Politeness trial.R')  # C:\\Users\\cheji902\Dropbox\\pycharam works\\Info_407_2020\\
    politeness_function_r = robjects.globalenv[
        'politeness_function']  # 'politeness_function' is the function defined in R source code
    df_polite_r = politeness_function_r(comment)
    df_polite = pd.DataFrame(pandas2ri.py2rpy(df_polite_r)).transpose()
    return df_polite

def level_1_RO_BERT_prediction(comment):
    bert_tokenizer = BertTokenizer.from_pretrained(config.L1_RO_model)
    trained_model = TFBertForSequenceClassification.from_pretrained(config.L1_RO_model,
                                                                    num_labels=config.L1_num_labels)
    trained_model.load_weights('C:\\StackOverflow_Work\\ML models\\L1 Transformer best model\\Transformer_BERT_Model.h5')
    test_input = bert_tokenizer.batch_encode_plus(comment,
                                                  add_special_tokens=config.add_special_tokens,
                                                  max_length=config.max_length,
                                                  padding=config.padding,
                                                  truncation=config.truncation,
                                                  return_attention_mask=config.return_attention_mask)

    val_inp = np.asarray(test_input['input_ids'])
    val_mask = np.asarray(test_input['attention_mask'])
    logits = np.array(trained_model.predict([val_inp, val_mask], batch_size=config.batch_size)).squeeze(axis=0)
    pred_prob = tf.nn.sigmoid(logits).numpy()       # adding a sigmoid layer to logits to get pred_probabilities
    print(pred_prob)
    # pred_label = 0 if pred_prob[0][0] <= prediction_threshold else 1
    pred_label = np.where(pred_prob <= prediction_threshold, 0, 1)
    print("Level 1 RO prediction:", pred_label[0][0])
    return pred_label

def level_1_UN_prediction(comment):     # this function accepts the comment, gets politeness, sentimental and tf-idf features
    df_polite = get_politeness_features(comment)

    # this block loads the vectrizer, and converts the comment to tf-idf vector, then concatenates with the politeness features
    vectorizer = pickle.load(open("C:\\StackOverflow_Work\\ML models\\L1 UN ML models\\"
                                  "Unwelcoming best models\\L1_ML_Unwelcoming_V1_Vectorizer.pkl", 'rb'))
    vect = vectorizer.transform([comment_preprocess(comment)])
    vect = pd.DataFrame(vect.toarray(), columns=vectorizer.get_feature_names())
    vect = pd.concat([df_polite, vect], axis=1)

    analyser = SentimentIntensityAnalyzer()      # this block is to get the sentiment score and make it a DF, combine to previous feature vector
    score = analyser.polarity_scores(comment)
    key = 1 if score["compound"] > 0.5 else 2 if score["compound"] < -0.5 else 0
    senti_df = pd.DataFrame({"Sentiment": [score["compound"]],
                             "Sentiment_Score": [key]})
    vect = pd.concat([senti_df, vect], axis=1)

    SVM = pickle.load(open("C:\\StackOverflow_Work\\ML models\\L1 UN ML models\\"
                           "Unwelcoming best models\\L1_SVM_Unwelcoming_V1_Classifier.pkl", 'rb'))
    SVM_precict=SVM.predict(vect)

    print("\n Level 1 UN prediction :", SVM_precict )
    return SVM_precict

def level_2_RO_prediction(comment):
    bert_tokenizer = BertTokenizer.from_pretrained(config.L2_RO_model)
    trained_model = TFBertForSequenceClassification.from_pretrained(config.L2_RO_model,
                                                                    num_labels=config.L2_num_labels)
    trained_model.load_weights("C:\\StackOverflow_Work\\ML models\\L2_RO_Transformer_model (Best model) 31.7.2021 Trimmed_Cased"
                               "\\L2_Transformer_BERT_Cased_Model_V9_RO.h5")
    test_input = bert_tokenizer.batch_encode_plus(comment,
                                                  add_special_tokens=config.add_special_tokens,
                                                  max_length=config.max_length,
                                                  padding=config.padding,
                                                  truncation=config.truncation,
                                                  return_attention_mask=config.return_attention_mask)

    val_inp = np.asarray(test_input['input_ids'])
    val_mask = np.asarray(test_input['attention_mask'])

    # code  for 1 label prediction      Use sigmoid for binary prediction
    logits = np.array(trained_model.predict([val_inp, val_mask])).squeeze(axis=0)  # code for single label prediction
    pred_prob = tf.nn.sigmoid(logits).numpy()
    pred_labels = np.where(pred_prob < prediction_threshold, 0, 1) # for multi label prediction
    # print(pred_labels)
    return pred_labels[0]

def level_2_RO_TASA_prediction(comment):
    df_polite=get_politeness_features(comment)

    # this block loads the vectrizer, and converts the comment to tf-idf vector, then concatenates with the politeness features
    vectorizer = pickle.load(open("C:\\StackOverflow_Work\\ML models\\L2 RO TASA Best model\\L2_ML_RO_TASA_Vectorizer.pkl", 'rb'))
    vect = vectorizer.transform([comment])
    vect = pd.DataFrame(vect.toarray(), columns=vectorizer.get_feature_names())
    vect = pd.concat([df_polite, vect], axis=1)

    SVM = pickle.load(open("C:\\StackOverflow_Work\\ML models\\L2 RO TASA Best model\\L2_SVM_RO_TASA_Classifier.pkl", 'rb'))
    SVM_precict = SVM.predict(vect)

    # print("Level 2 RO TASA :", SVM_precict)
    return SVM_precict

def level_2_UN_prediction(comment):
    df_polite=get_politeness_features(comment)

    # this block loads the vectrizer, and converts the comment to tf-idf vector, then concatenates with the politeness features
    vectorizer = pickle.load(open("C:\\StackOverflow_Work\\ML models\\L2_UN_ML_models\\"
                                  "\\L2_UN_Vectorizer.pkl", 'rb'))
    vect = vectorizer.transform([comment_preprocess(comment)])
    vect = pd.DataFrame(vect.toarray(), columns=vectorizer.get_feature_names())
    vect = pd.concat([df_polite, vect], axis=1)

    SVM = pickle.load(open("C:\\StackOverflow_Work\\ML models\\L2_UN_ML_models\\"
                           "\\L2_UN_SVM_Classifier.pkl", 'rb'))
    SVM_precict = SVM.predict(vect)
    # print("Level 2 UN :", SVM_precict)
    return SVM_precict

def level_3_prediction(comment): 

    comment = {"Comment_Text": comment, "Racial": '', "Swearing": ''}
    Df = pd.DataFrame(columns=["Comment_Text", "Racial", "Swearing"])
    Df = Df.append(comment, ignore_index=True)

    ludwig_model = LudwigModel.load("C:\\StackOverflow_Work\\ML models\\L3_Ludwig_Models\\Old best model\\model",
                                    allow_parallel_threads=False)  # Already trained model
    predictions, _ = ludwig_model.predict(dataset=Df)

    return (predictions['Racial_predictions'][0],
            predictions['Swearing_predictions'][0])

def level_1_UN_Lime_explanation(comment):
    hot_words=[]
    vectorizer = pickle.load(open("H:\\L1_UN_Lime_Vectorizer.pkl", 'rb'))
    SVM = pickle.load(open("H:\\L1_UN_Lime_SVM_Classifier.pkl", 'rb'))
    SVM_ml_pipe = make_pipeline(vectorizer, SVM)

    ml_exp = explainer.explain_instance(comment_preprocess(comment),
                                        SVM_ml_pipe.predict_proba,
                                        num_features=5,
                                        labels=(1,)
                                        )

    explanation_raw= dict(sorted(ml_exp.as_list(), key=lambda x: x[1], reverse=True))
    scaler = MinMaxScaler()  # min-max scaler to normalize Lime explanation values, default scale (0,1)
    explanation_normalized = scaler.fit_transform(np.array(list(explanation_raw.values())).reshape(len(explanation_raw.values()), 1))
    explanation_normalized = explanation_normalized.reshape(len(explanation_normalized), ).tolist()
    explanation_list = list(explanation_raw.items())

    for score in explanation_normalized:  # Locate the words to be masked/Unwelcoming
        if score >= 0.5:
            index = explanation_normalized.index(score)
            hot_words.append(explanation_list[index][0])
    print(hot_words)
    return hot_words

def SE_related(comment):        # this module checks the SE relevance of a comment before paraphrasing
    se_count = 0
    with open('H:\\SE_Thesaurus_New_v2.json') as json_file:     # The latest SE relation file, after discarding SO terms
        se_thesaurus = json.load(json_file)  # this outputs as a dictionary
    se_list = [item[0] for item in se_thesaurus.items()]

    comment = comment_preprocess(comment.lower())
    for word in comment.split():
        if word in se_list:
            se_count += 1
            break

    return se_count


'''     This block is part of the old implementation of paraphrasing, not needed in the actual pipeline
def find_regex(Comment):
    Highlight_List = []
    Regex_Data=pd.read_csv("H:\\Regex_Updated.csv")
    Regexes=Regex_Data["Regex_Text"]

    for Regex in Regexes:
        p=re.compile(Regex)
        for m in p.finditer(Comment):
            #print(m.start(),m.end(),m.group(),Regex,Regex_Data.loc[Regex_Data["Regex_Text"]==Regex,"Regex_Class"].iloc[0])
            Dict={"Start" : m.start(),
                  "End" : m.end(),
                  "Class": Regex_Data.loc[Regex_Data["Regex_Text"]==Regex,"Regex_Class"].iloc[0]}
            Highlight_List.append(Dict)

    #print(Highlight_List)
    return Highlight_List

def Comment_Mask_Paraphraser(Comment):
    from transformers import pipeline
    unmasker = pipeline('fill-mask', model='bert-base-uncased')
    p = re.compile(r'[\w!]+')
    for location in find_regex(Comment):
        Sub_String=Comment[location['Start']:location['End']]
        for i in p.finditer(Comment):
            if Sub_String in i.group():
                Comment=Comment[:i.start()]+'[MASK]'+Comment[i.end():]      #Returns MASKed answer
                print(Comment)
                # Results=unmasker(Comment)               # Returns paraphrases instead of MASK
                # Comment=Results[0]['sequence'][6:-6]    # to trim extra characters added as part of masking
                #Comment = Comment[6:-6]
                break
    #print(Comment)
    return (Comment)
    
'''

def analyze(comment):
    # Actual pipeline
    if (level_1_RO_BERT_prediction([comment])):     # RO pipeline
        if(regex_and_PAPI_check(comment)):      # if yes, go to Levels 2 and 3, then paraphrase
            print("Level 1 RO = 1")
            l2_RO_outcomes=level_2_RO_prediction(comment)
            print("Individually targetted: ", l2_RO_outcomes[0])
            print("Group targetted: ", l2_RO_outcomes[1])
            print("Others: ", l2_RO_outcomes[2])
            l2_TASA_outcomes=level_2_RO_TASA_prediction(comment)
            print("Technology abuse: ", l2_TASA_outcomes[0,0])
            print("Self-abuse: ", l2_TASA_outcomes[0,1])
            l3_outcomes=level_3_prediction(comment)
            print("Racial abuse:", l3_outcomes[0])
            print("Swearing abuse:", l3_outcomes[1])
            if SE_related(comment):
                Paraphrase_module.paraphrase(comment)               # Use the paraphrasing pipeline for SE related commments
            else:
                Paraphrase_module.seq2seq_paraphraser(comment)      # use seq2seq model for non-SE related comments

        elif(level_1_UN_prediction(comment)):   # UN pipeline; if RO=1 but no regex and PAPI score, so can be potential UN
            l2_UN_outcomes=level_2_UN_prediction(comment)
            print("Individually targetted: ", l2_UN_outcomes[0, 0])
            print("Group targetted: ", l2_UN_outcomes[0, 1])
            print("Technology abuse: ", l2_UN_outcomes[0, 2])
            print("Others: ", l2_UN_outcomes[0, 3])
            l3_outcomes = level_3_prediction(comment)
            print("Racial abuse:", l3_outcomes[0])
            print("Swearing abuse:", l3_outcomes[1])
            print(level_1_UN_Lime_explanation(comment))
        else:
            print("No violations detected")
    elif (level_1_UN_prediction(comment)):      # if not Level 1 RO, UN check
        l2_UN_outcomes = level_2_UN_prediction(comment)
        print("Individually targetted: ", l2_UN_outcomes[0, 0])
        print("Group targetted: ", l2_UN_outcomes[0, 1])
        print("Technology abuse: ", l2_UN_outcomes[0, 2])
        print("Others: ", l2_UN_outcomes[0, 3])
        l3_outcomes = level_3_prediction(comment)
        print("Racial abuse:", l3_outcomes[0])
        print("Swearing abuse:", l3_outcomes[1])
        print(level_1_UN_Lime_explanation(comment))
    else:
        print("No violations detected")


if __name__=='__main__':
    comment="Since it's Thanksgiving I'll thank you for bothering to read my question. Your answer on the other hand is worthless."
    analyze(comment)


