import os
import re
import math
import json
import torch
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import *
from operator import itemgetter
from transformers import pipeline
from transformers import BertTokenizer
from sklearn.preprocessing import MinMaxScaler
from transformers import logging
logging.set_verbosity_warning()
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


num_features=5   # Lime attributes
label_names = [0,1]  # The labels whose explanation is needed
LIME_num_samples=500    # The number of training instances LIME takes to create the perturbation
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=label_names)

from transformers import logging as hf_logging
hf_logging.set_verbosity_warning()
from transformers import RobertaTokenizer, RobertaForMaskedLM
roberta_tokenizer=RobertaTokenizer.from_pretrained('roberta-base')
fluency_roberta_model=RobertaForMaskedLM.from_pretrained('roberta-base')

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")     # For L1 RO model
L1_RO_trained_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
L1_RO_trained_model.load_weights('C:\\StackOverflow_Work\\ML models\\L1 Transformer best model\\Transformer_BERT_Model.h5')


# model trained on UN dataset to get negatively toned suggestions instead of RO terms
RO_to_UN_roberta_model=RobertaForMaskedLM.from_pretrained("./RO_to_UN_roberta_model")


from sentence_transformers import SentenceTransformer, util
similarity_score_model = SentenceTransformer('nli-distilroberta-base-v2')
# similarity_score_model = SentenceTransformer('roberta-base') can use this model as well, but result get compromises a bit

from transformers import AutoTokenizer, AutoModelForSequenceClassification
flunecy_tokenizer = AutoTokenizer.from_pretrained("dweb/deberta-base-CoLA")
fluency_model = AutoModelForSequenceClassification.from_pretrained("dweb/deberta-base-CoLA")

# l1_RO_model=None
# l1_RO_tokenizer=None

from simpletransformers.seq2seq import Seq2SeqModel

seq2seq_model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 250,
        "num_beams":None,
        "train_batch_size": 8,
        "num_train_epochs": 2,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "evaluate_generated_text": True,
        "learning_rate" : 5e-5,
        "do_sample":True,
        "top_k":50,
        "top_p":0.95,
        "num_return_sequences":3,
        "evaluate_during_training": False,
        "evaluate_during_training_verbose": False,
        "use_multiprocessing" :False,
        "reprocess_input_data": True,
    }

def seq2seq_paraphraser(comment):
    model=Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="C:\\StackOverflow_Work\\ML models\\seq2seq_model\\outputs",     # Already trained model
        args=seq2seq_model_args,
        use_cuda=False,
    )
    # comment=comment_Preprocess(Comment)
    return (model.predict([comment]))

def Mask_Regex(Comment):
    Regex_df=pd.read_csv("H:\\SO Norms Work\\Regex_Updated.csv")
    Regexes=Regex_df["Regex_Text"]

    for Regex in Regexes:
        # print(Regex)
        pattern = re.compile(Regex)
        for m in pattern.finditer(Comment):
            # print(m.start(),m.end(),m.group(),Regex,Regex_Data.loc[Regex_Data["Regex_Text"]==Regex,"Regex_Class"].iloc[0])
            location = {"Start": m.start(),
                        "End": m.end()}
            # print(location)
            Sub_String=Comment[location['Start']:location['End']]
            word=r"[a-zA-Z0-9]*"+re.escape(Sub_String)+r"[a-zA-Z0-9]*"
            Comment=re.sub(word,'<mask>', Comment).strip()
            # print(Comment)
            break
    return  Comment

def get_paraphrase_sugestions(sentence):
    token_ids = roberta_tokenizer.encode(sentence, return_tensors='pt')
    masked_position = torch.nonzero(token_ids.squeeze() == roberta_tokenizer.mask_token_id)
    masked_pos = [mask.item() for mask in masked_position]

    output = RO_to_UN_roberta_model(token_ids)
    last_hidden_state = output[0].squeeze()
    list_of_suggestions = []
    for index, mask_index in enumerate(masked_pos):
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=5, dim=0)[1]
        words = [roberta_tokenizer.decode(i.item()).strip() for i in idx]
        list_of_suggestions.append(words)
        # print("mask ", index + 1, "Guesses: ", words)
        # best_guess = ""       #to return the best guess, usually the 0th item from the list
        # for j in list_of_list:
        #     best_guess = best_guess + " " + j[0]
        # print(list_of_list)
    # print(list_of_suggestions)
    return list_of_suggestions

def remove_regex(suggestions):
    censored_suggestions =[]
    Regex_df = pd.read_csv("H:\\SO Norms Work\\Regex_Updated.csv")
    Regexes = Regex_df["Regex_Text"]
    for suggestion in suggestions:
        comment=" ".join(suggestion)
        # print(comment)
        for Regex in Regexes:
            # print(comment)
            pattern = re.compile(Regex)
            for m in pattern.finditer(comment):
                location = {"Start": m.start(),
                            "End": m.end()}
                # print(location)
                Sub_String = comment[location['Start']:location['End']]
                # print(Sub_String)
                word=r"[a-zA-Z0-9]*"+re.escape(Sub_String)+r"[a-zA-Z0-9]*"
                comment = re.sub(word, '', comment).strip()
                # print(comment)
                break
        # print(comment)
        comment=re.sub(' +', ' ', comment)      # remove multiple spaces, otherwise that will be an entry in the list
        censored_words=comment.split(" ")
        censored_suggestions.append(censored_words)
    return censored_suggestions

def classification_accuracy(suggestions):
    classifier = pipeline("sentiment-analysis")   # this is the bert model trained on sst-2 (GLUE dataset)
    # for suggestion in suggestions:        # sentiment classifier using VADER score
    #     # acc=[0 if score["compound"]>-0.05 else 1 for score in analyser.polarity_scores(suggestion)  ]
    #     score=analyser.polarity_scores(suggestion)
    #     if(score["compound"]>0):            # if the sentiment is positive, that is an undesired suggestion for an RO comment
    #         acc.append(0)
    #     else:                                   # if the sentiment is negative, that can be a potential suggestion
    #         acc.append(1)
    # return np.array(acc)


    # for suggestion in suggestions:        # Replaced the whole block with single line below
    #     result = classifier(suggestion)[0]
    #     if result['label'] == 'NEGATIVE':
    #         # acc.append((round(result["score"], 4)))
    #         acc.append(1)
    #     else:
    #         acc.append(0)

    acc=[1 if ((classifier(suggestion)[0])['label']=='NEGATIVE') else 0 for suggestion in suggestions]
    print("Sentiment accuracy: ", np.array(acc))
    return np.array(acc)

def semantic_similarity_calcularor(suggestions, actual_comment):
    # Compute embedding for the suggestions and actual comment, then returns the cosine similarity distnace between those
    embeddings1 = similarity_score_model.encode(suggestions, convert_to_tensor=True)
    embeddings2 = similarity_score_model.encode(actual_comment, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)  # Compute cosine-similarities
    print("Semantic similarity :",  cosine_similarity.numpy()[:,0])
    return cosine_similarity.numpy()[:,0]

def fluency_calculator(suggestions):
    fluency_model = 'textattack/bert-base-uncased-CoLA'
    fluency = pipeline("sentiment-analysis",
                       model=fluency_model,
                        tokenizer=fluency_model)
    for suggestion in suggestions:
        print(fluency(suggestion)[0])
    fluency_score=[1 if fluency(suggestion)[0]["label"]=='LABEL_1' else 0 for suggestion in suggestions]
    # # print("fluency score:", fluency_score)
    # return np.array(fluency_score)
    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    fluency_normalized=scaler.fit_transform(fluency_score)
    # print("fluency normalized:", fluency_normalized)
    print("Fluency :", np.array(fluency_normalized).reshape(len(fluency_normalized),))
    return np.array(fluency_normalized).reshape(len(fluency_normalized),)

def calculate_j_score(suggestions, actual_comment):
    accuracy=classification_accuracy(suggestions)
    # print("sentiment accuracy:", accuracy)
    similarity_score=semantic_similarity_calcularor(suggestions, actual_comment)
    # print("similarity score:", similarity_score)
    fluency_score=fluency_calculator(suggestions)
    # print("fluency score:", fluency_score)
    return np.multiply(accuracy, similarity_score, fluency_score)

def L1_BERT_predict(comments):
    # print(comments)
    # bert_tokenizer=l1_RO_tokenizer
    # L1_RO_trained_model=l1_RO_model
    test_input = bert_tokenizer.batch_encode_plus(comments,
                                                  add_special_tokens=True,
                                                  max_length=250,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=True)

    val_inp = np.asarray(test_input['input_ids'])
    val_mask = np.asarray(test_input['attention_mask'])
    logits = np.array(L1_RO_trained_model.predict([val_inp, val_mask], batch_size=16)).squeeze(axis=0)
    # print(logits)
    pred_prob = tf.nn.sigmoid(logits).numpy()  # adding a sigmoid layer
    pred_prob = np.pad(pred_prob, ((0, 0), (1, 0)))  # Sigmoid returns only one probability, to make it 2, increasing the size of columns
    pred_prob[:, 0] = 1 - pred_prob[:, 1]
    # print(pred_prob)
    return (pred_prob)

def Lime_BERT_explanation(comment):
    # This module finds the explanation of a comment being offensive.
    # Then it normalizes the weights of words using min-max scaling and returns the corresponding words.
    hot_words=[]
    bert_exp = explainer.explain_instance(str(comment[0]),
                                          L1_BERT_predict,
                                          num_features=num_features,
                                          num_samples=LIME_num_samples,
                                          labels=(1,)
                                          )
    # print (bert_exp.as_map())
    # print(type(bert_exp))
    explanation_raw= dict(sorted(bert_exp.as_list(), key=lambda x: x[1], reverse=True))
    print("Lime explanation :", explanation_raw)

    scaler = MinMaxScaler()    # min-max scaler to normalize Lime explanation values, default scale (0,1)
    explanation_normalized=scaler.fit_transform(np.array(list(explanation_raw.values())).reshape(len(explanation_raw.values()),1))
    explanation_normalized=explanation_normalized.reshape(len(explanation_normalized),).tolist()
    print("Lime normalized scores:", explanation_normalized)

    explanation_list = list(explanation_raw.items())

    for score in explanation_normalized:    # Locate the words to be masked
        if score >= 0.5:
            index = explanation_normalized.index(score)
            hot_words.append(explanation_list[index][0])
    return hot_words

def infill_suggestions(comment, censored_suggestions): # function to fill the masks with the suggestions
    comment_restore = comment
    filled_suggestions = []
    for option in itertools.product(*censored_suggestions):
        comment = comment_restore
        for token in option:
            comment = comment.replace('<mask>', token,
                                      1)  # 1 -> replace only once, ie, the first instance from left to right
        filled_suggestions.append(comment)
    # print("Filled in suggestions: ", filled_suggestions)
    return filled_suggestions

# def load_models():      # this function is to keep the models in memory so that it avoids loading the models again and again.
#     global l1_RO_model
#     global l1_RO_tokenizer
#     l1_RO_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # For L1 RO model
#     l1_RO_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
#     l1_RO_model.load_weights('C:\\StackOverflow_Work\\ML models\\L1 Transformer best model\\Transformer_BERT_Model.h5')


def paraphrase(comment):
    # load_models()
    actual_comment=comment
    print("Actual comment:", actual_comment)

    L1_Lime_words_to_mask=Lime_BERT_explanation([comment])
    print("Lime words to mask:", L1_Lime_words_to_mask)

    for word in L1_Lime_words_to_mask:      # LIME masking
        comment=re.sub(r'\b'+word+r'\b','<mask>', comment)
    print("Comment with Lime masks:", comment)

    comment=Mask_Regex(comment)             # Regex masking
    print("Comment with Regex masks:", comment)

    total_suggestions=get_paraphrase_sugestions(comment)    # get the paraphrasing suggestions
    print("Token suggestions: ", total_suggestions)

    censored_suggestions=remove_regex(total_suggestions)    # censor potential RO terms from the suggestions
    # print(len(censored_suggestions))
    if (len(censored_suggestions)==1 and censored_suggestions[0][0]==''):
        censored_suggestions[0][0]="***"
    print("Censored suggestions: ", censored_suggestions)

    filled_suggestions=infill_suggestions(comment, censored_suggestions)
    print("Infilled suggestions:", *filled_suggestions, sep = "\n")

    j_scores=calculate_j_score(filled_suggestions, actual_comment)
    print("J-scores:", j_scores)

    print("Top five suggestions:")
    for suggestion, score in zip(filled_suggestions, j_scores):     # suggestions with J-scores
            print(suggestion,":", score)

    # print ("Final suggestions")
    final_suggestions=[(filled_suggestions[i],j_scores[i]) for i in range(len(j_scores)) if j_scores[i]>0.5]     # Ranking based on j_score
    # final_suggestions = [(filled_suggestions[i], j_scores[i]) for i in range(len(filled_suggestions))]
    if final_suggestions:
        # print(final_suggestions)
        final_suggestions.sort(key=itemgetter(1),reverse=True)
        # print(final_suggestions)
        if len(final_suggestions)<=5:    # top 5 suggestions
            # print(final_suggestions)
            return final_suggestions
        else:
            # print(final_suggestions[:5])
            return  final_suggestions[:5]
    else:                   # if no more suggestions from paraphrases, then use seq2seq outputs
        # print("Here is the comment", actual_comment)
        # exit(0)
        # print("Suggestions from seq2seq model:", seq2seq_paraphraser(actual_comment))
        print("SE related, but suggestions are from seq2seq model:")
        return seq2seq_paraphraser(actual_comment)


def SE_related(comment):        # this module checks the SE relevance of a comment before paraphrasing
    se_count = 0
    with open('H:\\SE_Thesaurus_New_v2.json') as json_file:     # The latest SE relation file, after discarding SO terms
        se_thesaurus = json.load(json_file)  # this outputs as a dictionary
    se_list = [item[0] for item in se_thesaurus.items()]

    comment = comment.lower()
    for word in comment.split():
        if word in se_list:
            se_count += 1
            break

    return se_count

if __name__=='__main__':

    comment = "Why the fucking phatom post, downvote it."

    if SE_related(comment):
        suggestions=paraphrase(comment)
        print("Recommended paraphrases:")
        print(*suggestions, sep = "\n")
    else:
        print("Not SE related, suggestions from seq2seq model:")
        seq2seqlist=seq2seq_paraphraser(comment)
        print(len(seq2seqlist))
        print(*seq2seqlist[0],  sep = "\n")



#
#     paraphrase_suggestions=[]
#     df = pd.read_csv("H:\\SO Norms Work\\Paraphrasing works\Evaluation\\Final _SE_RO_comments1.csv")
#     comments=df.Comment_Text
#     for comment in comments:
#         paraphrase_suggestions.append(paraphrase(comment))
#
# # why the fucking phantom downvote
#
# #   To see how paraphrasing works with a set of comments
# #     paraphrase_suggestions=[]
# #     masked_suggestions=[]       # the comment with RO terms masked
# #     df=pd.read_csv("H:\\Sample_RO_comments_to_paraphrase.csv")
# #     comments=df.Comment_Text
# #     for comment in comments:
# #         masked_suggestions.append(Mask_Regex(comment))
# #         paraphrase_suggestions.append(remove_regex(get_paraphrase_sugestions(Mask_Regex(comment))))
# #
#     out_df = pd.DataFrame()
#     # out_df["Comment_Id"]=pd.Series(Comment_Id)
#     out_df["Comment_Text"] = pd.Series(comments)
#     # out_df["Masked_suggestions"]=pd.Series(masked_suggestions)
#     out_df["Suggestions"] = pd.Series(paraphrase_suggestions)
#     print(out_df)
#     out_df.to_csv("H:\\sample_RO_paraphrase_suggestions_Nov_5.csv", index=False)




    # print(explanation)
        # print(BERT_predict(Comment_Text))
        # print(softmax(list(explanation.values())))
    # print(list(value/max(explanation.values()) for value in explanation.values()))

    # explanation=np.array(list(explanation.values())).reshape(1,num_features) This is same as the above normalization
    # print(normalize(explanation, norm='max'))


