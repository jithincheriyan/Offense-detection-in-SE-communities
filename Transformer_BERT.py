import  pandas as pd
import  numpy as np
from transformers import *
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import logging as hf_logging
import tensorflow as tf
import wandb
import  torch
import os
hf_logging.set_verbosity_error()
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
os.environ["WANDB_API_KEY"] = "980f6fd5d9bd3d0adbd7b9a244e7b151326bd44c"
wandb.init(project='Transformer_Bert', entity='cheriyan', config={"bert_model":"bert-large-uncased",
                                                                  "batch_size": 8,
                                                                  "learning_rate": 2e-5, # 3e-5 is the BERT recommended Learning rate
                                                                  "num_labels":2,
                                                                  "add_special_tokens": True,
                                                                  "max_length" : 100,
                                                                  "padding" : True,
                                                                  "truncation": True,
                                                                  "return_attention_mask":True,
                                                                  "random_seed":123
                                                                  })
config=wandb.config
tf.random.set_seed(config.random_seed)
bert_tokenizer = BertTokenizer.from_pretrained(config.bert_model)
bert_model = TFBertForSequenceClassification.from_pretrained(config.bert_model,num_labels=config.num_labels)
loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate,epsilon=1e-08)
metric = tf.keras.metrics.BinaryAccuracy('accuracy')


def data_encode():
    # text = "Sorry i didn t get u post can explain clearly"
    # print(len(text))
    # tokens = bert_tokenizer.tokenize(text)
    # token_ids = bert_tokenizer.encode(text, add_special_tokens=True, max_length=250, padding=True)
    # print(tokens)
    # print(token_ids)

    df = pd.read_csv("H:\\Training Data\\BERT_Training_Data_Augmented_V3.csv", encoding = "ISO-8859-1").sample(frac=1)
    sentences=df.Comment_Text.tolist()
    # print(max(sentences, key=len))

    bert_inp=bert_tokenizer.batch_encode_plus(sentences,
                                            add_special_tokens = config.add_special_tokens,
                                            max_length =config.max_length,
                                            padding=config.padding,
                                            truncation = config.truncation,
                                            return_attention_mask = config.return_attention_mask)
    input_ids =np.asarray(bert_inp['input_ids'])
    attention_masks =np.array(bert_inp['attention_mask'])
    labels = np.array(df.Label)
    print(input_ids.shape, attention_masks.shape,labels.shape)
    return input_ids,attention_masks,labels

def train(input_ids,attention_masks,labels):
    print("training")
    train_inp,val_inp,train_label,val_label,train_mask,val_mask=train_test_split(input_ids,
                                                                                 labels,
                                                                                 attention_masks,
                                                                                 test_size=0.2)
    bert_model.compile(loss=loss,
                       optimizer=optimizer,
                       metrics=[metric])
    # callbacks = [EarlyStopping(monitor='val_loss',
    #                            verbose=1,
    #                            baseline=None)]
    # model_checkpoint = ModelCheckpoint('H:\\best_model_Checkpoint.h5',
    #                                    monitor='val_loss',
    #                                    save_best_only=True)
    bert_model.fit([train_inp,train_mask],
                   train_label,
                   batch_size=config.batch_size,
                   validation_data=([val_inp,val_mask],val_label),
                   # steps_per_epoch=config.steps_per_epoch
                   # callbacks=[callbacks,model_checkpoint]
                   )
    print("training over")
    #bert_model.save(model_save_path)
    bert_model.save_weights('H:\\Transformer_DistilBERT_Model.h5')
    print("model saved")


def test():
    out_df = pd.DataFrame()
    # path="H:\\Testing Data\\Nov_CSV_Merged_Preprocessed"
    path="H:\\Test"
    test_df=pd.read_csv(path+".csv", encoding = "ISO-8859-1")
    Comment_Text=test_df.Comment_Text.tolist()

    trained_model = TFBertForSequenceClassification.from_pretrained(config.bert_model,num_labels=config.num_labels)
    # trained_model.load_weights('C:\\StackOverflow_Work\\ML models\\Transformer best model\\Transformer_BERT_Model.h5')
    trained_model.load_weights("H:\\Transformer_BERT_Large_Model.h5")
    trained_model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=[metric])
    test_input=bert_tokenizer.batch_encode_plus(Comment_Text,
                                                add_special_tokens=config.add_special_tokens,
                                                max_length=config.max_length,
                                                padding=config.padding,
                                                truncation=config.truncation,
                                                return_attention_mask=config.return_attention_mask)

    val_inp=np.asarray(test_input['input_ids'])
    val_mask=np.asarray(test_input['attention_mask'])

    # prediction = np.array(trained_model.predict([val_inp, val_mask], batch_size=16)).squeeze(axis=0)
    prediction = trained_model.predict([val_inp, val_mask], batch_size=config.batch_size)
    # print(type(prediction))
    print(np.shape(prediction))
    print(prediction)

    pred_prob = tf.nn.softmax(prediction).numpy()
    # pred_prob=torch.sigmoid(prediction).numpy()
    print(pred_prob)
    print(np.shape(pred_prob))
    pred_labels=pred_prob[0].argmax(axis=1)

    # pred_prob=tf.nn.sigmoid(prediction).numpy()         # for single label prediction , where 0.5 is the threshold
    # pred_labels = [0 if pred <= 0.5 else 1 for pred in pred_prob]

    print(pred_prob)

    
    '''
    prediction = trained_model.predict([val_inp, val_mask], batch_size=16)
    print(type(prediction))
    print(np.shape(prediction))
    print(prediction)
    print(max(prediction), min(prediction))

    # print(prediction)
    pred_proba = tf.nn.softmax(prediction).numpy()
    pred_labels = pred_proba[1].argmax(axis=1)
    print(pred_labels)
    # pred_labels=[0 if pred < 0.75 else 1 for pred in prediction]         # use argmax to get the CLASS with the maximum probability
    # prediction[prediction>0.75]=1
    # prediction[prediction<= 0.75] = 0

'''
    out_df["Comment_Text"] = pd.Series(Comment_Text)
    out_df["Bert_Label"] =pd.Series(pred_labels)
    print(out_df)
    out_df.to_csv(path + "_BERT_Prediction.csv", index=False)

if __name__ == '__main__':
    # input_ids, attention_masks, labels=data_encode()
    # train(input_ids, attention_masks, labels)
    test()

