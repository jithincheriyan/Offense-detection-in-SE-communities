import streamlit as st
import  numpy as np
import tensorflow as tf
from transformers import *
from transformers import BertTokenizer
tf.random.set_seed(42)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=1)

def L1_test(comment):
    # add this to test a single comment
    # Comment_Text = []
    # Comment_Text.append(Comment)          # where Comment is the comment to be tested

    trained_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    trained_model.load_weights('H:\\L1_Transformer_BERT_Model.h5')

    test_input = bert_tokenizer.batch_encode_plus(comment,
                                                  add_special_tokens=True,
                                                  max_length=250,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=True)

    val_inp = np.asarray(test_input['input_ids'])
    val_mask = np.asarray(test_input['attention_mask'])

    # code for single label prediction, numb_labels=1
    prediction = np.array(trained_model.predict([val_inp, val_mask], batch_size=250)).squeeze(axis=0)
    pred_prob = tf.nn.sigmoid(prediction).numpy()  # for single label prediction , where 0.5 is the threshold
    pred_labels = 0 if pred_prob[0][0] <= 0.5 else 1
    print(pred_prob)
    print(pred_labels)
    return (pred_labels)


if __name__ == '__main__':
    st.title('Abuse detector')
    comment=st.slider("Input the comment")
    st.text(L1_test([comment]))
