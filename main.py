import pandas as pd
import numpy as np
import tensorflow as tf


def load_data(path):
    df=pd.read_json(path)
    return df

def preprocessing(df,split_size):
    sentences=df['headline']
    labels=df['is_sarcastic']
    train_data = sentences[0:int(len(sentences)*split_size)]
    train_labels = labels[0:int(len(labels)*split_size)]

    test_data = sentences[int(len(sentences)*split_size):]
    test_labels = labels[int(len(sentences)*split_size):]
    training_sentences=[]
    training_labels=[]
    testing_sentences=[]
    testing_labels=[]

    # Loop over all training examples and save the sentences and labels
    for s in train_data:
        training_sentences.append(s)
    for l in train_labels:
        training_labels.append(l)

    # Loop over all test examples and save the sentences and labels
    for s in test_data:
        testing_sentences.append(s)
    for l in test_labels:
        testing_labels.append(l)

    # Convert labels lists to numpy array
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000,oov_token='<OOV>')
    tokenizer.fit_on_texts(training_sentences)
    training_sequences=tokenizer.texts_to_sequences(training_sentences)
    train_padded = tf.keras.preprocessing.sequence.pad_sequences(training_sequences,maxlen=120,truncating='post')
    test_sequences =tokenizer.texts_to_sequences(testing_sentences)
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=120,truncating='post')

    return test_padded,testing_labels_final,train_padded,training_labels_final

def create_model(vocab_size, embedding_dim, maxlen):
    model=tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
        tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(56, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

    ])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


df=load_data("C:\Progetti\Personali\MachineLearning\\nlp\Coursera\Sarcasm\Data\sarcams.json")
testing_sentences,testing_labels_final,training_sentences,training_labels_final=preprocessing(df,0.8)

model=create_model(10000,16,120)
num_epochs = 30

# Train the model
history = model.fit(training_sentences, training_labels_final, epochs=num_epochs, validation_data=(testing_sentences, testing_labels_final), verbose=2)