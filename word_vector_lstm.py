#coding:utf-8
import sys
from tensorflow import keras
# reload(sys)
# sys.setdefaultencoding('utf8')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
from tensorflow.keras.models import Sequential
import gensim
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Embedding,LSTM
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
from tensorflow.keras.models import Sequential
VECTOR_DIR = 'baike.vectors.bin'

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.16
TEST_SPLIT = 0.2


print('(1) load texts...')
train_texts = open('train_contents.txt', 'rb').read().decode('utf-8').split('\n')
train_labels = open('train_labels.txt', 'rb').read().decode('utf-8').split('\n')
test_texts = open('test_contents.txt', 'rb').read().decode('utf-8').split('\n')
test_labels = open('test_labels.txt', 'rb').read().decode('utf-8').split('\n')
all_texts = train_texts + test_texts
all_labels = train_labels + test_labels

print('(2) doc to var...')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


print('(3) split data set...')
p1 = int(len(data)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(data)*(1-TEST_SPLIT))
x_train = data[:p1]
y_train = labels[:p1]
x_val = data[p1:p2]
y_val = labels[p1:p2]
x_test = data[p2:]
y_test = labels[p2:]
print('train docs: '+str(len(x_train)))
print('val docs: '+str(len(x_val)))
print('test docs: '+str(len(x_test)))

print('(4) load word2vec as embedding...')

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
not_in_model = 0
in_model = 0
for word, i in word_index.items(): 
    if word in w2v_model:
        in_model += 1
        embedding_matrix[i] = np.asarray(w2v_model[word], dtype='float32')
    else:
        not_in_model += 1
print(str(not_in_model)+' words not in w2v model')

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print('(5) training model...')


model = Sequential()
model.add(embedding_layer)
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()
# plot_model(model, to_file='model.png',show_shapes=True)
# exit(0)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print(model.metrics_names)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
model.save('word_vector_lstm.h5')

print('(6) testing model...')
print(model.evaluate(x_test, y_test))

        




