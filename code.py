chars_to_int = {'é':1, '’':2, 'j':3, '‘':4, '&':5, '4':6, 'h':7, 'o':8, '5':9, '—':10, 'k':11, '?':12, ':':13, 'p':14, '!':15, '1':16, '3':17, 'x':18, 'v':19,
 'l':20, '”':21, '0':22, 'b':23, 'y':24, 'æ':25, 'â':26, '(':27, '7':28, 'à':29, 'e':30, '-':31, '£':32, '9':33, '½':34, 'è':35, 'i':36, 'c':37,
 'q':38, 'œ':39, ';':40, 'a':41, ')':42, 's':43, 't':44, 'w':45, '“':46, '\n':47, '6':48, 'n':49, ' ':50, '.':51, 'f':52, 'm':53, 'd':54, 'z':55, 
 '2':56, '8':57, 'r':58, 'g':59, 'u':60, ',':61}

import numpy as np
import pandas as pd
from keras.utils import to_categorical

data = open("/home/honza/programování/python/deeplearning/Text Generator/files/Sherlock",encoding='UTF-8')

data = data.read()

data = data.lower()

data_list = [data]





letter_counter = 0
sentence_counter = 0
batch = []
text = []
labels = []

for word in data_list:
	for letter in word:
		sentence_counter += 1

		batch.append(chars_to_int[letter])
		
		if sentence_counter == 100:
			text.append(batch)
			batch = []
			sentence_counter = 0
		letter_counter += 1
		if letter_counter == 101:
			labels.append(chars_to_int[letter])
			letter_counter = 1




text = pd.DataFrame(text)
labels = pd.DataFrame(labels)
l = {0:0}

labels.loc[-1] = [0]  # adding a row
labels.index = labels.index + 1  # shifting index
labels.sort_index(inplace=True) 
print(labels)

text = to_categorical(text)
labels = to_categorical(labels)



input_data = np.array(text)
output_data = np.array(labels)

n_patterns = len(input_data)

input_data = input_data.reshape(n_patterns,6200,1)
 

from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,Dropout
from keras.optimizers import nadam,adam,SGD,RMSprop

model = Sequential([
	LSTM(256, input_shape=(6200,1), return_sequences=True),
	Dropout(0.2),
	LSTM(256,recurrent_dropout=0.1),
	Dropout(0.2),
	Dense(62, activation='softmax')
	])

model.compile(optimizer=nadam(lr=.001),loss="binary_crossentropy")

model.fit(input_data,output_data,validation_split=0.1,batch_size=128,epochs=1,verbose=1)

model.save("text-gen.h5")

model = model_load("text-gen.h5")

chars = []


#for i in range(1000):
#	predict = model.predict_classes(begining)
#
#	for i in predict:
#		chars.append(np.argmax(i, axis=-1))
