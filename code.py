chars_to_int = {'é':1, '’':2, 'j':3, '‘':4, '&':5, '4':6, 'h':7, 'o':8, '5':9, '—':10, 'k':11, '?':12, ':':13, 'p':14, '!':15, '1':16, '3':17, 'x':18, 'v':19,
 'l':20, '”':21, '0':22, 'b':23, 'y':24, 'æ':25, 'â':26, '(':27, '7':28, 'à':29, 'e':30, '-':31, '£':32, '9':33, '½':34, 'è':35, 'i':36, 'c':37,
 'q':38, 'œ':39, ';':40, 'a':41, ')':42, 's':43, 't':44, 'w':45, '“':46, '\n':47, '6':48, 'n':49, ' ':50, '.':51, 'f':52, 'm':53, 'd':54, 'z':55, 
 '2':56, '8':57, 'r':58, 'g':59, 'u':60, ',':61, "*":62,'ù':63,'[':64,']':65}

import numpy as np
import pandas as pd
from keras.utils import to_categorical

data = open("/home/honza/programování/python/deeplearning/Text Generator/files/alice",encoding='UTF-8')

data = data.read()

data = data.lower()

data_list = [data]




text = []
labels = []

start_of_list = 0
labels = []
end_of_list = 30


for i in data_list:
	for v in i:
		text.append(chars_to_int[v])
		labels.append(chars_to_int[v])

		
data_input = []
	
for i in text:

	data_input.append(text[start_of_list:end_of_list])

	start_of_list += 1
	end_of_list += 1 


text = pd.DataFrame(data_input[:-30])
labels = pd.DataFrame(labels[30:])
print(text)


text = to_categorical(text)
labels = to_categorical(labels)


input_data = np.array(text)
output_data = np.array(labels)


n_patterns = len(input_data)

input_data = input_data.reshape(n_patterns,1980,1)
 
from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,Dropout
from keras.optimizers import nadam,adam,SGD,RMSprop

model = Sequential([
	LSTM(80, input_shape=(1980,1), return_sequences=True),
	Dropout(0.2),
	LSTM(70,recurrent_dropout=0.1),
	Dropout(0.2),
	Dense(66, activation='softmax')
	])

model.compile(optimizer=nadam(lr=.001),loss="binary_crossentropy")

model.fit(input_data,output_data,validation_split=0.1,batch_size=10,epochs=1,verbose=1)

model.save("text-gen.h5")

model = model_load("text-gen.h5")

chars = []

#for i in range(1000):
#	predict = model.predict_classes(begining)
#
#	for i in predict:
#		chars.append(np.argmax(i, axis=-1))
