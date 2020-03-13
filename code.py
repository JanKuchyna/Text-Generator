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
end_of_list = 100


for i in data_list:
	for v in i:
		text.append(chars_to_int[v])
		labels.append(chars_to_int[v])

		
data_input = []
	
for i in text:

	data_input.append(text[start_of_list:end_of_list])

	start_of_list += 1
	end_of_list += 1 

text = pd.DataFrame(data_input[:-100])
labels = pd.DataFrame(labels[100:])

text = (text / 65)
labels = to_categorical(labels)


input_data = np.array(text)
output_data = np.array(labels)

print(np.shape(input_data))

n_patterns = len(input_data)

input_data = input_data.reshape(143690, 100, 1)

beginning = input_data[2]

from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,Dropout
from keras.optimizers import nadam,adam,SGD,RMSprop
from keras.callbacks import ModelCheckpoint

#284 644 800
model = Sequential([
	LSTM(256, input_shape=(100,1), return_sequences=True),
	Dropout(0.2),
	LSTM(256),
	Dropout(0.2),
	Dense(100,activation="relu"),
	Dense(66, activation='softmax')
	])

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#model.fit(input_data,output_data,validation_split=0.1,batch_size=128,
#	epochs=105,verbose=1,callbacks=callbacks_list)

#model.save("text-gen.h5")

model = load_model("text-gen-smarter.h5")


beginning = np.array(beginning)

int_to_chars = {v: k for k, v in chars_to_int.items()}

predict_int = []

predict_merge = []

output = []

ints = []

test_text = []



test_data = beginning.reshape(1,100)
test_data = 65*test_data
for i in test_data:
	for v in i:
		test_text.append(int_to_chars[v])

print("".join(test_text))

for i in range(1000):
	beginning = beginning.reshape(1,100,1)

	test_data = beginning.reshape(100,1)

	predict = model.predict(beginning)
	predict_merge =[ (np.argmax(predict[0]))]
	#for i in predict:
	#	predict_merge.append(np.argmax(i))

	predict_merge = pd.DataFrame(predict_merge)
	predict_merge = predict_merge/65

	new_test_data = pd.DataFrame(test_data)

	new_test_data = new_test_data.append(predict_merge, ignore_index=True)


	new_test_data = new_test_data[1:101]

	beginning = np.array(new_test_data)


	for i in predict:
		
		predict_int.append(np.argmax(i, axis=-1))


	for i in predict_int:
		output.append(int_to_chars[i])
	predict_int = []

prediction = ("".join(test_text),"".join(output))
print("".join(prediction))
