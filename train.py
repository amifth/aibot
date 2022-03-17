# import libs
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy
import tflearn
from tflearn.optimizers import Adam
from tensorflow.python.framework import ops
import json
import pickle

with open('knowledge.json') as file:
	data = json.load(file)

try:
	with open("data.pkl", "rb") as f:
		words, labels, trainings, outputs = pickle.load(f)
except:
	words = []
	labels = []
	doc_x = []
	doc_y = []
	for i in data["knowledge"]:
		for pattern in i["patterns"]:
			word = nltk.word_tokenize(pattern)
			words.extend(word)
			doc_x.append(word)
			doc_y.append(i['tag'])
			if i["tag"] not in labels:
				labels.append(i["tag"])
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))
	labels = sorted(labels)
	trainings = []
	outputs = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(doc_x):
		bag = []
		kta = [stemmer.stem(w) for w in doc]
		for w in words:
			if w in kta:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:]
		output_row[labels.index(doc_y[x])] = 1

		trainings.append(bag)
		outputs.append(output_row)

	trainings = numpy.array(trainings)
	outputs = numpy.array(outputs)

	with open("data.pkl", "wb") as f:
		pickle.dump((words, labels, trainings, outputs),f)

# generate model use tf
ops.reset_default_graph()
net = tflearn.input_data(shape=[None, len(trainings[0])])
net = tflearn.fully_connected(net, 32)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 16)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(outputs[0]), activation="softmax")
adam = Adam(learning_rate=0.001, beta1=0.99)
net = tflearn.regression(net, optimizer=adam)
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.fit(trainings, outputs, n_epoch=100000, batch_size=64, show_metric=True)
model.save("models/model.tflearn")

print(len(trainings[0]))
print(len(outputs[0]))
print(labels)
print(words)
print("training done")