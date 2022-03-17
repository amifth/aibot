import nltk
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy
import tflearn
from tensorflow.python.framework import ops
import random
import json
from tflearn.optimizers import Adam
import jsonify

ops.reset_default_graph()
net = tflearn.input_data(shape=[None, 64])
net = tflearn.fully_connected(net, 32)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 16)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8, activation="softmax")
adam = Adam(learning_rate=0.001, beta1=0.99)
net = tflearn.regression(net, optimizer=adam)
model = tflearn.DNN(net)
model.load('models/model.tflearn')
print(model)

with open('knowledge.json') as file:
	data = json.load(file)

labels = []
words = []
for i in data["knowledge"]:
    # labels.append(i["tag"])
    if i["tag"] not in labels:
        labels.append(i["tag"])
    for j in i["patterns"]:
        word = nltk.word_tokenize(j)
        words.extend(word)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# print(labels)
# print(words)

def bag_of_words(s, kata):
	bag = [0 for _ in range(len(kata))]

	s_words = nltk.word_tokenize(s)
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(kata):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

def chat():
	print("Halo Human Selamat Datang! Press 0 for end session")
	while True:
		inp = input("Request: ")
		if inp.lower()=="0":
			break

		result = model.predict([bag_of_words(inp,words)])[0]
		result_index = numpy.argmax(result)
		tag = labels[result_index]
		print(result[result_index])
		if result[result_index] > 0.80:
			for tg in data["knowledge"]:
				if tg["tag"] == tag:
					responses = tg['responses']

			print("AI Response: ",random.choice(responses))
		else:
			print("Maaf saya tidak mengerti, silahkan pertanyaan lain")

# get response for api
def get_response(inp):
	result = model.predict([bag_of_words(inp,words)])[0]
	result_index = numpy.argmax(result)
	tag = labels[result_index]
	if result[result_index] > 0.80:
		for tg in data["knowledge"]:
			if tg["tag"] == tag:
				responses = tg['responses']
		return random.choice(responses),result[result_index]
	else:
		return "maaf saya tidak mengerti, silahkan pertanyaan lain",result[result_index]

if __name__ == "__main__":
    print("\n")
    print("\n")
    print("\n")
    chat()
