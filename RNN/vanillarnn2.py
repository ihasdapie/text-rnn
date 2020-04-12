#!/usr/bin/python

#imports

import numpy as np
import theano as theano
import theano.tensor as T
from utils import *
import operator
import csv
import itertools
import itertools as it
import operator
import nltk
import sys
import os
import time
from datetime import datetime
from rnn_theano import RNNTheano
import timeit

#can be changed
vocabsize= 8000 #FLAG

starttoken='SENTENCE_START'
unknowntoken='UKNOWN_TOKEN'
endtoken= 'SENTENCE_END'

#read data and add start+end tokens
print 'Reading .csv file...'
with open('/home/ihasdapie/Documents/AI/Data/RedditComments.csv', 'rb') as f:
	reader= csv.reader(f, skipinitialspace= True)
	reader.next()
	#split textdump into sentences
	sentences= itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
	#add sentencestart and sentenceend tokens
	sentences= ["%s %s %s" % (starttoken, x, endtoken) for x in sentences]
print 'Parsed %d sentences. ' % (len(sentences))

#tokenize senteces to words
tokenizedsentences= [nltk.word_tokenize(words) for words in sentences]

#count word frequencies; rarely used words can be omitted for resource conservation
wordfreq= nltk.FreqDist(itertools.chain(*tokenizedsentences))
print '%d unique words' % len(wordfreq.items())

#get most common words and create index-words and word-index vectors
vocab= wordfreq.most_common(vocabsize-1)
indextoword= [x[0] for x in vocab]
indextoword.append(unknowntoken)
wordtoindex= dict([(w, i) for i,w in enumerate(indextoword)])

print 'Vocababulary size: %d' % vocabsize
print "The least frequent word is '%s', and appeared %d times" % (vocab[-1][0], vocab[-1][1])

#replace all words not in the vocab with unknown token

for i, words in enumerate(tokenizedsentences):
	tokenizedsentences[i] = [w if w in wordtoindex else unknowntoken for w in words]

'''
print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenizedsentences[0]
'''
#create training data
Xtrain = np.asarray([[wordtoindex[w] for w in words[:-1]] for words in tokenizedsentences])
ytrain = np.asarray([[wordtoindex[w] for w in words[1:]] for words in tokenizedsentences])

'''
#implementation
class RNNumpy:
	#worddim= size of vocab, hiddendim= size of hiddenlayer
	def __init__(self, worddim, hiddendim=100, bptttruncate=4):
		#assign inst. var
		self.worddim = worddim
		self.hiddendim= hiddendim
		self.bptttruncate= bptttruncate

		#random initialization of parameters
		self.U = np.random.uniform(-np.sqrt(1./worddim), np.sqrt(1./worddim), (hiddendim, worddim))
		self.V = np.random.uniform(-np.sqrt(1./hiddendim), np.sqrt(1./hiddendim), (worddim, hiddendim))
		self.W = np.random.uniform(-np.sqrt(1./hiddendim), np.sqrt(1./hiddendim), (hiddendim, hiddendim))

#forward propragation - Partially taken from the introductary nn code

def forwardprop(self, x):
	#total # of time-steps
	T= len(x)
	#must save all hidden stages of s
	#add another element to hidden layer
	s= np.zeros((T+1, self.hiddendim))
	#the outputs at each time step
	o= np.zeros((T, self.worddim))
	#during every time step:
	for t in np.arange(T):
		#index U by x[t]; same as multiplying U with one (hot) vector
		s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
		o[t] = softmax(self.V.dot(s[t]))
	return [o, s]
RNNumpy.forwardprop= forwardprop

def perdict(self, x):
	#take highest score of forward prop.
	o, s= self.forwardprop(x)
	return np.argmax(o, axis= 1)

RNNumpy.perdict= perdict


#see example output
np.random.seed(10)
model= RNNumpy(vocabsize)
o, s = model.forwardprop(Xtrain[10])
#print o.shape
#print o

perdictions= model.perdict(Xtrain[10])
#print perdictions.shape
#print perdictions

def totalloss(self, x, y):
	L= 0
	for i in np.arange(len(y)):
		o, s = self.forwardprop(x[i])
		#only perdictions of correct words
		correctwordpred = o[np.arange(len(y[i])), y[i]]
		#add to loss
		L += -1*np.sum(np.log(correctwordpred))
	return L
RNNumpy.totalloss= totalloss
def calcloss(self, x, y):
	#divide loss by number of training examples
	N= np.sum((len(y_i) for y_i in y))
	return self.totalloss(x, y)/N
RNNumpy.calcloss= calcloss

#limit to 1000 examples
print "expected loss for random predictions: %f" % np.log(vocabsize)
print "actual loss: %f" % model.calcloss(Xtrain[:1000], ytrain[:1000]) #this will give the total loss, but this is quite intensive so it can be quoted out
'''
'''
def bptt(self, x, y):
	T= len(y)
	#forwardprop
	o, s= self.forwardprop(x)
	#accumalate gradients in var.
	dLdU= np.zeros(self.U.shape)
	dLdV= np.zeros(self.V.shape)
	dLdW= np.zeros(self.W.shape)
	delta_o= o
	delta_o[np.arange(len(y)), y] -= 1.
	#per-output backwards
	for t in np.arange(T)[::-1]:
		dLdV += np.outer(delta_o[t], s[t].T)
		#initial delta calc
		delta_t= self.V.T.dot(delta_o[t]) * (1 - (s[t] **2 ))
		#bptt
		for bpttstep in np.arange(max(0, t-self.bptttruncate), t+1)[::-1]:
			# print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
			dLdW += np.outer(delta_t, s[bpttstep-1])              
			dLdU[:,x[bpttstep]] += delta_t
			# Update delta for next step
			delta_t = self.W.T.dot(delta_t) * (1 - s[bpttstep-1] ** 2)
	return [dLdU, dLdV, dLdW]

RNNumpy.bptt = bptt
'''
#gradient check
'''
def gradient_check(self, x, y, h= 0.001, errorthres= 0.01):
	#calcgradients
	bpttgradients= self.bptt(x, y)
	#list of parameters
	modelparam= ['U', 'V', 'W']
	#grad. check per param
	for pidx, pname in enumerate(model_parameters):
		#get param. value from mode
		parameter= operator.attrgetter(pname)(self)
		print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
		it= np.ndither(parameter, flags=['multi_index'], op_flags= ['readwrite'])
		while not it.finished:
			ix= it.multi_index
			#save org. value
			originalvalue= parameter[ix]
			#estimate gradient
			parameter[ix] = originalvalue + h
			gradplus = self.totalloss([x],[y])
			parameter[ix] = originalvalue - h
			gradminus = self.totalloss([x],[y])
			estimated_gradient = (gradplus - gradminus)/(2*h)
			#reset param.
			parameter[ix]= original_value
			#calc gradient
			backprop_gradient= bptt_gradients[pdix][ix]
			#calc error
			relative_error= np.abs((backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient)))
			#if error is too large fail
			if relative_error > errorthres:
				print "Gradient Check Error: parameter=%s ix=%s" % (pname, ix)
				print "+h loss: %f" % gradplus
				print "-h loss: %f" % gradminus
				print "Estimated_gradient: %f" % estimated_gradient
				print "Backpropagation gradient: %f" % backprop_gradient
				print "Relative Error: %f" % relative_error
				return
			it.iternext
		print "Gradient Check for parameter %s passed" % (pname)

RNNumpy.gradient_check=gradient_check
'''
'''
#use smaller vocab for performance

grad_check_vocab_size= 100
np.random.seed(10)
model=RNNumpy(grad_check_vocab_size, 10, bptttruncate= 1000)
model.gradient_check([0, 1, 2, 3 ], [1, 2, 3, 4])
'''
#perform sgd once


'''

def numpy_sgd_step(self, x, y, learning_rate):
	#calc gradientts
	dLdU, dLdV, dLdW = self.bptt(x, y)
	#change param according to gradients and learning rates
	self.U -= learning_rate * dLdU
	self.V -= learning_rate * dLdV
	self.W -= learning_rate * dLdW

RNNumpy.numpy_sgd_step = numpy_sgd_step
'''
#outer sgd loop
#model: dataset
#ytrain= datalabels
#learningrate= learningrate (initial)
#nepoch - number of epochs
#evallossafter- evalulate loss after this amount of epochs



def trainwithsgd(model, Xtrain, ytrain, learning_rate= 0.005, nepoch=100, evaluate_loss_after= 5):
	#keep track of losses for future plotting
	losses= []
	num_examples_seen = 0
	for epoch in range(nepoch):
		if (epoch% evaluate_loss_after == 0):
			loss= model.calcloss(Xtrain, ytrain)
			losses.append((num_examples_seen, loss))
			time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
			#adjust learning rate if the loss increases
			if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
				learning_rate= learning_rate*0.5
				print "setting learning rate to %f" % learning_rate
			sys.stdout.flush()
		#for each training example:
		for i in range(len(ytrain)):
			#one sgd step
			model.numpy_sgd_step(Xtrain[i], ytrain[i], learning_rate)
			num_examples_seen += 1
#train with NUMPY or THEANO; NUMPY is the one i coded, which is slower than theano but is already done. Subsitite pre-trained model. 
'''
np.random.seed(10) #FLAG
# Train on a small subset of the data to see what happens
model = RNNumpy(vocabsize)
losses = trainwithsgd(model, Xtrain[:100], ytrain[:100], nepoch=10, evaluate_loss_after=1)



np.random.seed(10) #FLAG
model = RNNTheano(vocabsize)
model.sgd_step(Xtrain[10], ytrain[10], 0.005)
'''

from utils import load_model_parameters_theano, save_model_parameters_theano

model = RNNTheano(vocabsize, hiddendim=50)
# losses = train_with_sgd(model, X_train, y_train, nepoch=50)
# save_model_parameters_theano('./data/trained-model-theano.npz', model)
load_model_parameters_theano('/home/ihasdapie/Documents/AI/Data/trained-model-theano.npz', model)
def generate_sentence(model):
	# We start the sentence with the start token
	new_sentence = [wordtoindex[starttoken]]
	# Repeat until we get an end token
	while not new_sentence[-1] == wordtoindex[endtoken]:
		next_word_probs = model.forward_propagation(new_sentence)
		sampled_word = wordtoindex[unknowntoken]
		# We don't want to sample unknown words
		while sampled_word == wordtoindex[unknowntoken]:
			samples = np.random.multinomial(1, next_word_probs[-1])
			sampled_word = np.argmax(samples)
		new_sentence.append(sampled_word)
	sentence_str = [indextoword[x] for x in new_sentence[1:-1]]
	return sentence_str


num_sentences= 5    #FLAG
senten_min_length= 5  #FLAG

for i in range(num_sentences):
	sent= []
	#to get long sentences, not ones with only 1-2 words. 
	while len(sent) < senten_min_length:
		sent= generate_sentence(model)
	print '', " ".join(sent)
