"""
Lisanne Wiengarten
Matriculation no. 3249897
Statistical Dependency Parsing
IMS, SuSe 17
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque

from Sentence import Sentence
from Sentence import Token
from Configuration import Configuration
	
# Performs actual parsing on new test sentences according to a previously trained model #
class GuideParser:
	def __init__(self, classifier, oracle):

		# Sets for correct and found arcs
		self.current_sent = Sentence(list())
		self.found_rarcs = set()
		self.found_larcs = set()
		
		# The classifier to ask for predicted transitions
		self.classifier = classifier
		
		self.oracle = oracle
		self.unique_feats = classifier.unique_feats
		self.oracle.oracle = False


	# Creates a leftarc from the front of the buffer to the top-most token on the stack #
	# And removes the top-most token on the stack
	# Also sets the head in the dependent token
	def doleftarc(self, c):
		self.current_sent.tokenlist[c.stack[0].id].head = c.buffer[0].id
		self.found_larcs.add((c.buffer[0].id, c.stack[0].id))
		
		# LA S[0] <- B[0]
		# My leftmost dependent is the smallest number that has me as head
		# d.h. ld von B[0]: wenn mein ld > S[0] setze mein ld neu
		#if c.buffer[0].ld > c.stack[0].id:
			#c.buffer[0].ld = c.stack[0].id
		
		# My rightmost dependent is the biggest number that has me as head
		# d.h. rd von B[0]: wenn rd < S[0] setze mein rd neu
		#if c.buffer[0].rd < c.stack[0].id:
			#c.buffer[0].rd = c.stack[0].id
		
		del c.stack[0]
			
		return c


	# Creates a right from the top-most token on the stack to the front of the buffer #
	# And moves the top-most token from the stack back onto the buffer
	# Also sets the head in the dependent token
	def dorightarc(self, c):
		self.current_sent.tokenlist[c.buffer[0].id].head = c.stack[0].id
		self.found_rarcs.add((c.stack[0].id, c.buffer[0].id))
		
		# My leftmost dependent is the smallest number that has me as head
		# d.h. ld von S[0]: wenn mein ld > B[0] setze mein ld neu
		if c.stack[0].ld > c.buffer[0].id:
			c.stack[0].ld = c.buffer[0].id
		
		# My rightmost dependent is the biggest number that has me as head
		# RA S[0] -> B[0]
		# d.h. rd von S[0]: wenn rd < B[0] setze mein rd neu
		if c.stack[0].rd < c.buffer[0].id:
			c.stack[0].rd = c.buffer[0].id
			
			
		del c.buffer[0]
		c.buffer.appendleft(c.stack[0])
		del c.stack[0]
		
		return c

		
	# Shift takes the first token from the front of the buffer and pushes it onto the stack #
	def shift(self, c):
		c.stack.appendleft(c.buffer[0])
		c.buffer.popleft()
			
		return c

	
	# Does the actual parsing on an unseen test sentence #
	# In: Test sentence from blind file
	# Out: Same sentence with annotations (heads and arcs)
	def parse_sentence(self, sent):
		self.current_sent = sent
		self.found_rarcs = set()
		self.found_larcs = set()
		
		# Start configuration: Root on stack, all tokens on buffer, empty arc set
		c = Configuration([sent.tokenlist[0]], sent.tokenlist[1:])
		
		while len(c.buffer) > 0:
		
			# Instead of looking up in the goldstandard, the GuideParser asks the Classifier to predict the next best transition
			cfeats = self.oracle.extract_feats(c, self.current_sent)
			predicted_transition = self.classifier.predict(cfeats)
			
			# If Guide predicts LA
			if predicted_transition == "LA" and len(c.stack) > 0 and c.stack[0].pos != "root_pos":
				c = self.doleftarc(c)
			
			# If Guide predicts RA
			elif predicted_transition == "RA" and len(c.stack) > 0:
				c = self.dorightarc(c)
					
			# If Guide predicts Shift
			else:
				c = self.shift(c)
					

		# When the parsing is done, annotate the sentence with the found arcs and return it
		self.current_sent.rightarcs = self.found_rarcs
		self.current_sent.leftarcs = self.found_larcs
		
		return self.current_sent

