#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Guide parsing with ArcStandard (maybe switch to ArcEager)
# in: a blind conll file (parse each sentence separately)
#	set of correct arcs A from wsj.gold?
# out: same conll file, but the heads and labels have been found
# replace Guide by multiclass classification
# Use weight vector to predict the next transition
#	Train the weight vector using Multiclass Perceptron (what are the classes?) (what is the input?)
#	Train on pairs of configurations (x i ) and transitions (y i ) given by the Guide transition sequence

from collections import deque

from Sentence import Sentence
from Sentence import Token

	
class Configuration:
	def __init__(self, st, buff, arcs):
		# Stack o: last-in first-out append_back pop_front only
		self.stack = deque(st)
		# Buffer b: first-in first-out append_front pop_front only
		self.buffer = deque(buff)
		self.created_arcs = arcs
		
	def write(self):
		out = "Stack: "
		for s in self.stack:
			out += s.write()
		out += "\n Buffer: "
		for b in self.buffer:
			out += b.write()
		out += "\n Arcs: "
		for a in self.created_arcs:
			out += a
		return out
	

class GuideParser:
	def __init__(self, classifier):
		# Set of Configurations	C
		# A config is a triple from (o, b, A)
		self.configs = list()
		# Start state cs in C
		self.cs = list()
		# Stack o: last-in first-out append_back pop_front only
		self.stack = deque()
		# Buffer b: first-in first-out append_front pop_front only
		self.buff = deque()
		# Set of correct arcs Ag
		self.arcs = list()
		
		# Sets for correct and found arcs
		self.current_sent = Sentence(list())
		self.found_rarcs = set()
		self.found_larcs = set()
		
		# The classifier to ask for predicted transitions
		self.classifier = classifier
		
		
	# Extracts the features of the current config for training
	# My smaller feature set:
	# B[0]-form form of buffer front dog
	# B[0]-pos pos of buffer front NN
	# S[0]-form form of stack top The
	# S[0]-pos pos of stack top DT
	# B[1]-pos pos of second buffer item VBZ
	# S[1]-pos pos of second stack item root POS
	# ld(B[0])-pos pos of left-most dep of buffer front JJ
	def extract_feats(self, c, transition):
		current_feats = list()
		current_feats.append("b0form_"+c.buffer[0].form)
		current_feats.append("b0pos_"+c.buffer[0].pos)
		current_feats.append("s0form_"+c.stack[0].form)
		current_feats.append("s0pos_"+c.stack[0].pos)
		token = self.current_sent.get_token_by_id(c.buffer[0].ld)
		current_feats.append("ldbopos_"+token.pos)
		
		if len(c.buffer) > 1:
			current_feats.append("b1pos_"+c.buffer[1].pos)
		else:
			current_feats.append("b1pos_NAN")
			
		if len(c.stack) > 1:
			current_feats.append("s1pos_"+c.stack[1].pos)
		else:
			current_feats.append("s1pos_NAN")
				
		return (transition, current_feats)


	# leftarc introduces an arc from the front of the buffer to the top-most token on the stack and removes the top-most token on the stack
	# BUT: top of the stack must not be root
	def doleftarc(self, c):
		#if len(c.stack) > 0:
			#self.extract_feats(c, "LA")
			
		if c.stack[0].pos != "root_pos":
			self.current_sent.tokenlist[c.buffer[0].id].head = c.stack[0].id
			print "I am ", self.current_sent.tokenlist[c.buffer[0].id].id, " and my new head is ", c.stack[0].id
			self.found_larcs.add((c.buffer[0].id, c.stack[0].id))
			del c.stack[0]
			
		return c


	# rightarc introduces an arc from the top-most token on the stack to the front of the buffer and moves the top-most token from the stack back onto the buffer
	def dorightarc(self, c):
		#if len(c.stack) > 0:
			#self.extract_feats(c, "RA")
			
		self.current_sent.tokenlist[c.stack[0].id].head = c.buffer[0].id
		print "I am ", self.current_sent.tokenlist[c.stack[0].id].id, " and my new head is ", c.buffer[0].id
		self.found_rarcs.add((c.stack[0].id, c.buffer[0].id))
		del c.buffer[0]
		c.buffer.appendleft(c.stack[0])
		del c.stack[0]
		
		return c

		
	# shift takes the first token from the front of the buffer and pushes it onto the stack
	# BUT: buffer size is at least 1 OR stack is empty
	def shift(self, c):
		#if len(c.stack) > 0:
			#self.extract_feats(c, "SH")
			
		if len(c.buffer) > 0 or len(c.stack) == 0:
			c.stack.appendleft(c.buffer[0])
			c.buffer.popleft()
			
		return c

	
	# Input: Unknown sentence
	def parse_sentence(self, sent):
		self.current_sent = sent
		print "Currrent sent: ", sent.write()
		self.found_rarcs = set()
		self.found_larcs = set()
		
		# current state/config is the start state/config
		c = Configuration([sent.tokenlist[0]], sent.tokenlist[1:], set())
		
		# while buffer is not empty
		while len(c.buffer) > 0:
			if len(c.stack) > 0:
				cfeats = self.extract_feats(c, "NA")
				predicted_transition = self.classifier.predict(cfeats)
			
				# If Guide predicts LA
				if len(c.stack) > 0 and predicted_transition == "LA":
					print "Predicted LA for config ", cfeats
					c = self.doleftarc(c)
			
				# If Guide predicts RA
				elif len(c.stack) > 0 and predicted_transition == "RA":
					print "Predicted RA for config ", cfeats
					c = self.dorightarc(c)
					
				# If Guide predicts Shift
				else:
					print "Predicted shift for config ", cfeats
					c = self.shift(c)
					
			else:
					print "shift because stack empty"
					c = self.shift(c)
				
		self.current_sent.rightarcs = self.found_rarcs
		self.current_sent.leftarcs = self.found_larcs
		
		return self.current_sent

# end: Parser

