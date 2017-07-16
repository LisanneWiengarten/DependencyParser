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
from Configuration import Configuration
	

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
		b0pos = c.buffer[0].pos
		b0form = c.buffer[0].form
		
		# Set some default values for the cases if stack and/or buffer are too sparse
		if len(c.stack) > 0:
			hs0 = self.current_sent.get_token_by_id(c.stack[0].head).pos
			lds0 = self.current_sent.get_token_by_id(c.stack[0].ld).pos
			rds0 = self.current_sent.get_token_by_id(c.stack[0].rd).pos
			s0form = c.stack[0].form
			s0pos = c.stack[0].pos	
		else:
			hs0 = "NAN"
			lds0 = "NAN"
			rds0 = "NAN"
			s0form = "NAN"
			s0pos = "NAN"
		
		if len(c.stack) > 1:
			s1pos = c.stack[1].pos
		else:
			s1pos = "NAN"
		
		if len(c.buffer) > 1:
			b1pos = c.buffer[1].pos
			b1form = c.buffer[1].form	
		else:
			b1pos = "NAN"
			b1form = "NAN"
			
		if len(c.buffer) > 2:
			b2pos = c.buffer[2].pos
			b2form = c.buffer[2].form
		else:
			b2pos = "NAN"
			b2form = "NAN"
			
		current_feats = list()
		current_feats.append("b0form_"+b0form)													# B[0]-form
		current_feats.append("b0pos_"+b0pos)													# B[0]-pos
		current_feats.append("b0form,pos_"+b0form+"_"+b0pos)									# B[0]-form,pos
		ldb0 = self.current_sent.get_token_by_id(c.buffer[0].ld)
		current_feats.append("ldb0pos_"+ldb0.pos)												# ld(B[0])-pos
		current_feats.append("s0form_"+s0form)													# S[0]-form
		current_feats.append("s0pos_"+s0pos)													# S[0]-pos
		current_feats.append("s0form,pos_"+s0form+"_"+s0pos)									# S[0]-form,pos
		current_feats.append("s0form,pos_"+s0form+"_"+s0pos+"+b0form,pos_"+b0form+"_"+b0pos)	# S[0]-form,pos+B[0]-form,pos
		current_feats.append("s0form,pos_"+s0form+"_"+s0pos+"+b0form_"+b0form)					# S[0]-form,pos+B[0]-form
		current_feats.append("s0form_"+s0form+"+b0form,pos_"+b0form+"_"+b0pos)					# S[0]-form+B[0]-form,pos
		current_feats.append("s0form,pos_"+s0form+"_"+s0pos+"+b0pos_"+b0pos)					# S[0]-form,pos+B[0]-pos
		current_feats.append("s0pos_"+s0pos+"+b0form,pos_"+b0form+"_"+b0pos)					# S[0]-pos+B[0]-form,pos
		current_feats.append("s0form_"+s0form+"+b0form_"+b0form)								# S[0]-form+B[0]-form
		current_feats.append("s0pos_"+s0pos+"+b0pos_"+b0pos)									# S[0]-pos+B[0]-pos
		current_feats.append("hs0pos_"+hs0+"+s0pos_"+s0pos+"b0pos_"+b0pos)						# h(S[0])-pos+S[0]-pos+B[0]-pos
		current_feats.append("s0pos_"+s0pos+"+lds0pos_"+lds0+"+b0pos_"+b0pos)					# S[0]-pos+ld(S[0])-pos+B[0]-pos
		current_feats.append("s0pos_"+s0pos+"+b0pos_"+b0pos+"+ldb0pos_"+ldb0.pos)				# S[0]-pos+B[0]-pos+ld(B[0])-pos
		current_feats.append("s0pos_"+s0pos+"+rds0_"+rds0+"+b0pos_"+b0pos)						# S[0]-pos+rd(S[0])-pos+B[0]-pos
		current_feats.append("b1pos_"+b1pos)													# B[1]-pos
		current_feats.append("b1form_"+b1form)													# B[1]-form
		current_feats.append("b1form,pos_"+b1form+"_"+b1pos)									# B[1]-form,pos
		current_feats.append("b0pos_"+b0pos+"+b1pos_"+b1pos)									# B[0]-pos+B[1]-pos
		current_feats.append("s0pos_"+s0pos+"+b0pos_"+b0pos+"+b1pos_"+b1pos)					# S[0]-pos+B[0]-pos+B[1]-pos
		current_feats.append("b2pos_"+b2pos)													# B[2]-pos
		current_feats.append("b2form_"+b2form)													# B[2]-form
		current_feats.append("b2form,pos_"+b2form+"_"+b2pos)									# B[2]-form,pos
		current_feats.append("b0pos_"+b0pos+"+b1pos_"+b1pos+"+b2pos_"+b2pos)					# B[0]-pos+B[1]-pos+B[2]-pos
		current_feats.append("s1pos_"+s1pos)													# S[1]-pos
				
		return (transition, current_feats)


	# leftarc introduces an arc from the front of the buffer to the top-most token on the stack and removes the top-most token on the stack
	# BUT: top of the stack must not be root
	def doleftarc(self, c):
		#if len(c.stack) > 0:
			#self.extract_feats(c, "LA")
			
		if c.stack[0].pos != "root_pos":
			self.current_sent.tokenlist[c.stack[0].id].head = c.buffer[0].id
			self.found_larcs.add((c.buffer[0].id, c.stack[0].id))
			del c.stack[0]
			
		return c


	# rightarc introduces an arc from the top-most token on the stack to the front of the buffer and moves the top-most token from the stack back onto the buffer
	def dorightarc(self, c):
		#if len(c.stack) > 0:
			#self.extract_feats(c, "RA")
			
		self.current_sent.tokenlist[c.buffer[0].id].head = c.stack[0].id
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
		self.found_rarcs = set()
		self.found_larcs = set()
		
		# current state/config is the start state/config
		c = Configuration([sent.tokenlist[0]], sent.tokenlist[1:])
		
		# while buffer is not empty
		while len(c.buffer) > 0:
			cfeats = self.extract_feats(c, "NA")
			predicted_transition = self.classifier.predict(cfeats)
			
			# If Guide predicts LA
			if predicted_transition == "LA" and len(c.stack) > 0 and c.stack[0].pos != "root_pos":
				#print "Predicted LA for config ", cfeats
				c = self.doleftarc(c)
			
			# If Guide predicts RA
			elif predicted_transition == "RA" and len(c.stack) > 0:
				#print "Predicted RA for config ", cfeats
				c = self.dorightarc(c)
					
			# If Guide predicts Shift
			else:
				#print "Predicted shift for config ", cfeats
				c = self.shift(c)
					
			#else:
					#print "shift because stack empty"
					#c = self.shift(c)
				
		self.current_sent.rightarcs = self.found_rarcs
		self.current_sent.leftarcs = self.found_larcs
		
		return self.current_sent

# end: Parser

