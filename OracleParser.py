#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Oracle parsing with ArcStandard (maybe switch to ArcEager)
# in: a blind conll file (parse each sentence separately)
#	set of correct arcs A from wsj.gold?
# out: same conll file, but the heads and labels have been found
# replace oracle by multiclass classification
# Use weight vector to predict the next transition
#	Train the weight vector using Multiclass Perceptron (what are the classes?) (what is the input?)
#	Train on pairs of configurations (x i ) and transitions (y i ) given by the oracle transition sequence

from collections import deque

from Sentence import Sentence
from Sentence import Token

	
class Configuration:
	def __init__(self, st, buff, arcs):
		# Stack o: last-in first-out append_back pop_front only
		self.stack = deque(st)
		# Buffer b: first-in first-out append_front pop_front only
		self.buffer = deque(buff)
		
	def write(self):
		out = "Stack: "
		for s in self.stack:
			out += s.write()
		out += "\n Buffer: "
		for b in self.buffer:
			out += b.write()
		return out
	

class OracleParser:
	def __init__(self):
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
		self.correct_sent = Sentence(list())
		self.correct_ras = set()
		self.correct_las = set()
		self.found_rarcs = set()
		self.found_larcs = set()
		
		self.raw_feats = list()
		self.unique_feats = list()
		
		
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
		token = self.correct_sent.get_token_by_id(c.buffer[0].ld)
		current_feats.append("ldbopos_"+token.pos)
		
		if len(c.buffer) > 1:
			current_feats.append("b1pos_"+c.buffer[1].pos)
		else:
			current_feats.append("b1pos_NAN")
			
		if len(c.stack) > 1:
			current_feats.append("s1pos_"+c.stack[1].pos)
		else:
			current_feats.append("s1pos_NAN")
		
		self.raw_feats.append((transition, current_feats))
		for item in current_feats:
			if item not in self.unique_feats:
				self.unique_feats.append(item)


	# leftarc introduces an arc from the front of the buffer to the top-most token on the stack and removes the top-most token on the stack
	# BUT: top of the stack must not be root
	def doleftarc(self, c):
		if len(c.stack) > 0:
			self.extract_feats(c, "LA")
			
		if c.stack[0].pos != "root_pos":
			self.correct_sent.tokenlist[c.stack[0].id].head = c.buffer[0].id
			self.found_larcs.add((c.buffer[0].id, c.stack[0].id))
			del c.stack[0]
			
		return c


	# rightarc introduces an arc from the top-most token on the stack to the front of the buffer and moves the top-most token from the stack back onto the buffer
	def dorightarc(self, c):
		if len(c.stack) > 0:
			self.extract_feats(c, "RA")
			
		self.correct_sent.tokenlist[c.buffer[0].id].head = c.stack[0].id
		self.found_rarcs.add((c.stack[0].id, c.buffer[0].id))
		del c.buffer[0]
		c.buffer.appendleft(c.stack[0])
		del c.stack[0]
		
		return c

		
	# shift takes the first token from the front of the buffer and pushes it onto the stack
	# BUT: buffer size is at least 1 OR stack is empty
	def shift(self, c):
		if len(c.stack) > 0:
			self.extract_feats(c, "SH")
			
		if len(c.buffer) > 0 or len(c.stack) == 0:
			c.stack.appendleft(c.buffer[0])
			c.buffer.popleft()
			
		return c

# !! Instead of canla and canra, we ask the classifier which operation it predicts !!

	# Checks whether all dependents for a specific token have already been found
	def has_all_children(self, current_token, gold):
		# Find all dependents of this token
		children = list()
		# If a token's head is the same number as the current token's id, this is a child
		for token in gold.tokenlist:
			if int(token.head) == current_token.id:
				children.append(token)
		
		# Check whether we already have a LA or RA from this head to each of its dependents
		for child in children:
			if (child.id, current_token.id) not in self.found_rarcs and (child.id, current_token.id) not in self.found_larcs and (current_token.id, child.id) not in self.found_rarcs and (current_token.id, child.id) not in self.found_larcs:
				return False
		
		return True


	# Given a state/config and the set of correct arcs, determine whether a leftarc might be build	
	def canleftarc(self, c):
		if c.stack[0].pos == "root_pos":
			return False
		# if there is a leftarc from the first item in b to the first item in o, return true
		if (c.stack[0].id, c.buffer[0].id) in self.correct_las or (c.buffer[0].id, c.stack[0].id) in self.correct_las:
			return True
		
		return False

		
	# Given a state/config and the set of correct arcs, determine whether a rightarc might be build 	
	def canrightarc(self, c, gold):
		# if there is a rightarc from the first item in o to the first item in b AND this first item in b already has all its children then return true
		if (c.stack[0].id, c.buffer[0].id) in self.correct_ras and self.has_all_children(c.buffer[0], gold):
			return True
		elif (c.buffer[0].id, c.stack[0].id) in self.correct_ras and self.has_all_children(c.buffer[0], gold):
			return True
			
		return False

	
	# Input: Correctly parsed sentence
	def parse_sentence(self, gold):
		self.correct_sent = gold
		self.correct_ras = gold.rightarcs
		self.correct_las = gold.leftarcs
		self.found_rarcs = set()
		self.found_larcs = set()
		
		# current state/config is the start state/config
		c = Configuration([gold.tokenlist[0]], gold.tokenlist[1:], set())
		
		# while buffer is not empty
		while len(c.buffer) > 0:
			# if it's possible to form a leftarc with the current state/config and A then build this leftarc
			if len(c.stack) > 0 and self.canleftarc(c):
				c = self.doleftarc(c)
			
			# else if it's possible to form a rightarc with the current state/config and A then build this rightarc
			elif len(c.stack) > 0 and self.canrightarc(c, gold):
				c = self.dorightarc(c)
			 
			else:
				# perform a shift
				# the current state is now the new config from the new transition
				c = self.shift(c)

		return 1

# end: Parser

