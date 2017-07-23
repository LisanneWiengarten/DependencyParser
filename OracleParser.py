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
	
# Does Oracle parsing with ArcStandard #
# In: The preprocessed list of sentences from a goldstandard
# Does parsing on this goldstandard ro extract the features needed for training a classifier
class OracleParser:
	def __init__(self):
		
		# Sets for correct and found arcs
		self.correct_sent = Sentence(list())
		self.correct_ras = set()
		self.correct_las = set()
		self.found_rarcs = set()
		self.found_larcs = set()
		
		self.extracted_feats = list()
		self.unique_feats = list()
		
		
	# Extracts the features of the current config for training
	# My feature set:
	# B[0]-form form of buffer front dog
	# B[0]-pos pos of buffer front NN
	# S[0]-form form of stack top The
	# S[0]-pos pos of stack top DT
	# B[1]-pos pos of second buffer item VBZ
	# B[1]-form form of second buffer item
	# B[2]-form
	# B[2]-pos
	# S[1]-pos pos of second stack item root POS
	# ld(B[0])-pos pos of left-most dep of buffer front JJ
	# S[0]-form,pos
	# B[0]-form,pos
	# B[1]-form,pos
	# B[2]-form,pos
	
	# S[0]-form,pos+B[0]-form,pos
	# S[0]-form,pos+B[0]-form
	# S[0]-form+B[0]-form,pos
	# S[0]-form,pos+B[0]-pos
	# S[0]-pos+B[0]-form,pos
	# S[0]-form+B[0]-form
	# S[0]-pos+B[0]-pos
	# B[0]-pos+B[1]-pos
	
	# B[0]-pos+B[1]-pos+B[2]-pos
	# S[0]-pos+B[0]-pos+B[1]-pos
	# h(S[0])-pos+S[0]-pos+B[0]-pos
	# S[0]-pos+ld(S[0])-pos+B[0]-pos
	# S[0]-pos+B[0]-pos+ld(B[0])-pos
	# S[0]-pos+rd(S[0])-pos+B[0]-pos
	
	def extract_feats(self, c, transition):
		current_feat = list()
	
		b0pos = c.buffer[0].pos
		b0form = c.buffer[0].form
		ldb0 = self.correct_sent.get_token_by_id(c.buffer[0].ld)
		
		# Set some default values for the cases if stack and/or buffer are too short
		
		# If the stack is empty, the form, pos, head, ld and rd from the front of the stack cannot be recovered
		if len(c.stack) > 0:
			hs0 = self.correct_sent.get_token_by_id(c.stack[0].head).pos
			lds0 = self.correct_sent.get_token_by_id(c.stack[0].ld).pos
			rds0 = self.correct_sent.get_token_by_id(c.stack[0].rd).pos
			s0form = c.stack[0].form
			s0pos = c.stack[0].pos	
		else:
			hs0 = "NAN"
			lds0 = "NAN"
			rds0 = "NAN"
			s0form = "NAN"
			s0pos = "NAN"
		
		# If the stack contains only one item, the pos from the second item cannot be recovered
		if len(c.stack) > 1:
			s1pos = c.stack[1].pos
		else:
			s1pos = "NAN"
		
		# If the buffer contains only one item, the pos and form from the second item cannot be recovered
		if len(c.buffer) > 1:
			b1pos = c.buffer[1].pos
			b1form = c.buffer[1].form	
		else:
			b1pos = "NAN"
			b1form = "NAN"
			
		# If the buffer contains only two items, the pos and form from the third item cannot be recovered
		if len(c.buffer) > 2:
			b2pos = c.buffer[2].pos
			b2form = c.buffer[2].form
		else:
			b2pos = "NAN"
			b2form = "NAN"
		
		feature_set = ["b0form_"+b0form, 												# B[0]-form
						"b0pos_"+b0pos, 												# B[0]-pos
						"b0form,pos_"+b0form+"_"+b0pos,									# B[0]-form,pos
						"ldb0pos_"+ldb0.pos,											# ld(B[0])-pos
						"s0form_"+s0form,												# S[0]-form
						"s0pos_"+s0pos,													# S[0]-pos
						"s0form,pos_"+s0form+"_"+s0pos,									# S[0]-form,pos
						"s0form,pos_"+s0form+"_"+s0pos+"+b0form,pos_"+b0form+"_"+b0pos,	# S[0]-form,pos+B[0]-form,pos
						"s0form,pos_"+s0form+"_"+s0pos+"+b0form_"+b0form,				# S[0]-form,pos+B[0]-form
						"s0form_"+s0form+"+b0form,pos_"+b0form+"_"+b0pos,				# S[0]-form+B[0]-form,pos
						"s0form,pos_"+s0form+"_"+s0pos+"+b0pos_"+b0pos,					# S[0]-form,pos+B[0]-pos
						"s0pos_"+s0pos+"+b0form,pos_"+b0form+"_"+b0pos,					# S[0]-pos+B[0]-form,pos		
						"s0form,pos_"+s0form+"_"+s0pos+"+b0pos_"+b0pos,					# S[0]-form,pos+B[0]-pos
						"s0pos_"+s0pos+"+b0form,pos_"+b0form+"_"+b0pos,					# S[0]-pos+B[0]-form,pos
						"s0form_"+s0form+"+b0form_"+b0form,								# S[0]-form+B[0]-form
						"s0pos_"+s0pos+"+b0pos_"+b0pos,									# S[0]-pos+B[0]-pos
						"hs0pos_"+hs0+"+s0pos_"+s0pos+"b0pos_"+b0pos,					# h(S[0],-pos+S[0]-pos+B[0]-pos
						"s0pos_"+s0pos+"+lds0pos_"+lds0+"+b0pos_"+b0pos,				# S[0]-pos+ld(S[0],-pos+B[0]-pos
						"s0pos_"+s0pos+"+b0pos_"+b0pos+"+ldb0pos_"+ldb0.pos,			# S[0]-pos+B[0]-pos+ld(B[0],-pos
						"s0pos_"+s0pos+"+rds0_"+rds0+"+b0pos_"+b0pos,					# S[0]-pos+rd(S[0],-pos+B[0]-pos
						"b1pos_"+b1pos,													# B[1]-pos
						"b1form_"+b1form,												# B[1]-form
						"b1form,pos_"+b1form+"_"+b1pos,									# B[1]-form,pos
						"b0pos_"+b0pos+"+b1pos_"+b1pos,									# B[0]-pos+B[1]-pos
						"s0pos_"+s0pos+"+b0pos_"+b0pos+"+b1pos_"+b1pos,					# S[0]-pos+B[0]-pos+B[1]-pos
						"b2pos_"+b2pos,													# B[2]-pos
						"b2form_"+b2form,												# B[2]-form
						"b2form,pos_"+b2form+"_"+b2pos,									# B[2]-form,pos
						"b0pos_"+b0pos+"+b1pos_"+b1pos+"+b2pos_"+b2pos,					# B[0]-pos+B[1]-pos+B[2]-pos
						"s1pos_"+s1pos]													# S[1]-pos
		
		for feat in feature_set:
			# If this feat is already in unique_feats, append its index to the current_featlist
			if feat in self.unique_feats:
				current_feat.append(self.unique_feats.index(feat))
		
			# If this feat is not yet in unique_feats, append it
			else:
				self.unique_feats.append(feat)
				current_feat.append(len(self.unique_feats)-1)
					
		self.extracted_feats.append((transition, current_feat))
		


	# Creates a leftarc from the front of the buffer to the top-most token on the stack #
	# And removes the top-most token on the stack
	def doleftarc(self, c):	
		self.correct_sent.tokenlist[c.stack[0].id].head = c.buffer[0].id
		self.found_larcs.add((c.buffer[0].id, c.stack[0].id))
		del c.stack[0]
			
		return c


	# Creates a right from the top-most token on the stack to the front of the buffer #
	# And moves the top-most token from the stack back onto the buffer
	def dorightarc(self, c):	
		self.correct_sent.tokenlist[c.buffer[0].id].head = c.stack[0].id
		self.found_rarcs.add((c.stack[0].id, c.buffer[0].id))
		del c.buffer[0]
		c.buffer.appendleft(c.stack[0])
		del c.stack[0]
		
		return c

		
	# Shift takes the first token from the front of the buffer and pushes it onto the stack #
	def shift(self, c):
		c.stack.appendleft(c.buffer[0])
		c.buffer.popleft()
			
		return c


	# Checks whether all dependents for a specific token have already been found #
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


	# Given a configuration and the set of correct arcs, determines whether a leftarc might be build #
	def canleftarc(self, c):
		if c.stack[0].pos == "root_pos":
			return False
		# If there is a leftarc from the first item in b to the first item in the goldstandard, return true
		if (c.stack[0].id, c.buffer[0].id) in self.correct_las or (c.buffer[0].id, c.stack[0].id) in self.correct_las:
			return True
		
		return False

		
	# Given a configuration and the set of correct arcs, determines whether a rightarc might be build #
	def canrightarc(self, c, gold):
		# If there is a rightarc from the first item in the goldstandard to the first item in b 
		# AND this first item in b already has all its children, then return true
		if (c.stack[0].id, c.buffer[0].id) in self.correct_ras and self.has_all_children(c.buffer[0], gold):
			return True
		elif (c.buffer[0].id, c.stack[0].id) in self.correct_ras and self.has_all_children(c.buffer[0], gold):
			return True
			
		return False

	
	# Does the actual oracle parsing #
	# In: Correctly annotated sentence from goldstandard
	# Performs all parsing steps on this sentence so the features can be extracted for training
	def parse_sentence(self, gold):
		self.correct_sent = gold
		self.correct_ras = gold.rightarcs
		self.correct_las = gold.leftarcs
		self.found_rarcs = set()
		self.found_larcs = set()
		
		# Start configuration: Root on stack, all tokens on buffer, empty arc set
		c = Configuration([gold.tokenlist[0]], gold.tokenlist[1:])
		
		# While buffer is not empty
		while len(c.buffer) > 0:
		
			# If it's possible to form a leftarc with the current configuration and the goldstandard
			# Then build this leftarc and extract the features
			if len(c.stack) > 0 and self.canleftarc(c):
				self.extract_feats(c, "LA")
				c = self.doleftarc(c)
			
			# Else if it's possible to form a rightarc with the current configuration and the goldstandard
			# Then build this rightarc and extract the features
			elif len(c.stack) > 0 and self.canrightarc(c, gold):
				self.extract_feats(c, "RA")
				c = self.dorightarc(c)
			 
			# Else perform a shift and extract the features
			else:
				self.extract_feats(c, "SH")
				c = self.shift(c)

		return 1
