"""
Lisanne Wiengarten
Matriculation no. 3249897
Statistical Dependency Parsing
IMS, SuSe 17
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import pickle
import random
from operator import sub, add

# Trains a model for dependency parsing #
# IN: The 'raw' features, the unique features and the number of iterations
# The trained model can predict the next best transition (SH, LA or RA) given a configuration
class Classifier:
	def __init__(self, extractedfeats, ufeats, iterations):
		
		# All features that have been seen in goldstandard (str)
		self.unique_feats = ufeats
		self.num_ufeats = len(self.unique_feats)
		
		self.classes = ["RA", "LA", "SH"]
		# Stores the weight vector for each class
		self.weightmatrix = dict()
		
		# Iterations of training
		self.iterations = iterations
		
		# List of tuples (transition, list of feats)
		# Instead of the very long vector with many 0's and a few 1's,
		# we only save the index in the unique_feats/weightmatrix where the found features are
		self.feats = extractedfeats
		
	
	# In: training set with pairs of input objects (train instance) and class labels (operation)
	# Out: Weight matrix W	
	def train(self):
		total, correct = 0, 0
	
		# Instantiate all zeros to every class label in W (Create a weight vector for every class)
		# Each vector has the length of unique feats found in the goldstandard
		for c in self.classes:	
			self.weightmatrix[c] = [0] * len(self.unique_feats)
			

		for i in range(self.iterations):
			
			# For each given training example
			for item in self.feats:
				total += 1
				
				correct_class = item[0]
				feature_list = item[1]
			
				# Initialize current_weights dict
				current_weights = {"RA":0,"LA":0,"SH":0}
				
				# For each class, instead of the dot product of two very long vectors,
				# We only sum all the values from the weight vector at the indices given by the training example (same result)
				for c in self.weightmatrix:
				
					# Take all the numbers in the feature_list as indices in weight vector: Get the vals at these indices and add them
					for i in feature_list:
						current_weights[c] += self.weightmatrix[c][i]
		
				# Predict the class with the highest activation
				predicted_class = max(current_weights, key=current_weights.get)
				
				# If we did not predict the correct class, update the weight matrix
				if (correct_class != predicted_class):
					# For updating, get all vals from feature vector as indices in the weight vector
					# Just add 1 at each of these indices for the correct class, and subtract 1 for the falsely predicted class
					for p in feature_list:
						self.weightmatrix[correct_class][p] += 1
						self.weightmatrix[predicted_class][p] -= 1
				
				else:
					correct += 1
		
		
			acc = (float(correct)/float(total))*100	
			# Print final accuracy of training	
			print "Acc: ", acc
			for c in self.classes:
				print c, sum(self.weightmatrix[c])

	
	# Predicts the best transition given the features of a config #
	def predict(self, feature_list):

        # Initialize current_weights dict
		current_weights = {"RA":0,"LA":0,"SH":0}
		
		# For each class, instead of the dot product of two very long vectors,
		# We only sum all the values from the weight vector at the indices given by the training example (same result)
		for c in self.weightmatrix:
			for i in feature_list:
				# The feature can only be considered if it was seen during training
				if i < len(self.weightmatrix[c])-1:
					current_weights[c] += self.weightmatrix[c][i]

		# Predict the class with the highest activation
		predicted_class = max(current_weights, key=current_weights.get)
				
		return predicted_class
		
		
		
	# Save the trained model to a pickle file #
	def save_model(self, model_name):
		with open(model_name, 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
			
			