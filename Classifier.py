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
from operator import sub, add

# Trains a model for dependency parsing #
# IN: The 'raw' features, the unique features and the number of iterations
# The trained model can predict the next best transition (SH, LA or RA) given a configuration
class Classifier:
	def __init__(self, rawfeats, ufeats, iterations):
	
		# List of tuples (transition, list of feats (str))
		self.raw_feats = rawfeats
		
		# All features that have been seen in goldstandard (str)
		self.unique_feats = ufeats
		self.num_ufeats = len(self.unique_feats)
		
		self.classes = ["RA", "SH", "LA"]
		# Stores the weight vector for each class
		self.weightmatrix = dict()
		
		# Iterations of training
		self.iterations = iterations
		
		# List of tuples (transition, list of feats)
		# Instead of the very long vector with many 0's and a few 1's,
		# we only save the index in the unique_feats/weightmatrix where the found features are
		self.feats = list()
		self.transform()
		
		
	# Transform from feature vector (str) to vector of ints (indices from unique_feats/weightmatrix) #
	def transform(self):
		# For every item in the training data,
		# check whether it has a certain feature (1) or not (0)
		counter = 0
		length = len(self.raw_feats)
		for item in self.raw_feats:
			if (counter % 1000 == 0):
				print "Transforming feat ", counter, " of ", length
			counter += 1
			current = list()
			for i in range(len(item[1])):
				if item[1][i] in self.unique_feats:
					current.append(self.unique_feats.index(item[1][i]))
					
			# Finally, for every train item, we get a tuple (transition, vector(with numbers where a 1 would be in the full vector))
			self.feats.append((item[0], current))
			
			
	# Transform from feature vector (string) to vector of ints (indices from unique_feats/weightmatrix) #
	def transform_one(self, raw_feat):
		# For every item in the training data,
		# check whether it has a certain feature (1) or not (0)
		# If a feature was seen in the goldstandard, we can simply look it up,
		# otherwise, we need to put in a higher number (index)
		current = list()
		for i in range(len(raw_feat[1])):
			if raw_feat[1][i] in self.unique_feats:
				current.append(self.unique_feats.index(raw_feat[1][i]))
			else:
				self.num_ufeats += 1
				current.append(self.num_ufeats)
					
		return sorted(current)
	
	
	# In: training set with pairs of input objects (train instance) and class labels (operation)
	# Out: Weight matrix W	
	def train(self):
		total, correct = 0, 0
	
		# Instantiate all zeros to every class label in W (Create a weight vector for every class)
		# Each vector has the length of unique feats found in the goldstandard
		for c in self.classes:	
			self.weightmatrix[c] = [0] * len(self.unique_feats)
		

		for i in range(self.iterations):
			print "Train iteration ", i
			
			# For each given training example
			for (category, feature_list) in self.feats:
				total += 1	
			
				# Initialize current_max value and predicted class
				current_max, predicted_class = 0, self.classes[0]
				
				# For each class, instead of the dot product of two very long vectors,
				# We only sum all the values from the weight vector at the indices given by the training example (same result)
				for c in self.classes:
				
					# Take all the numbers in the feature_list as indices in weight vector: Get the vals at these indices and add them
					current_activation = 0
					for i in feature_list:
						current_activation += self.weightmatrix[c][i]
					
					# Predict the class with the highest activation
					if current_activation >= current_max:
						current_max = current_activation
						predicted_class = c
				
				
				# If we did not predict the correct category, update the weight matrix
				if (category != predicted_class):
					# For updating, get all vals from feature vector as indices in the weight vector
					# Just add 1 at each of these indices for the correct class, and subtract 1 for the falsely predicted class
					for p in feature_list:
						self.weightmatrix[category][p] += 1
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
	
		# Transform the current feature vector into the correct representation with indices
		transformed = self.transform_one(feature_list)

        # Initialize current_max value and predicted class
		current_max, predicted_class = 0, self.classes[0]
		

		# For each class, instead of the dot product of two very long vectors,
		# We only sum all the values from the weight vector at the indices given by the training example (same result)
		for c in self.classes:
			
			current_activation = 0
			for i in transformed:
				# The feature can only be considered if it was seen during training
				if i < len(self.weightmatrix[c])-1:
					current_activation += self.weightmatrix[c][i]
			
			
			# Predict the class with the highest activation
			if current_activation >= current_max:
				current_max = current_activation
				predicted_class = c
				
		return predicted_class
		
		
		
	# Save the trained model to a pickle file #
	def save_model(self, model_name):
		with open(model_name, 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
			
			