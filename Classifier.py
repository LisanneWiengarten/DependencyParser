import numpy
import pickle
from operator import sub, add

# My feature set:
# B[0]-form form of buffer front dog
# B[0]-pos pos of buffer front NN
# S[0]-form form of stack top The
# S[0]-pos pos of stack top DT
# B[1]-pos pos of second buffer item VBZ
# S[1]-pos pos of second stack item root POS
# ld(B[0])-pos pos of left-most dep of buffer front JJ


# Trains a model for dependency parsing
# IN: The 'raw' features, the unique features, the number of iterations and the number of features that were extracted
# The trained model can predict the next best transition (shift, LA or RA) given a configuration
class Classifier:
	def __init__(self, rawfeats, ufeats, iterations):
		# List of tuples (transition, list of feats (strings!))
		self.raw_feats = rawfeats
		# All features that have been seen in goldstandard (strings)
		self.unique_feats = ufeats
		
		self.classes = ["RA", "SH", "LA"]
		# Stores the weight vector for each class
		self.weightmatrix = dict()
		
		# Iterations of training
		self.iterations = iterations
		
		
		# List of tuples (transition, longer list of feats (0/1!))
		self.feats = list()
		self.transform()
		
		
	# Transform from short feature vector (string) to the vector with len(no(feats))
	def transform(self):
		# For every item in the training data,
		# check whether it has a certain feature (1) or not (0)
		for item in self.raw_feats:
			current = list()
			for i in range(len(item[1])):
				if item[1][i] in self.unique_feats:
					current.append(self.unique_feats.index(item[1][i]))
					
			# Finally, for every train item, we get a tuple (transition, vector(with numbers where a 1 would be in the long vector))
			self.feats.append((item[0], current))
			
			
	# Transform from short feature vector (string) to the vector with len(no(feats))
	def transform_one(self, raw_feat):
		# For every item in the training data,
		# check whether it has a certain feature (1) or not (0)
		current = list()
		for i in range(len(raw_feat[1])):
			if raw_feat[1][i] in self.unique_feats:
				current.append(self.unique_feats.index(raw_feat[1][i]))
			else:
				self.unique_feats.append(raw_feat[1][i])
				current.append(self.unique_feats.index(raw_feat[1][i]))
					
			# Finally, for every train item, we get a tuple (transition, vector(with numbers where a 1 would be in the long vector))
			#self.feats.append((item[0], current))
		return sorted(current)
	
	
	# In: training set D with pairs of input objects (train instance) and class labels (operation)
	# Input: D = {(x1,y1), (x2,y2), ... (xN,yN)} bzw. (phi(xi), yi)
	# N = no. of train examples
	# Input: Number of iterations T
	# Out: weight matrix W
	# Output: Weight matrix W
	# We use vector w in Rd which holds a weight for each feature
	# We need to predict an operation to do based on the current given config		
	def train(self):
		total, correct = 0, 0
	
		# Instantiate all zeros to every class label in W (Create a weight vector wy for every class y)
		# W[y] = ->0 for every y in Y
		for c in self.classes:	
			self.weightmatrix[c] = [0] * len(self.unique_feats)
		
		# durch alle Iterationen gehen (feste val)
		# for _ in xrange(self.iterations):
		# for t in 1..T do
		for i in range(self.iterations):
			# durch alle train instances gehen (self.feats)
			# for category, feature_dict in self.train_set:
			# for i in 1..N do
			for (category, feature_list) in self.feats:
				total += 1	
			
				# Initialize current_max value and predicted class
				current_max, predicted_class = 0, self.classes[0]
				
				# Multi-Class Decision Rule:
				# For each class, get dot product of the current feat vector and the weight vector of the current class
				# If the calculated val is higher than the current max, update the max and the class we currently predict
				for c in self.classes:
					# Take all the numbers in the feature_list as indices in weight vector:
						# Get the vals at these indices and add them
					current_activation = 0
					for i in feature_list:
						current_activation += self.weightmatrix[c][i]
					#current_activation = numpy.dot(feature_list, self.weightmatrix[c])
					#print "For class ", c, " we found current_activation ", current_activation, " - current_max is ", current_max
					if current_activation >= current_max:
						# ^yi = argmaxy (W[y]*phi(xi))
						current_max = current_activation
						predicted_class = c
				
				#print "Category: ", category, "Featlist: ", feature_list, sum(feature_list)
				#print "Predicited: ", predicted_class
				# Update Rule:
				# If we did not predict the correct category, update the weight matrix
				# if ^yi != yi then
					# W[yi] = W[yi]+phi(xi)
					# W[^yi] = W[^yi]-phi(xi)
				if (category != predicted_class):
					# For updating, get all vals from feature_list as indices in weight vector
					# Just add/subtract 1 at each of these indices
					for p in feature_list:
						self.weightmatrix[category][p] += 1
						self.weightmatrix[predicted_class][p] -= 1
					#self.weightmatrix[category] = map(add, self.weightmatrix[category], feature_list)
					#self.weightmatrix[predicted_class] = map(sub, self.weightmatrix[predicted_class], feature_list)
				
				else:
					correct += 1
			
			acc = (float(correct)/float(total))*100
		print "Acc: ", acc
		for c in self.classes:
			print c, sum(self.weightmatrix[c])

	
	# Predicts the best transition given the features of a config
	def predict(self, feature_list):
		transformed = self.transform_one(feature_list)
		#print "Classifier got ", feature_list, " and transformed into ", transformed

        # Initialize current_max value and predicted class
		current_max, predicted_class = 0, self.classes[0]
		

        # Multi-Class Decision Rule:
		# For each class, get dot product of the current feat vector and the weight vector of the current class
		# If the calculated val is higher than the current max, update the max and the class we currently predict
		for c in self.classes:
			# Reshape weightmatrix because number of unique feats becomes bigger during testing
			#self.weightmatrix[c].extend([0] * (len(self.unique_feats) - len(self.weightmatrix[c])))
			current_activation = 0
			for i in transformed:
				if i < len(self.weightmatrix[c])-1:
				#print "I am adding ", current_activation, "+", self.weightmatrix[c][i]
					current_activation += self.weightmatrix[c][i]
			# current_activation = numpy.dot(transformed, self.weightmatrix[c])
			#print "For class ", c, " we found current_activation ", current_activation, " - current_max is ", current_max
			if current_activation >= current_max:
				current_max = current_activation
				predicted_class = c
				
		return predicted_class
		
		
		
	# Save the trained model to a pickle file
	def save_model(self, model_name):
		with open(model_name, 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
			
			
# During each iteration of training, the data (formatted as a feature vector)
# is read in, and the dot product is taken with each unique weight vector
# (which are all initially set to 0). The class that yields the highest product
# is the class to which the data belongs. In the case this class is the correct
# value (matches with the actual category to which the data belongs),
# nothing happens, and the next data point is read in. However, in the case that
# the predicted value is wrong, the weight vectors are corrected as follows:
# The feature vector is subtracted from the predicted weight vector,
# and added to the actual (correct) weight vector. This makes sense, as we want
# to reject the wrong answer, and accept the correct one.
# After the final iteration, the final weight vectors should be somewhat stable
# (it is of importance to note that unlike the assumptions of the binary
# perceptron, there is no guarantee the multi-class perceptron will reach
# a steady state), and the classifier will be ready to be put to use.

# We have a pair <c, t> where:
    # c in C = a configuration (where C - all possible configurations)
    # t in T = a transition (where T - all possible transitions)
# We build a mapping from pairs <c, t> to a high-dimensional feature vector
    # phi(c, t) : CxT->Rd
    # phi is the feature vector