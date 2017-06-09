# We have a pair <c, t> where:
    # c in C = a configuration (where C - all possible configurations)
    # t in T = a transition (where T - all possible transitions)
# We build a mapping from pairs <c, t> to a high-dimensional feature vector
    # phi(c, t) : CxT->Rd
    # phi is the feature vector

# Features to (possibly) extract:
# B[0] form, lemma, cpos, fpos, feat
# B[1] form, fpos
# B[2] fpos
# B[3] fpos
# ld(B[0]) dep
# rd(B[0]) dep
# S[0] form, lemma, cpos, fpos, feat
# S[1] fpos
# ld(S[0]) dep
# rd(S[0]) dep

# My smaller feature set:
# B[0]-form form of buffer front dog
# B[0]-pos pos of buffer front NN
# S[0]-form form of stack top The
# S[0]-pos pos of stack top DT
# B[1]-pos pos of second buffer item VBZ
# S[1]-pos pos of second stack item root POS
# ld(B[0])-pos pos of left-most dep of buffer front JJ

class Classifier:
	def __init__(self, rawfeats, ufeats):
		# List of tuples (transition, list of feats (strings!))
		self.raw_feats = rawfeats
		# All features that have been seen during training (strings)
		self.unique_feats = ufeats
		
		# List of tuples (transition, longer list of feats (0/1!))
		self.feats = list()
		self.transform()
		
		
	# Transform from short feature vector (string) to the long feat vector of 0s and 1s
	def transform(self):
		# For every item in the training data,
		# check whether it has a certain feature (1) or not (0)
		for item in self.raw_feats:
			current = list()
			for i in range(len(self.unique_feats)-1):
				if self.unique_feats[i] in item[1]:
					current.append(1)
				else:
					current.append(0)
					
			# Finally, for every train item, we get a tuple (transition, long vector (ints))
			self.feats.append((item[0], current))
			
				

# In: training set D with pairs of input objects (train instance) and class labels (operation)
# Out: weight matrix W
# We use vector w in Rd which holds a weight for each feature
# We need to predict an operation to do based on the current given config

# Input: D = {(x1,y1), (x2,y2), ... (xN,yN)} bzw. (phi(xi), yi)
# N = no. of train examples
# Input: Number of iterations T
# Output: Weight matrix W
# Instantiate all zeros to every class label in W (Create a weight vector wy for every class y)
# W[y] = ->0 for every y in Y
# do all given iterations T
# for t in 1..T do
    # go through all train examples
    # for i in 1..N do
        # 
        # ^yi = argmaxy (W[y]*phi(xi))
        # if ^yi != yi then
            # W[yi] = W[yi]+phi(xi)
            # W[^yi] = W[^yi]-phi(xi)
# return W
# W is in fact a matrix of class vectors
# ^yi is the highest-scoring tree according to w
# yi is the gold tree from the treebank


# OUTPUT: t = argmax w * phi(c,t)
             # t in T
# (here: T != no. iters)


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
