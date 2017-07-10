import sys
import timeit

from OracleParser import OracleParser
from GoldReader import GoldReader
from Classifier import Classifier
from TestReader import TestReader
from GuideParser import GuideParser

def main(trainfile, testfile):
	start = timeit.default_timer()
	
	### TRAINING ###
	# Read in train data
	goldstandard = GoldReader(trainfile)
	
	# Parse all sentences in golstandard
	oracleparser = OracleParser()
	for s in goldstandard.goldlist:
		oracleparser.parse_sentence(s)
		
		
	# Train the classifier
	classifier = Classifier(oracleparser.raw_feats, oracleparser.unique_feats, 300, 7)
	classifier.train()
	#classifier.save_model("model")
	
	
	#c2 = classifier.load_model("model")
	
	### TESTING ###
	# Use the classifier to test new data
	testsents = TestReader(testfile)
	parser = GuideParser(classifier)
	for s in testsents.sentlist:
		parsed = parser.parse_sentence(s)
		print parsed.write()
		print "LAs: ", parsed.leftarcs
		print "RAs: ", parsed.rightarcs
		
		
	stop = timeit.default_timer()
	print stop - start


		
if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "Error: Train and test file needed"
	else:
		main(sys.argv[1], sys.argv[2])