import numpy as np
import argparse
import pdb
import os

from rsvm import RankSVM
import pickle

def argparser():

	parser = argparse.ArgumentParser(description='Train RANK SVM')
	parser.add_argument('--dataset', dest='dataset',
                      help='training dataset location',
                      default='None', type=str)

	return parser.parse_args()

def main():

	args = argparser()

	data = np.genfromtxt(args.dataset,delimiter=',')

	# feature columns
	X = data[:,:-2]
	y = data[:,-2:]

	# train test split
	X_train = X[:-100,:]
	y_train = y[:-100,:]

	X_test = X[-100:,:]
	y_test = y[-100:,:]

	model = RankSVM().fit(X_train, y_train)

	print("model score on test: ",model.score(X_test,y_test))

	with open('sk_model.pkl','wb') as f:
		pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)

	


if __name__ == "__main__":
	main()

