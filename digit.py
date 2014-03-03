import os
import csv 
import cPickle
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from kaggle import *
from multiprocessing import Pool
from skimage.transform import resize
from pilutil import imresize

class DigitData():

	def __init__(self, trainfile, testfile, sampleresult):

		self.n_dim 		= 0
		self.n_class 	= 0
		self.n_train	= 0
		self.n_test		= 0
		self.savefile	= 'result.csv'
		self.header 	= self.get_header(sampleresult)
		self.data_trn	= self.load_data(trainfile, 'train')
		self.data_tst 	= self.load_data(testfile, 'test')

	def get_header(self, samplefile):

		f1 = open(samplefile, 'rb')
		content = csv.reader(f1)
		raw_data = []
		for r in content:
			raw_data.append(r)
		f1.close()
		header = raw_data[0]
		return header  

	def load_data(self, filename, type):

		f1 = open(filename, 'rb')
		content = csv.reader(f1)
		data = []
		for r in content:
			data.append(r)
		f1.close()

		if type is 'train':
			
			X = np.asarray(data)[1:, 1:].astype(np.uint8)
			Y = np.asarray(data)[1:, 0].astype(np.int)
			return X, Y 

		elif type is 'test':
			
			X = np.asarray(data)[1:, :].astype(np.uint8)
		
		else:

			raise NameError('Unknown data type')


	def save_data(self, result, filename=None):
		'''
			save result into a csv file
		'''
		if filename is None:
			filename = self.savefile

		data = []
		data.append(self.header)

		for i in range(len(result)):
			data.append([str(i+1), str(result[i][0])])

		if os.path.isfile(filename):
			os.remove(filename)

		f = open(filename, 'wb')
		c = csv.writer(f)
		c.writerows(data)
		f.close()


	def extract_feat(self, imgs, ftype):
		'''
			extract features
		'''
		if ftype is 'hog':
			
			from skimage.feature import hog
			L = np.sqrt(len(imgs[0])).astype(int)
			X_im = [imresize(arr.reshape((L, L)), (100, 100)) for arr in imgs]	
			pool = Pool(processes=8)
			X = pool.map(hog, X_im)
			pool.close()	
			return np.asarray(X)

		elif ftype is 'pix':

			return imgs 

		else:

			raise NameError('{0} is not implemented!'.format(ftype))


class DigitModel():

	def __init__(self, X, Y, Mopts):

		self.X = X
		self.Y = Y
		self.Mopts = Mopts

	def print_acc(self, predict, label):
		'''
			print the accuracy of prediction
		'''
		print sum(map(lambda x, y: x == y, predict, label))*1./len(label)

	def train(self):

		if self.Mopts['type'] is 'RF':

			rf = RandomForestClassifier(self.Mopts['nTree'], n_jobs=8)
			rf.fit(self.X, self.Y)
			self.print_acc(rf.predict(X), Y)

		self.model = rf

	def cv_train(self, nFold):

		self.model = []
		for train, valid in cv:
			self.model.append(RandomForestClassifier(self.Mopts['nTree'], n_jobs=8))
			self.model[-1].fit(self.X[train], self.Y[train])
			self.print_acc(self.model[-1].predict(self.X[valid]), self.Y[valid])
	
	def test(self, X_test):

		if type(self.model) is not list:

			Y_pred = model.predict(X_test)
			return Y_pred

		else:

			nFold = len(self.model)
			#Y_pred_cv = np.zeros((nFold, X_test.shape[0]))
			Y_pred_cv = np.asarray([rf.predict(X_test) for rf in self.model])
			Y_pred = stats.mode(Y_pred_cv)[0].T.astype(np.int)
			return Y_pred


if __name__ == '__main__':

	digit = DigitData(	trainfile = 'train.csv', 
						testfile = 'test.csv', 
						sampleresult = 'rf_benchmark.csv')

	images_trn, labels_trn = digit.data_trn
	feats_trn = digit.extract_feat(imgs = images_trn, ftype='hog')
	feats_tst = digit.extract_feat(imgs = digit.data_tst, ftype='hog')

	model = DigitModel(	X = feats_trn, 
						Y = labels_trn, 
						Mopts = {'type': 'RF', 'nTree': 100})

	model.train()

	labels_tst = model.test(feats_tst)
	digit.save_data(labels_tst)
