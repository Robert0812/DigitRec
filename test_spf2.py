import os
import csv 
import cPickle
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
#from kaggle import *
from multiprocessing import Pool
from skimage.transform import resize

def print_acc(predict, label):
	'''
		print the accuracy of prediction
	'''
	print sum(map(lambda x, y: x == y, predict, label))*1./len(label)

def print_err(predict, label):
	'''
		print the error of prediction
	'''
	print sum(map(lambda x, y: abs(x-y), predict, label))*1./len(label)

def opencsv(filename):
	f1 = open(filename, 'rb')
	content = csv.reader(f1)
	raw_data = []
	for r in content:
		raw_data.append(r)
	f1.close()
	return raw_data

def savecsv(data, filename):
	f = open(filename, 'wb')
	c = csv.writer(f)
	c.writerows(data)
	f.close()

def imresize(im, shape, interp='bicubic'):
    '''
        replacement of scipy imresize
    '''
    if interp is 'bicubic':
        return (resize(im, shape, order=3)*255).astype(np.uint8)

def data2feat(X_arr, ftype='pix'):

	if ftype is 'hog':
		from skimage.feature import hog
		L = np.sqrt(len(X_arr[0])).astype(int)
		X_im = [imresize(arr.reshape((L, L)), (100, 100)) for arr in X_arr]	
		pool = Pool(processes=8)
		X = pool.map(hog, X_im)
		pool.close()	
		return np.asarray(X)
		
	else:
		return X_arr


nFold = 10
nTree = 100
ftype = 'hog'

def train_model(X, Y, finfo):
	model = []
	cv = KFold(n=len(X), n_folds=nFold, indices=True)
	print 'RF training for feature {} ...'.format(finfo)
	for train, valid in cv:
		model.append(RandomForestClassifier(nTree, n_jobs=8))
		model[-1].fit(X[train], Y[train])
		print_acc(model[-1].predict(X[valid]), Y[valid])
	return model


data = opencsv('../data/train.csv')
X_pix = np.asarray(data)[1:, 1:].astype(np.uint8) 
Y = np.asarray(data)[1:, 0].astype(np.int)

from scipy.io import loadmat
data_feat = loadmat('../data/digit_feat.mat')
X_spf = data_feat['featTrn1'].astype(np.float32).T

X_hog = data2feat(X_pix, ftype)
#X = np.hstack((X_hog, X_spf))

#cv = KFold(n=len(X), n_folds=nFold, indices=True)

#model = []
#for train, valid in cv:
#	model.append(RandomForestClassifier(nTree, n_jobs=8))
#	model[-1].fit(X[train], Y[train])
#	print_acc(model[-1].predict(X[valid]), Y[valid])

model_hog = train_model(X_hog, Y, 'HOG')
model_spf = train_model(X_spf, Y, 'sparse filtering')

data_test = opencsv('../data/test.csv')
X_test = np.asarray(data_test)[1:, :].astype(np.uint8)
X_test_hog = data2feat(X_test, ftype)

X_test_spf = data_feat['featTst1'].astype(np.float32).T

Y_pred_hog = np.asarray([clf.predict(X_test_hog) for clf in model_hog])
Y_pred_spf = np.asarray([clf.predict(X_test_spf) for clf in model_spf])

#Y_pred_cv = np.zeros((nFold, X_test.shape[0]))
#X_test_feat = np.hstack((X_test_hog, X_test_spf))
#for i in range(nModel):
#	clf = model[i]
#	Y_pred_cv[i, :] = clf.predict(X_test_feat)

Y_pred_cv = np.vstack((Y_pred_hog, Y_pred_spf))
Y_pred = stats.mode(Y_pred_cv)[0].T.astype(np.int)

header = opencsv('../data/rf_benchmark.csv')[0]
save_data = []
save_data.append(header)

for i in range(len(Y_pred)):
	save_data.append([str(i+1), str(Y_pred[i][0])])

os.remove('../data/new.csv')
savecsv(save_data, '../data/new.csv')



