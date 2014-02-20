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

	elif ftype is 'daisy':
		from skimage.feature import daisy
		L = np.sqrt(len(X_arr[0])).astype(int)
		X_im = [imresize(arr.reshape((L, L)), (50, 50)) for arr in X_arr]	
		pool = Pool(processes=8)
		X = pool.map(daisy, X_im)
		pool.close()	
		return np.asarray(X)

	else:
		return X_arr

data = opencsv('train.csv')
X_arr = np.asarray(data)[1:, 1:].astype(np.uint8)
Y = np.asarray(data)[1:, 0].astype(np.int)

nFold = 10
nTree = 100
ftype = 'hog'

X = data2feat(X_arr, ftype)

cv = KFold(n=len(X), n_folds=nFold, indices=True)

model = []
for train, valid in cv:
	model.append(RandomForestClassifier(nTree, n_jobs=8))
	model[-1].fit(X[train], Y[train])
	print_acc(model[-1].predict(X[valid]), Y[valid])

data_test = opencsv('test.csv')
X_test = np.asarray(data_test)[1:, :].astype(np.uint8)

Y_pred_cv = np.zeros((nFold, X_test.shape[0]))

X_test_feat = data2feat(X_test, ftype)
for i in range(nFold):
	clf = model[i]
	Y_pred_cv[i, :] = clf.predict(X_test_feat)

Y_pred = stats.mode(Y_pred_cv)[0].T.astype(np.int)

header = opencsv('rf_benchmark.csv')[0]
save_data = []
save_data.append(header)

for i in range(len(Y_pred)):
	save_data.append([str(i+1), str(Y_pred[i][0])])

os.remove('new.csv')
savecsv(save_data, 'new.csv')


