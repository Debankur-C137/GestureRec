from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import scipy as sp
import os, signals
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV



#Check if the module is executed as main, needed for parallel processing
if __name__ == '__main__':
	#List of parameters
	SHOW_CONFUSION_MATRIX = False

	x_data = []
	y_data = []

	classes = {}

	root="data" 

	print "Loading the dataset from '{directory}'...".format(directory=root),

	#Fetch all the data files from the root directory of the dataset
	for path, subdirs, files in os.walk(root):
		for name in files:
			
			filename = os.path.join(path, name)
			
			sample = signals.Sample.load_from_file(filename)
			
			x_data.append(sample.get_linearized())
			
			
			category = name.split("_")[0]
			
			
			number = ord(category) - ord("a")
			
			y_data.append(number)
			
						classes[number] = category

	print "DONE"

	
	
	
	params = {'C':[0.001,0.01,0.1,1], 'kernel':['linear']}

	
	svc = svm.SVC(probability = True)
	#Inizialize the GridSearchCV with 8 processing cores and maximum verbosity
	clf = GridSearchCV(svc, params,verbose =10, n_jobs=8)

	#Split the dataset into two subset, one used for training and one for testing
	X_train, X_test, Y_train, Y_test = train_test_split(x_data, 
				y_data, test_size=0.35, random_state=0)

	print "Starting the training process..."

	#Start the training process
	clf.fit(X_train, Y_train)

	#If SHOW_CONFUSION_MATRIX is true, prints the confusion matrix
	if SHOW_CONFUSION_MATRIX:
		print "Confusion Matrix:"
		Y_predicted = clf.predict(X_test)
		print confusion_matrix(Y_test, Y_predicted)
	
	print "\nBest estimator parameters: "
	print clf.best_estimator_
	
	#Calculates the score of the best estimator found.
	score = clf.score(X_test, Y_test)

	print "\nSCORE: {score}\n".format(score = score)

	print "Saving the model...",

	#Saves the model to the "model.pkl" file
	joblib.dump(clf, 'model.pkl') 
	joblib.dump(classes, 'classes.pkl') 

	print "DONE"
