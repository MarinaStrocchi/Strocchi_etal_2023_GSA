import os
import numpy as np
import joblib

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

def model_training(X_train, y_train,
				   model_choice,
				   savepath,
				   X_test=None, y_test=None):

	"""
	This code trains different binary classifies
		- support vector machine
		- logistic regression
		- decision tree
		- random forest
		- k-nearest neighbors
	given a training dataset and an optional test dataset

	Args:
		- X_train: input training parameter set 
		- y_train: input training binary output
		- model_choice: SVC, LogisticRegression, DecisionTree, RandomForest or KNeighbors
		- savepath: where to save the models
		- X_test: input test parameter set 
		- y_test: input test binary output to compute metrics 

	Outputs:
		- model: trained model

	"""	
	if not os.path.exists(savepath+"/"+model_choice+".sav"):
		if model_choice == "SVC":
			print('Training support vector machine model...')
			model = LinearSVC()	

		elif model_choice == "LogisticRegression":
			print('Training logistic regression model...')
			model = LogisticRegression()	

		elif model_choice == "DecisionTree":
			print('Training decision tree model...')
			model = DecisionTreeClassifier()	

		elif model_choice == "RandomForest":
			print('Training random forest model...')
			model = RandomForestClassifier()	

		elif model_choice == "KNeighbors":
			print('Training k-neighbors model...')
			model = KNeighborsClassifier()	

		model.fit(X_train, y_train)	

		joblib.dump(model, savepath+"/"+model_choice+".sav")	

	else:
		model = joblib.load(savepath+"/"+model_choice+".sav")

	if X_test is not None and y_test is not None:	

		print('Computing predictions on the test set...')
		predictions = model.predict(X_test)		

		print('Computing accuracy metrics...')		

		cm = confusion_matrix(y_test, predictions)		

		TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()		

		print('==================================================')
		print(' 	 MODEL : '+model_choice)
		print('==================================================')
		print('--------------------------------------------------')
		print('Test set size: '+str(X_test.shape[0]))
		print('--------------------------------------------------')
		print('True Positive(TP)  = ', TP)
		print('False Positive(FP) = ', FP)
		print('True Negative(TN)  = ', TN)
		print('False Negative(FN) = ', FN)	
		print('--------------------------------------------------')		

		accuracy = accuracy_score(predictions, y_test)
		precision = precision_score(predictions, y_test)		

		print('--------------------------------------------------')
		print('Accuracy  = ', accuracy)
		print('Precision = ', precision)
		print('--------------------------------------------------')

	return model

def binary_classifier(X,
					  y,
					  savepath,
					  X_new=None,
					  model_type="full"):

	"""
	This code trains different binary classifies
		- support vector machine
		- logistic regression
		- decision tree
		- random forest
		- k-nearest neighbors
	and can evaluate the models on a new set of parameters
	to predict which simulations will fail. The method takes 
	the union of the usable samples from all methods to reduce
	the chances of getting rid of too many simulations

	Args:
		- X: input parameter set 
		- y: input binary outcome 
		- savepath: where to save the models
		- X_new: optional new parameter set to evaluate the models after they are trained
		- model_type: full means that the whole initial dataset is used
					  for training, while "validation" splits the initial dataset in
					  training and test 

	Outputs:
		- nimp_final: if X_new is provided, the function returns the indices of the 
				      samples that are ok

	"""	

	if model_type=="validation":	

		print('Splitting in 75% train and 25% test...')
		X_train, X_test, y_train, y_test = train_test_split(X, y , 
														    test_size=0.25, 
														    random_state=0)	
	

		print('Normalising the data...')
		ss_train = StandardScaler()
		X_train = ss_train.fit_transform(X_train)	

		ss_test = StandardScaler()
		X_test = ss_test.transform(X_test)

		svc_model = model_training(X_train, y_train,
							   "SVC", savepath, 
							   X_test=X_test, y_test=y_test)	

		logreg_model = model_training(X_train, y_train,
							  "LogisticRegression", savepath, 
							  X_test=X_test, y_test=y_test)	

		tree_model = model_training(X_train, y_train,
							  "DecisionTree", savepath,
							  X_test=X_test, y_test=y_test)	

		rf_model = model_training(X_train, y_train,
							   "RandomForest", savepath, 
							   X_test=X_test, y_test=y_test)	

		kn_model = model_training(X_train, y_train,
							   "KNeighbors", savepath, 
							   X_test=X_test, y_test=y_test)

	elif model_type=="full":

		print('Normalising the data...')
		ss_train = StandardScaler()
		X_train = ss_train.fit_transform(X)		
		y_train = y

		svc_model = model_training(X_train, y_train,
							   "SVC",
							   savepath)	

		logreg_model = model_training(X_train, y_train,
							  "LogisticRegression",
							   savepath)	

		tree_model = model_training(X_train, y_train,
							  "DecisionTree",
							   savepath)	

		rf_model = model_training(X_train, y_train,
							   "RandomForest",
							   savepath)	

		kn_model = model_training(X_train, y_train,
							   "KNeighbors",
							   savepath)

	else:
		raise Exception("Do not recognise the model_type you chose. Pick between full or validation.")

	if X_new is not None:

		print('Computing predictions for new vector...')

		X_new = ss_train.transform(X_new)

		new_prediction_svc = svc_model.predict(X_new)
		new_prediction_logreg = tree_model.predict(X_new)
		new_prediction_tree = tree_model.predict(X_new)
		new_prediction_rf = rf_model.predict(X_new)
		new_prediction_kn = kn_model.predict(X_new)

		print('Combining positives from different methods...')

		nimp_final = np.concatenate((np.where(new_prediction_svc==1)[0],
									 np.where(new_prediction_logreg==1)[0],
									 np.where(new_prediction_tree==1)[0],
									 np.where(new_prediction_rf==1)[0],
									 np.where(new_prediction_kn==1)[0]))

		nimp_final = np.unique(nimp_final)

		return nimp_final

	else:
		return 0


