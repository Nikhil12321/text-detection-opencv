import string
import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn import svm

positive_main_list = []
negative_main_list = []

positive_path = "/home/nikhil/project/positive.txt"
negative_path = "/home/nikhil/project/negative.txt"


file_positive = open(positive_path)
num = 1;

for line in file_positive:
	numbers = line.split()
	temp_list = []
	for n in numbers:
		temp_list.append(float(n))
	positive_main_list.append(temp_list)
	print num*100./78000,"%"
	num = num + 1




num = 1
file_negative = open(negative_path)
for line in file_negative:
	numbers = line.split()
	temp_list = []
	for n in numbers:
		temp_list.append(float(n))
	negative_main_list.append(temp_list)
	print num*100./36000,"%"



len_positive_main_list = len(positive_main_list)
len_negative_main_list = len(negative_main_list)

print len(positive_main_list)
print len(negative_main_list)


train_list = []
test_list_positive = []
test_list_negative = []
num_of_training_positive = len_positive_main_list - 105
num_of_training_negative = len_negative_main_list - 105
num_of_testing = 100
X_train = []
i = 0
j = 0

while i < num_of_training_positive:
	train_list.append(positive_main_list[i])
	X_train.append(0)
	i = i + 1

while i < ( (num_of_training_positive) + num_of_testing):
	test_list_positive.append(positive_main_list[i])
	i = i+1


i = 0

while i < num_of_training_negative:
	train_list.append(negative_main_list[i])
	X_train.append(1)
	i = i + 1

while i < ( (num_of_training_negative) + num_of_testing):
	test_list_negative.append(negative_main_list[i])
	i = i + 1



# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(train_list, X_train)

clf = svm.SVC()
clf.fit(train_list, X_train)
joblib.dump(clf, 'svc.pkl') 
clf = joblib.load('svc.pkl') 

# Testing
result_positive = clf.predict(test_list_positive)
result_negative = clf.predict(test_list_negative)
correct = 0
incorrect = 0
clf.decision_function(test_list_positive)

for i in result_positive:
	if i == 0:
		correct = correct+1
	else:
		incorrect = incorrect+1

for i in result_negative:
	if i == 1:
		correct = correct + 1
	else:
		incorrect = incorrect+1

print "correct ", correct
print "incorrect ", incorrect
