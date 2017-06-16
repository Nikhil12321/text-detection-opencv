from sklearn.externals import joblib
from sklearn import svm

test_list = []
inp = open("tmp_hog.txt", 'r')
op = open("result.txt", 'w')
clf = joblib.load("svc.pkl")

for line in inp:
	tmp_list = []
	features = line.split()
	for n in features:
		tmp_list.append(float(n))
	test_list.append(tmp_list)
	print tmp_list

ans = clf.predict(test_list)
for a in ans:
	op.write(str(a))
	op.write("\n")


