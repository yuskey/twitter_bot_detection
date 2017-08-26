#This file will contain all machine learning algorithms and data wrangling methods for use in sklearn for twitter bot detection
#Data is found in the path global variable, datasets were originally split into bot and human based on follower count (1k, 100k, 1M, 10M)
#Labels are 1 for bot and 0 for human, label datasets need to be created

import os
import csv
from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib
import pickle #used to save the model

path = "C:/Users/Brian/Documents/Python Scripts/Twitter_bot/twitter_bot_detection/data"

def main():
    bot_list = get_data(os.path.join(path, "bots.1k.csv"))
    human_list = get_data(os.path.join(path, "humans.1k.csv"))
    bot_labels = make_labels_bot(bot_list)
    human_labels = make_labels_human(human_list)

    #print len(bot_list)
    #print len(human_list)
    #print len(human_list) + len(bot_list)
    #print len(bot_labels)
    #print len(human_labels)
    #print len(bot_labels) + len(human_labels)

    data, labels = combine(bot_list, human_list, bot_labels, human_labels)

    #print len(data)
    #print len(labels)

#creates a list of zeroes of size len(data) for bot tweet data
def make_labels_bot(data):
    labels = [1] * len(data)
    return labels

#creates a list of zeroes of size len(data) for human tweet data
def make_labels_human(data):
    labels = [0] * len(data)
    return labels

#reads from file and puts data into a list
def get_data(filename):
    with open(filename, 'r') as inf:
        reader = csv.reader(inf)
        my_list = list(reader)
    inf.close()
    return my_list

#combines bot and human data, and bot and human labels, keeps data in same order
def combine(data_b, data_h, labels_b, labels_h):
    data = data_b + data_h
    labels = labels_b + labels_h
    return data, labels


def SVM_tutorial(digits):
	clf = svm.SVC(gamma=0.001, C=100) #creates an SVM classifier with parameters
	print clf.fit(digits.data[:len(digits.data)/2], digits.target[:len(digits.data)/2]) #trains the SVM classifier on the dataset except the last instance: fit(data, labels)
	#print clf.predict(digits.data[len(digits.data)/2:]) #tests the model on the last instance: predict(data)
	print clf.score(digits.data[len(digits.data)/2:], digits.target[len(digits.data)/2:])

	save_model(clf, digits)

def save_model_tutorial(clf, digits):
	joblib.dump(clf, 'test.pkl')
	clf2 = joblib.load('test.pkl')
	#s = pickle.dumps(clf)
	#clf2 = pickle.loads(s)
	print clf2.predict(digits.data[-1:])
    
#def accuracy(labels, p_labels):


if __name__ == "__main__":
	main()
