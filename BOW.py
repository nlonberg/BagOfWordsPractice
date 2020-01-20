import numpy as np
import string
import os
from sklearn.svm import SVC

#Instance of a vocabulary array
#Contains an array representing all unique words in a set
#Array updated with a Bag of Words
class Vocabulary:
    def __init__(self):
        self.StackedVocabFrequency = np.array([''])

    def getVocabArray(self):
        return self.StackedVocabFrequency
    
    def updateVocabulary(self,BOW,init = False):
        VocabArray = self.getVocabArray()
        for word in BOW:
            skipWords = ['a','the','in','an','and','of','for','to','it','with']
            if word in skipWords:
                continue
            if len(word)==1:
                continue
            contains = word in VocabArray
            if not contains:
                VocabArray = np.append(VocabArray,word)
            else:
                pass
        self.StackedVocabFrequency = VocabArray
        return VocabArray


#A bag of words which represents a single set of data within a vocabulary
class BOW:
    def __init__(self, sentence):
        self.sentence = sentence
        self.bagArray = self.createBagArray()

    def createBagArray(self):
        lowerSent = self.sentence.lower()
        noPunc = lowerSent.translate(str.maketrans('', '', string.punctuation))
        nonumbers = noPunc.translate(str.maketrans('', '', string.digits))
        return nonumbers.split()

    def getBagArray(self):
        return self.bagArray

    def getFrequencyArray(self,VocabArray):
        freqArray = np.zeros(VocabArray.shape)
        for word in self.bagArray:
            if word in VocabArray:
                loc = np.where(VocabArray == word)[0][0]
                freqArray[loc] += 1
        return freqArray

#Method of accepting input to the bag of words model
def takeUserInput():
    sentence = input("Enter Sentence: ")
    return sentence

#Import pipeline for grabbing training set from a directory
def importTrainingSet(dictionary, directory):
    TrainSet = []
    LabelSet = []
    for filename in os.listdir(directory):
        if ".txt" in filename: 
            f = open(directory+filename,"r")
            TrainBag = BOW(f.read())
            TrainSet.append(TrainBag)
            #dumb gimmack for labeling the training data
            if "bio" in filename:
                LabelSet.append(1)
            else:
                LabelSet.append(0)
    return TrainSet,LabelSet

#Import pipeline for importing the test set
def importTestSet(dictionary,directory):
    TestSet = []
    FileNameSet = []
    for filename in os.listdir(directory+"/Test"):
        if ".txt" in filename: 
            f = open(directory+"/Test/"+filename,"r")
            FileNameSet.append(filename)
            TrainBag = BOW(f.read())
            TestSet.append(TrainBag)
    return TestSet, FileNameSet

def main():
    Dictionary = Vocabulary()
    directory = "C:/Users/samta/Documents/MachineLearningPractice"

    TrainBagSet,LabelSet = importTrainingSet(Dictionary, directory)

    for DocBag in TrainBagSet:
        Dictionary.updateVocabulary(DocBag.getBagArray())

    TrainSet = []
    for DocBag in TrainBagSet:
        freq = DocBag.getFrequencyArray(Dictionary.getVocabArray())
        TrainSet.append(freq)
    
    clf = SVC(kernel='linear')
    clf.fit(TrainSet,LabelSet)

    TestBagSet,FileNameSet = importTestSet(Dictionary, directory)
    TestSet = []
    for DocBag in TestBagSet:
        TestSet.append(DocBag.getFrequencyArray(Dictionary.getVocabArray()))

    print(FileNameSet,clf.predict(TestSet))



main()