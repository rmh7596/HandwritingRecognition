import os
testPath = "Test/"
trainPath = "Train/"

os.chdir(trainPath)
for i in range(97,123):
    os.mkdir(chr(i))