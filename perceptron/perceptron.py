import numpy as np
import time


def loadData(path):
    dataArr = []
    labelArr = []
    fr = open(path,'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        # 由于需要构造二分类，将>=5作为1，<5作为-1
        if int(curLine[0]) >=5:
            labelArr.append(1)
        else:
            labelArr.append(-1)
        dataArr.append(curLine[1:])
    return dataArr,labelArr

def perceptron(dataArr,labelArr,iter=50):
    dataMat = np.mat(dataArr,dtype='float64')
    labelMat = np.mat(labelArr,dtype='float64').T

    m,n = np.shape(dataMat)
    w = np.zeros((1,n))
    b = 0
    h = 0.0001
    for k in range(iter):
        for i in range(m):
            xi = dataMat[i]
            yi = labelMat[i]
            if -1*yi*(w*xi.T+b) >=0:
                w = w + h * yi*xi
                b = b + h * yi
        print('Round %d:%d training' % (k,iter))

    return w,b


def test(dataArr,labelArr,w,b):
    dataMat = np.mat(dataArr,dtype='float64')
    labelMat = np.mat(labelArr,dtype='float64').T

    m,n = np.shape(dataMat)
    errorCnt = 0
    for i in range(m):
        xi = dataMat[i]
        yi = labelMat[i]
        result = -1 * yi * (w * xi.T +b)
        if result >= 0: errorCnt +=1
    accruRate = 1-(errorCnt/m)
    return accruRate

if __name__ == "__main__":
    start = time.time()

    trainData,trainLabel = loadData('./MNIST/mnist_train.csv')
    testData,testLabel = loadData('./MNIST/mnist_test.csv')

    w,b = perceptron(trainData,trainLabel,iter=10)

    acc = test(testData,testLabel,w,b)

    print('accuracy rate is :',acc)



