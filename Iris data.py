import random
import numpy as np
with open("iris.txt","r") as f:
    vec = []
    for line in f:
        vec.append(line.split(','))


    # while i take the vector of iris.txt
    # i get two more empty room at the
    # end of the vector so the folowing lines
    # delit thoth empty element => '\n'
    del vec[-1]
    del vec[-1]

    for line in vec:
        if len(line) == 5:
            if line[4]=="Iris-setosa\n":
                line[4]=1
            if line[4]=="Iris-versicolor\n":
                line[4]=0

print(random.shuffle(vec))
print(vec)

test_2d_vec = [[0,0],
               [0,1],
               [1,0],
               [1,1]]

 
test_1d_target = [0,1,1,1]

def my_perceptron_train(X,Y):
    weightVec = np.zeros(len(X[0]))  #init as zero's vec of weight
    waightIndex    = 0               #represent the current cell of the weight vec by mod % of the len of the weight
    currentTarget  = 0               #represent the current Target cell
    alpha           = 0.1            #the lerning rate
    inputVec        =[]              #the input vec is change every i iteration
    numberOfUpdates = 0
    answer = ()                      #return tuple, contain the weight and the num of updates
    for i in range(len(X)):
        mistake = 0                  #Zigma of the input and the weights
        for j in range(len(X[i])):
            mistake += X[i][j] * weightVec[waightIndex % len(weightVec)]
            inputVec.append(X[i][j])
            waightIndex += 1
        delta = mistake
        if(delta > 0 ):
            delta = 1
        else:
            delta = 0
        if delta != Y[currentTarget]:
            skalar = alpha * (Y[currentTarget] - delta)
            inputVec = np.multiply(inputVec, skalar)
            weightVec = np.add(weightVec, inputVec)
            numberOfUpdates += 1
        currentTarget += 1
        inputVec=[]
    answer = (weightVec, numberOfUpdates)
    print(answer)

my_perceptron_train(test_2d_vec,test_1d_target)

















