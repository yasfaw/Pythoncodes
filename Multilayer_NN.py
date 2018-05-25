#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 01:23:13 2018

@author: yasfaw
"""
import numpy as np
from matplotlib import pyplot as mp
from random import shuffle
#class for creating neural nodes
class node:
    def __init__(self, inbound_nodes,init_weights = [],bias=0,intit_weight_range=1,act_func ='sigmoid'):
        #if initial value is not given, initialize randomly
        if init_weights == []:
            self.weights = np.random.uniform(-1*intit_weight_range, intit_weight_range
                                             ,len(inbound_nodes) + 1)
        else:
          self.weights = np.append(np.array(init_weights),bias) 
        self.bias = bias
        self.activity = 0
        self.activation = 0
        self.sigma = 0
        #kind of activation function used
        self.act_func = act_func
        # a list variable holding inbound nodes from previous layer connected to
        # this node
        self.inbound_nodes = inbound_nodes
        #variable for holding derivative of activation function
        self.derivative = 0
        #a list variable for holding out bound nodes in the next layer
        #connected to this node
        self.outbound_nodes = []
        #for all inbound nodes, add current node as their out bound node
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)   
    def calc_activity(self):
        ex_input = []
        #get the activation value of all inbound nodes
        #these are inout for current node
        for v in self.inbound_nodes:
            ex_input.append(v.feedfrwrd())
        #append 1 to create extended input corresponding to bias
        ex_input = np.append(ex_input,1)
        #caclulate the activity value
        self.activity = np.dot(self.weights,ex_input)
    def calc_activation(self):
        # calculate the activation value based on type of activation function
        if self.act_func == 'sigmoid':
            self.activation = 1/(1 + np.exp(-1*self.activity))
        if self.act_func == 'ramp':
            self.activation = np.log( 1 + np.exp(self.activity))
    def derivatives(self):
        #compute derivatives based on type of activation function
        if self.act_func == 'sigmoid':
            return self.activation*(1 - self.activation)   
        if self.act_func == 'ramp':
            num = np.exp(self.activity)
            return  num/(1 + num )
    # function for carrying out feed forward computation
    def feedfrwrd(self):
        self.calc_activity()
        self.calc_activation()
        self.derivative = self.derivatives()
        return self.activation
# Class for implementing input nodes as they are different from 
#the nodes in other layer
class Input:
    def __init__(self):
        self.activation = 0
        self.outbound_nodes =[]
    def feedfrwrd(self):
        return self.activation
# Class that implements a layer in a network
class layer:
    def __init__(self, number_nodes, inbound_layer,weight_matrix =[], act_func ='sigmoid'
                 , intit_weight_range :"range of uniform dist"=1):
        self.nodes = []
        self.number_nodes = number_nodes
        self.inbound_layer = inbound_layer
        #create nodes in the layer
        if(number_nodes == 1):
           self.nodes.append(node(self.inbound_layer.nodes,weight_matrix
                                  ,intit_weight_range = intit_weight_range, act_func = act_func)) 
        else:
            indx = 0
            for i in range(self.number_nodes):
                if weight_matrix == []:
                    wt = []
                else:
                    wt = weight_matrix[indx]
                self.nodes.append(node(self.inbound_layer.nodes,wt,act_func = act_func))
                indx+=1
#Class for implemnting input layer. Input layer is different from other layers
class input_layer:
    def __init__(self,number_nodes):
        self.nodes = []
        self.number_nodes = number_nodes
        for i in range(number_nodes):
            self.nodes.append(Input())
    # a function that sets input layer nodes activation value to input value
    def addinput(self,input_x):
        assert len(input_x) == self.number_nodes,"Dimension don't match"
        for i in range(self.number_nodes):
            self.nodes[i].activation = input_x[i]
#class implemnting network
class Network:
    def __init__(self):
        #list variable holding layers in network
        self.layers = []
        #a variable for holding output layer 
        self.output_layer = None
    #function for adding fully connected layer to network
    def add_layer(self,layer):
        self.layers.append(layer)
        #the last layer added is the output layer
        self.output_layer = layer
    # function executing feed forward 
    def feed_forward(self, inputx):
        self.layers[0].addinput(inputx)
        for n in self.output_layer.nodes:
            n.feedfrwrd()
    # function executing back propagation 
    def back_propagation(self, y):
        #output layer needs to be treated differently
        for n in self.output_layer.nodes:
            n.sigma = (n.activation -y)*n.derivative
        # back propagation for the rest of hidden layers 
        for l in self.layers[-2:0:-1]:
            idx =0
            for n in l.nodes:
                for ob in n.outbound_nodes:
                    n.sigma = 0
                    n.sigma += ob.sigma*ob.weights[idx] *n.derivative
                idx +=1
    def weight_update(self, rate):
        for l in self.layers[1:]:
            for n in l.nodes:
                indx =0
                for ib in n.inbound_nodes:
                    n.weights[indx] -= rate*n.sigma*ib.activation
                    indx +=1
                #routine that update bias
                n.weights[-1] -= rate * n.sigma
    def nn_train(self, inputx, outputy,rate):
        #self.layers[0].addinput(inputx)
        self.feed_forward(inputx)
        self.back_propagation(outputy)
        self.weight_update(rate)
    def nn_output(self,inputx):
        #self.layers[0].addinput(inputx)
        self.feed_forward(inputx)
        return self.output_layer.nodes[0].activation
# function that computes mean square error based on desired and actual 
#output
def mean_square_error(desired, actual):
    E = 0
    Z = zip(desired,actual)
    for (d,a) in Z:
        E += 1/(len(desired)) *(d - a)**2
    return E
#function that compute error in terms of number of misclassified data points
def error_count(desired, actual):
    count =0
    Z= zip(desired,actual)
    for (d,a) in Z:
        if (d != a):
            count += 1
    return count
#function that compute the ROC values for a given class based on
# desired and actual output
def ROC (desired, actual, class_label):
    Z = zip(desired, actual)
    #True negative and positive, false negative and positive
    TN,FN,TP,FP = 0,0,0,0
    for (d,a) in Z:
        if(d == class_label):
            if (a == class_label):
                TP +=1
            else:
                FN += 1
        else:
            if (a == class_label):
                FP +=1
            else:
                TN += 1
    #print(TN,FN,TP,FP)
 
    sensitivity = TP*100.0/(TP +FN) 
    specificity = TN*100.0/(TN + FP)
    neg_pred_prob  = TN/(TN + FN)
    if(TP + FP == 0):
        pos_pred_prob = 0
    else:
        pos_pred_prob = TP/(TP + FP)
    return  sensitivity,specificity , pos_pred_prob,neg_pred_prob  
# function for plotting data points.
def visual_display(data, output):
    data_0 = [x for x in data if x[2] == 0]
    data_1 = [x for x in data if x[2]  == 1]
    data_2 = [x for x in data if x[2] == 2]
    np_data_0 = np.array(data_0).T
    np_data_1 = np.array(data_1).T
    np_data_2 = np.array(data_2).T
    #ploting NN output
    error = [[x[0],x[1]] for [x,y] in zip(data,output) if x[2] != y[2]]
    ot_0 = [x for x in output if x[2] == 0]
    ot_1 = [x for x in output if x[2]  == 1]
    ot_2 = [x for x in output if x[2] == 2]
    np_ot_0 = np.array(ot_0).T
    np_ot_1 = np.array(ot_1).T
    np_ot_2 = np.array(ot_2).T
    
    print('\n *************** ploting ******************* \n')
    error = np.array(error).T
    try:
        #mp.subplot(3,1,1)
        #ploting original data
        mp.plot(np_data_0[0,:], np_data_0[1,:],'g*',label = 'class 0')
        mp.legend()
        #mp.show()
        mp.plot( np_data_1[0,:], np_data_1[1,:],'r+',label = 'class 1')
        mp.legend()
        mp.plot(np_data_2[0,:], np_data_2[1,:],'bo', label = 'class 2')
        mp.legend()
        mp.title('original data')
        mp.show()
        #ploting output of NN
        #mp.subplot(3,1,2)
        if(ot_0 != []):
            mp.plot (np_ot_0[0,:], np_ot_0[1,:],'g*',label = 'class 0')
            mp.legend()
        mp.plot (np_ot_1[0,:], np_ot_1[1,:],'r+', label = 'class 1')
        mp.legend()
        mp.plot(np_ot_2[0,:], np_ot_2[1,:],'bo',label = 'class 2')
        mp.legend()
        mp.title('labled by NN')
        mp.show()
        #mp.subplot(3,1,3)
        #ploting mis-labeled data
        mp.plot(error[0,:], error[1,:] , 'ro')
        #mp.set_autoscale_on(False)
        mp.ylim(0.0,1.0)
        mp.xlim(0.1,1.0)
        mp.title ('Error')
        mp.show()
    except IndexError:
        if(ot_0 == []):
            print("class 0 is blank")
if __name__ == '__main__':  
    
    #training data
    training_data = [[1.6, 105000, 2],[1.05, 196000, 1],[0.52, 105000, 2],\
                      [1.80, 32000, 1],[2.3, 106000, 0],[2.4, 151000, 1],\
                      [2.5, 170000, 1],[0.50, 150000, 2],[1.1, 35000, 1],[0.85, 70000, 2]]
    testing_data = [[1.98, 10000, 0],[1.80, 10000, 1],[1.05, 160000, 2],\
                    [1.45, 180000, 1],[1.8, 80000, 1],[1.96, 110000, 1],[0.4, 40000, 2],\
                    [2.05, 130000, 1],[0.90, 10000, 1],[2.5, 60000, 0]]
    #Data Processing -- Normalizing LAC and SOW
    training_data = [[x[0]/3.0,x[1]/200000,x[2]] for x in training_data]
    y_desired_training =[y[2] for y in training_data]
    testing_data = [[x[0]/3.0,x[1]/200000,x[2]] for x in testing_data]
    y_desired_testing = [y[2] for y in testing_data]
    #Builiding the network
    ntk3 = None
    ntk3 = Network()
    # add input layer of 2 nodes
    ntk3.add_layer(input_layer(2))
    #add hidden layer of 4 nodes with default sigmoid act. func
    ntk3.add_layer(layer(4,ntk3.layers[0]))
    #add second hidden layer of 2 nodes with sigmoid function
    #ntk3.add_layer(layer(2,ntk3.layers[1]))
    #add output layer of 1 node with ramp function
    ntk3.add_layer(layer(1,ntk3.layers[1], act_func = 'ramp',intit_weight_range = 5))
    
    # training the network
    print('**** training ......')
    #learning rate
    lrate =0.1
    # priniting initial weight set randomly
    print('***** initial weight *********')
    for l in ntk3.layers[1:]:
        for n in l.nodes:
            print(n.weights)
    #training many number of epochs        
    for itr in range(5000):
        trn = training_data[:]
        shuffle(trn)
        for data in trn:
            ntk3.nn_train(data[:2], data[2],lrate)
    #weights after training 
    print('\n ********** weight after trained ********** \n')
    for l in ntk3.layers[1:]:
            for n in l.nodes:
                print(n.weights)
    #testing the network
    print('\n **** testing based on training data**** \n')
    #get the actual label of training data labeled by NN.
    output_training = [] 
    for inp in training_data:
        output_training.append([inp,ntk3.nn_output(inp[:2])])
    print(output_training)
    #Thresholding the output 
    output_training = [[x[0],x[1],0 if y <=0.6 else 1 if 0.6 <y <= 1.5 else 2] for [x,y] in output_training]
    y_output_training = [y[2] for y in output_training]
    # training error
    ms_err = mean_square_error (y_desired_training,y_output_training)
    err_count = error_count(y_desired_training,y_output_training)
    print("\nMeans Square Error = %f " % ms_err)
    print("Error count = %d \n" % err_count)
    # computing ROC for each class
    for c in [0,1,2]:
        s,sp,npr,ppr = ROC(y_desired_training,y_output_training,c)
        print("\n class %d : \n sensitivity = %f  \n specificity = %f \n pos_pred_prob = %f \n neg_pred_prob = %f"
          % (c,s,sp,npr,ppr))
    visual_display(training_data,output_training)
    print('\n **** testing based on test data**** \n')
    #get the actual label of testing data labeled by NN.
    output_testing = []
    for inp in testing_data:
        output_testing.append([inp,ntk3.nn_output(inp[:2])])    
    print(output_testing) 
    #Thresholding the output 
    output_testing = [[x[0],x[1],0 if y <=0.6 else 1 if 0.6 <y <= 1.5 else 2] for [x,y] in output_testing]
    y_output_testing = [y[2] for y in output_testing]
    #testing error
    ms_err = mean_square_error (y_desired_testing,y_output_testing)
    err_count = error_count(y_desired_testing,y_output_testing)
    print("\nMeans Square Error = %f " % ms_err)
    print("Error count = %d \n" % err_count)
    # computing ROC for each class
    for c in [0,1,2]:
        s,sp,npr,ppr = ROC(y_desired_testing,y_output_testing,c)
        print("\n class %d : \n sensitivity = %f  \n specificity = %f \n pos_pred_prob = %f \n neg_pred_prob = %f"
          % (c,s,sp,npr,ppr))
    visual_display(testing_data,output_testing)
    
    
    #Over all performance and visual display
    print('\n*************** over all performance **************\n')
    data_all = training_data
    data_all.extend(testing_data)
    y_desired_all = [y[2] for y in data_all]
    output_all = output_training
    output_all.extend(output_testing)
    y_output_all = [y[2] for y in output_all]

    ms_err = mean_square_error (y_desired_all,y_output_all)
    err_count = error_count(y_desired_all,y_output_all)
    print("\nMeans Square Error = %f " % ms_err)
    print("Error count = %d \n" % err_count)
    # computing ROC for each class
    for c in [0,1,2]:
        s,sp,npr,ppr = ROC(y_desired_all,y_output_all,c)
        print("\n class %d : \n sensitivity = %f  \n specificity = %f \n pos_pred_prob = %f \n neg_pred_prob = %f"
          % (c,s,sp,npr,ppr))
    visual_display(data_all,output_all)

 
