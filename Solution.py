#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 06:16:09 2017

@author: yasfaw
"""
import csv
import numpy as np
import os
import os.path
#import pandas as pd 

#read data and prepare training and test data
path = "/Users/yasfaw/Desktop/IML/Vote.csv"
#arrays for holding traing and test data 
d_data = []
d_data_training = []
d_data_test =[]
d_training_class =[]
d_training_feature = []
d_test_class = []
d_test_feature =[]
# Data imputing: as some of the records in input data have missing values a logic 
#imputing random value is added 
with open(path, 'r') as csvfile:
    #next(f1)
    reader = csv.reader(csvfile, delimiter=',')
    # define the feature columns and class columns based on the data sources
    # also set the maximum random number for imputattion
    if ("Glass" in path or "Breast_Cancer" in path):
            class_idx = 10
            feature_min_idx = 1
            feature_max_idx = 10
            rnd_max = 10
    elif("Iris" in path):
        class_idx = 4
        feature_min_idx = 0 
        feature_max_idx = 4
    elif ("Soybean" in path):
        class_idx = 35
        feature_min_idx = 0 
        feature_max_idx = 35
    elif("Vote" in path):
        class_idx = 0
        feature_min_idx = 1 
        feature_max_idx = 16
        rnd_max = 2

    for row in reader:
        #print(row)
        # imputing random value for missing data and also coverting non numeric binary (n/y)
        #data to numeric binary (0/1) for Vote data source
        corrected_row = row
        if("Breast_Cancer" in path ):
            rnd_psn = np.random.randint(rnd_max) # radom number imputing 
            i =0
            for val in row:
                if (val == '?'):
                    corrected_row[i] = rnd_psn + 1
                i = i + 1
        elif("Vote" in path ):
            rnd_psn = np.random.randint(rnd_max) # radom number imputing 
            i =0
            for val in row:
                if(val == '?'):
                    corrected_row[i] = rnd_psn
                elif(val == 'n'):
                    corrected_row[i] = 0
                elif(val == 'y'):
                    corrected_row[i] = 1
                i =i + 1      
        d_data.append(corrected_row)
        #sampling 2/3 of the data for training and 1/3 for testing
        rnd_psn = np.random.randint(3)# for splitting data set
        if(rnd_psn <= 1):#2/3 for training
            d_training_class.append(corrected_row[class_idx])
            d_training_feature.append(corrected_row[feature_min_idx:feature_max_idx])
            d_data_training.append(corrected_row)
        if(rnd_psn == 2):#1/3 for testing
            d_test_class.append(corrected_row[class_idx])
            d_test_feature.append(corrected_row[feature_min_idx:feature_max_idx])
            d_data_test.append(corrected_row)

# the following function is used to discrete noncategorical attributes into 10 category 
# it uses np.digitize built in function
# input: the data to be digitized  and number of category as number of bins for 
#        np.digitize function. I used defualt value = 10 categories
#output:  array of discrete data
def descritize(data =[], nbr_bins = 10): 
    digitized_data = []
    for  instance in np.asarray(data).astype(float).T:
         minval = min(instance)
         maxval = max(instance)
         delta = (maxval - minval)/nbr_bins
         bins = []
         for dl in range(nbr_bins):
             bins.append(minval + dl*delta)
         row_T = np.digitize(instance,np.asarray(bins), right=False)
         digitized_data.append(row_T)
    return np.asarray(digitized_data).T
# the following function is used to convert the discretized data into 'one hot'
#code of
#input: discrete data and one hot code length
#output:one hot coded data  
def getonehot (data = [], code_length =10):
    onehot_data =[]
    for instance in data:
        binary_instance = []
        for val in instance:
            onehot_val = np.zeros(code_length)
            #print (int(val) - 1)
            onehot_val[ int(val) - 1] = 1
            binary_instance = np.hstack((binary_instance,onehot_val))
        onehot_data.append(binary_instance)
    return onehot_data
# The next function is used to train a winnow_2 model for two class problem.
# input: training data and class. 
#        class_label: is the class we are learning model for.
#       threshold and alpha values. 
def winnow_2_trainings(training_feature =[], training_class = [],class_label ='2',threshold = 0.5, alpha = 2,nbr_iter =1):
    #get the number of attributes
    nbr_attribute = np.asarray(training_feature).shape[1]
    #create an array of ones of size = #attributes for initial wieghts 
    winnow_weights = np.ones(nbr_attribute)
    #learning: do the weight update nbr_iter times. defaulted it to 1
    for iteration in range(nbr_iter):
        row = 0
        for instance in training_feature:
            #go through each attributes on a training instance and do 
            #Demotion and Promottions
            weighted_sum = np.dot(winnow_weights,np.transpose(np.asarray(instance).astype(float)))
            if(weighted_sum > threshold and training_class[row] != class_label): # false positive
                indx=0
                for val in instance:
                    if(val == 1):
                        winnow_weights[indx] =  winnow_weights[indx]/alpha
                    indx = indx + 1 
            if(weighted_sum <= threshold and training_class[row] == class_label):#false negative
                indx = 0
                for val in instance:
                    if(val ==1):
                        winnow_weights[indx] =  winnow_weights[indx]*alpha
                    indx = indx +1
            row =row+1
    return winnow_weights, threshold
# The next function is used to learn winnow_2 models for problem with more than 2  classes
# it makes use of winnow_2_trainings for each class
#output: the list of weights, threshold and the classes
def multi_winnow_2_training (training_feature =[], training_class = [], threshold = 0.5 ,alph = 2):
    # First get the number of classes from training set
    classes, count = np.unique(training_class, return_counts = True)
    # a list for storing wieght matrix for each classes
    multi_winnow_wieght = []
    # for each class, learn a winnow_2 model
    for clas in classes:
        winnow_wieght, threshold = winnow_2_trainings(training_feature, training_class,class_label = clas,threshold = 0.5, alpha = 2)
        multi_winnow_wieght.append(winnow_wieght)
    return  multi_winnow_wieght,threshold,classes,alph
# the following function is used to train Naive Bayes' model. in addition to 
# training data it use m and p for smoothing purpose to account for attributes 
# with zero conditional probability
# input: training_features and classes.
#        : m and p for smoothing parameters
def Naive_Bayes_trainings(training_feature = [], training_class =[], m =1, p = 0.001):
    #get the number of attributes
    nbr_attributes = np.asarray(training_feature).shape[1]
    #get total number of training data 
    nbr_training_data = np.asarray(training_feature).shape[0]
    # get class lables and number of instances per class
    classes, count = np.unique(training_class, return_counts = True)
    count_per_class = np.asarray((classes,count)).T
    #compute class a priori probabilities
    class_prob = np.asarray(count).astype(float)
    #Normalize
    class_prob /= (nbr_training_data*1.0)
    #compute number of classess
    nbr_class = count_per_class.shape[0]
    #compute conditional probability
    # an array for holding class conditional probabilities
    conditional_prob = np.zeros((nbr_class,2,nbr_attributes))
    row = 0
    for instance in training_feature:
        attribute = 0
        # find the class index in classes
        class_indx = np.where(count_per_class[:,0] == training_class[row])[0][0]
        # go through each attribute in an instance and count the number of
        #occurances
        for val in instance:
            conditional_prob[class_indx][int(val)][attribute] += 1
            attribute += 1
        row += 1
    #Normalize
    class_indx = 0
    for clas in count_per_class:
        conditional_prob[class_indx,:,:] = (conditional_prob[class_indx,:,:] + m*p)/(clas[1].astype(int)*1.0 + m)
        class_indx += 1  
    return class_prob, conditional_prob,count_per_class

# The following function is used for performance testing of window_model
#inputs:test features and corresponding classes. 
#        model: weight, threshold 
#       classes: the class list obtained from training set
#output: accuracy and list of predicted classes
def multi_winnow_2_testing(test_feature, test_class, multi_winnow_weights, threshold,classes):
    # get number of testing data
    nbr_test_data = np.asarray(test_feature).shape[0]
    nbr_incorrect=0
    row = 0
    # a list for holding predicted classes
    prediction = []
    # compute the class for each instances
    for instance in test_feature:
        # computed the weighted the sum
        weighted_sum = np.dot(np.asarray(multi_winnow_weights),np.transpose(np.asarray(instance).astype(float)))
        #compare with threshold
        weighted_sum = (weighted_sum > threshold)
        #get the first class with value 1
        clas_predicted_idx = np.argmax(weighted_sum)
        #compute model accuracy
        if(classes[clas_predicted_idx] !=  test_class[row]): # false positive
            nbr_incorrect += 1 
        #capture the predicted classes
        prediction.append(classes[clas_predicted_idx])
        row = row + 1
    #percentage of correcness
    accuracy = 1 - (nbr_incorrect )/nbr_test_data
    return accuracy,prediction
# the following function is used to evaluate performance of Naive Bayes' model
#inputs: test data, class probability. conditional probability and class frequency matrix
#output: accuracy and list of predicted classes
def Naive_Bayes_testings(test_feature, test_class, class_prob , conditional_prob, count_per_class):
    nbr_incorrect = 0
    row = 0
    #get number of test data
    nbr_test_data = np.asarray(test_feature).shape[0]
    #list for capturing predicted classes
    prediction = []
    # get number of classes obtained from training set
    nbr_class = count_per_class.shape[0]
    #go through each attribute of an instance and compute likely hood * class probability
    for instance in test_feature:
        #array for holding aposteriori 
        discriminators = np.zeros(nbr_class)
        class_indx = 0
        # compute aposteriori for each class
        for clas in count_per_class:
            attribute = 0
            #discriminator = []
            aposteriori =1
            for val in instance:
                aposteriori *= (conditional_prob[class_indx][int(val)][attribute]*class_prob[class_indx])
                attribute += 1
            discriminators[class_indx]= aposteriori
            class_indx += 1
        #predict the class as argmax of aposteriori
        class_predicted = np.argmax(np.asarray(discriminators))
        #capture predicted classes
        prediction.append(count_per_class[class_predicted][0])
        #compute accuracy
        if (count_per_class[class_predicted][0] != test_class[row]):
            nbr_incorrect += 1
        row += 1
        #break
    accuracy = 1 - nbr_incorrect/nbr_test_data
    return accuracy , prediction     
   
# training
# the following are already categorical. no need to dicretize
if("Vote" in path or "Breast_Cancer" in path or "Soybean" in path):
    digitized_training = d_training_feature
else:
    digitized_training = descritize(d_training_feature, nbr_bins = 10)
#the following sources are already digitized
if("Vote" in path):
    onehot_training_feature = digitized_training
else:
    onehot_training_feature = getonehot(digitized_training)
multi_winnow_weights,threshold,classes,alpha = multi_winnow_2_training(onehot_training_feature,d_training_class)
if ("Breast_Cancer" in path):
    fname ="/Users/yasfaw/Desktop/IML/Winnow_Breast_Cancer_weight.txt"
if ("Glass" in path):
    fname ="/Users/yasfaw/Desktop/IML/Winnow_Glass_weight.txt"
if ("Iris" in path):
    fmt='%s, %s, %s, %s, %s, %s'
    fname ="/Users/yasfaw/Desktop/IML/Winnow_Iris_weight.txt"
if ("Soybean" in path):
     fname ="/Users/yasfaw/Desktop/IML/Winnow_Soybean_weight.txt"
if ("Vote" in path):
    fname ="/Users/yasfaw/Desktop/IML/Winnow_Vote_weight.txt"

np.savetxt(fname, multi_winnow_weights
           , delimiter=' ', newline='\n', header='Winnow_2 Weights')
with open(fname, "a") as myfile:
    myfile.write("Threshold :" + str(threshold))
    myfile.write("alpha :" + str(alpha))
    #myfile.write(threshold)
print("priniting weights")
print(np.asarray(multi_winnow_weights))
print(np.asarray(onehot_training_feature).shape)
#Naive Bayes Training
class_p, cond_p, count_per_class= Naive_Bayes_trainings(onehot_training_feature, d_training_class)
print(class_p)
print(count_per_class)
print(cond_p[0][:][0:20]*int(count_per_class[0][1]))
#print(cond_prob_tes[0][:][0:10])

if ("Breast_Cancer" in path):
    fname ="/Users/yasfaw/Desktop/IML/Bayes_Breast_Cancer"
if ("Glass" in path):
    fname ="/Users/yasfaw/Desktop/IML/Bayes_Glass"
if ("Iris" in path):
    fmt='%s, %s, %s, %s, %s, %s'
    fname ="/Users/yasfaw/Desktop/IML/Bayes_Iris"
if ("Soybean" in path):
     fname ="/Users/yasfaw/Desktop/IML/Bayes_Soybean"
if ("Vote" in path):
    fname ="/Users/yasfaw/Desktop/IML/Bayes_Vote"



np.savetxt(fname+"_class_p", class_p
           , delimiter=' ', newline='\n', header='Class Probability')
#with open(fname, "a") as myfile:
#np.savetxt(fname+"_cond_p.txt", cond_p
#          , delimiter=' ', newline='\n', header='Conditional Probability')
#testing
# the following are already categorical. no need to dicretize
if("Vote" in path or "Breast_Cancer" in path or "Soybean" in path):
    digitized_test = d_test_feature
else:
    digitized_test = descritize(d_test_feature, nbr_bins = 10)
#the following sources are already digitized
if("Vote" in path):
    onehot_test_feature = digitized_test
else:
    onehot_test_feature = getonehot(digitized_test)
print("test data")
print(np.asarray(d_test_feature).T.shape)
print(d_test_class)
accuracy,prediction= multi_winnow_2_testing(onehot_test_feature,d_test_class,multi_winnow_weights,threshold,classes)
print ("winnow_2 accuracy: " )
print(accuracy)
print(prediction)

# Saving Results
nbr_test =np.asarray(d_data_test).shape[0]
Bayes_result = np.hstack((np.asarray(d_data_test),np.asarray(prediction).reshape((nbr_test,1))))
print(np.asarray(Bayes_result).shape)
if ("Breast_Cancer" in path):
    fmt='%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s'
    fname ="/Users/yasfaw/Desktop/IML/Winnow_Breast_Cancer_test.csv"
if ("Glass" in path):
    fmt='%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s'
    fname ="/Users/yasfaw/Desktop/IML/Winnow_Glass_test.csv"
if ("Iris" in path):
    fmt='%s, %s, %s, %s, %s, %s'
    fname ="/Users/yasfaw/Desktop/IML/Winnow_Iris_test.csv"
if ("Soybean" in path):
    fmt='%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s'
    fname ="/Users/yasfaw/Desktop/IML/Winnow_Soybean_test.csv"
if ("Vote" in path):
    fmt='%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s,%s,%s'
    fname ="/Users/yasfaw/Desktop/IML/Winnow_Vote_test.csv"
np.savetxt(fname, Bayes_result, delimiter=",", fmt = fmt)

print(nbr_test)
print(np.asarray(prediction).reshape((nbr_test,1)))
print(np.asarray(d_data_test).shape)
winnow_result = np.hstack((np.asarray(d_data_test),np.asarray(prediction).reshape((nbr_test,1))))
print(row)
accuracy, prediction = Naive_Bayes_testings(onehot_test_feature, d_test_class, class_p , cond_p , count_per_class)

print ("Naive Bayes accuracy: " )
print(accuracy)
print(prediction)


## Saving result
print("Ssaving")
#print(np.asarray(Bayes_result).shape)
Bayes_result = np.hstack((np.asarray(d_data_test),np.asarray(prediction).reshape((nbr_test,1))))
print(np.asarray(Bayes_result).shape)
if ("Breast_Cancer" in path):
    fmt='%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s'
    fname ="/Users/yasfaw/Desktop/IML/Bayes_Breast_Cancer_test.csv"
if ("Glass" in path):
    fmt='%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s'
    fname ="/Users/yasfaw/Desktop/IML/Bayes_Glass_test.csv"
if ("Iris" in path):
    fmt='%s, %s, %s, %s, %s, %s'
    fname ="/Users/yasfaw/Desktop/IML/Bayes_Iris_test.csv"
if ("Soybean" in path):
    fmt='%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s'
    fname ="/Users/yasfaw/Desktop/IML/Bayes_Soybean_test.csv"
if ("Vote" in path):
    fmt='%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s,%s,%s'
    fname ="/Users/yasfaw/Desktop/IML/Bayes_Vote_test.csv"
np.savetxt(fname, Bayes_result, delimiter=",", fmt = fmt)
#Bayes_result.tofile('/Users/yasfaw/Desktop/IML/Breast_Cancer_test.csv',sep=',',format='%10.5f')
#df = pd.DataFrame(Bayes_result)
#df.to_csv("/Users/yasfaw/Desktop/IML/Breast_Cancer_test.csv")
