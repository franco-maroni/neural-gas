import numpy as np
import sys
from neural_gas_marconi import NeuralGasNode
from numpy import *
from PIL import Image
import datetime
from random import shuffle
import os
import errno

#import sys

start_time= datetime.datetime.now()

'''
Training input file can be selected by properly specifying n_cats andn_reps
'''
#number of categories --> possible values: 2,4,8
n_cats=2
#number of examples for each category --> possible values: 20(only for n_cats=2),30
n_reps=30

#number of training examples  of each class actually used in the training phase
#(randomly extracted from the n_reps input file examples)
n_examples=5

#enable/disable normailzation over the training/testing set
normalize=False

infile = 'chartrj_'+str(n_cats)+'cat'+str(n_reps)+'rep_train.txt'

testfile = 'chartrj_'+str(n_cats)+'cat'+str(n_reps)+'rep_test.txt'

'''
Neural Gas parameters initialization
'''
#n_xnodes*n_ynodes -> total number of neurons
n_xnodes=10
n_ynodes=10
k=3
n_of_features=3
#n_features_vectors=len(features_vector)
n_epochs=5
init_lambda=0.05*n_xnodes*n_ynodes 
final_lambda=0.01 
init_epsilon=0.3
final_epsilon=0.001
beta=0.5
# the alfa parameters weights express the importance of the current
# input signal over the past
alfa0=0.5  
alfak=0.5/k #equal to 0.5/K

print 'Starting...', start_time

def make_sure_path_exists(path):
    """"Check if the provided path exists. If it does not exist, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


# piece of code from create_ds_all.py
def slices_from_resets_with_cat(cat,reset):
    """
    Receives an array composed of 0s and 1s and returns the slices contained
    between 1s
    """
    resets = np.argwhere(reset).ravel()
    # Computes the slices for each repetition
    repetition_slices = {}
    begin = 0
    for r in resets:
        if r == begin:
            continue
        
        c = cat[begin]
        if not repetition_slices.has_key(c):
            repetition_slices[c] = []
            
        repetition_slices[c].append(slice(begin, r))
        begin = r
    
    c = cat[begin]
    if not repetition_slices.has_key(c):
        repetition_slices[c] = []
    repetition_slices[c].append(slice(begin, len(reset)))
    
    return repetition_slices



def normalize_vector(input_vector):
    normalized_vector=zeros((len(input_vector),input_vector.shape[1]))
    for i in range(len(input_vector)):
        v=input_vector[i]
        sq_v=v*v
        norm=sum(sq_v)
       # print v
        if norm == 0: # some are all zeros
            normalized_vector[i]=v
        else:        
            #print sq
            v=v/sqrt(norm)
            normalized_vector[i]=v
    return normalized_vector


print 'Loading data from ', infile, '....'
data = np.loadtxt(infile,skiprows=1)

#total number of rows in the input file
total_points = data.shape[0]

print 'loaded: ', data.shape

print 'input_dim', data.shape[1]

print "Categories: ", np.unique(data[:,0])

'''
print 'categories_vector', data[:,0]

print 'only 2,3,4 columns:'
print data[:,(2,3,4)]
'''

print "total points",total_points 
print '******************'

#slices to be used in order to have indexes of the examples in the dataset (not currently used)
total_slices = slices_from_resets_with_cat(data[:,0],data[:,1])
slices=[]

#extract n_examples from slices
for cat in total_slices:
    shuffle(total_slices[cat])
    slices.append(total_slices[cat][0:n_examples])


#print slices

#features vectors list (categories and reset columns are omitted)
features_vector = data[:,(2,3,4)]
#categories vacctor (needed for supervised learning)
categories_vector = data[:,0]

#parameter to be passed to the network for classification
categories = np.unique(data[:,0])

print 'input classes'
for c in total_slices.keys():
    print c
    

if normalize:
    norm_features_vector = normalize_vector(features_vector)
else:
    norm_features_vector = features_vector

#print norm_features_vector

#categories_vector reshaped in order to be re-assembled to features_vectors
categories_vector = categories_vector.reshape(norm_features_vector.shape[0],1)        

#categories_vector re-assembled with the features_vectors
normalized_data_vector = np.append(categories_vector, norm_features_vector, axis=1)


#based on the previously found slices, the input_vector is built as a list of examples(i.e. sequences of samples (cat,x,y,z))
input_vector=[]


for s in slices:
        for i in s:
            input_vector.append(normalized_data_vector[i])



'''
Neural Gas instantiation and training
'''

ng=NeuralGasNode(
                n_xnodes,
                n_ynodes,
                k,
                init_epsilon,
                final_epsilon,
#                epsilon,
                init_lambda,
                final_lambda,
#                current_lambda,
                beta,
                n_epochs,
                n_of_features,
#                n_features_vectors,
                categories,
                alfa0,
                alfak,
                weight_matrix=None)

print 'start training - ', n_xnodes * n_ynodes,'nodes'
#print 'category_matrix'+str(n_cats)+'cat_'+str(n_examples)+'_rep_beta'+str(beta)+\
#           '__'+str(n_epochs)+'epochs__'+str(n_xnodes*n_ynodes)+'_neurons.png'

print 'beta', beta
print 'K', k

start_training=datetime.datetime.now()

ng.train(input_vector)

'''
w_m = ng.weight_matrix

print 'Weight Matrix:'
for i in range(len(w_m)):
    for j in range(len(w_m)):
        print      w_m[i][j]
'''

end_training=datetime.datetime.now()
training_time = end_training-start_training


print 'end training - ', n_xnodes * n_ynodes,'nodes', ' - Training Time: ',training_time


'''
#draw ng.bmu_matrix to glance at the occurrences of each neuron as BMU in a graphic way
w,h = ng.n_xnodes,n_ynodes
data = np.zeros( (w,h,3), dtype=np.uint8)
for x in range(w):
#    print a.bmu_matrix[x]
    for y in range(h):
        data[x,y] = [0,ng.bmu_matrix[x][y],0]
img = Image.fromarray(data, 'RGB')
img.save('bmu_matrix.png')
'''


#TESTING SECTION
#Testing is performed over the corresponding testing dataset
#All the examples of the testing dataset are submitted to the neural gas
#Each record in the dataset is labeled depending on the label of the BMU

print '---------TESTING------------'
print 'Loading data from ', testfile, '....'
test_data = np.loadtxt(testfile,skiprows=1)

test_slices = slices_from_resets_with_cat(test_data[:,0],test_data[:,1])

test_features_vector = test_data[:,(2,3,4)]
test_categories_vector = test_data[:,0]

if normalize:
    test_features_vector = normalize_vector(test_features_vector)
else:
    test_features_vector = test_features_vector

#print test_features_vector

test_categories_vector = test_categories_vector.reshape(test_features_vector.shape[0],1)        

test_data_vector = np.append(test_categories_vector, test_features_vector, axis=1)

test_input_vector=[]
for s in test_slices.values():
    for i in s:
        test_input_vector.append(test_data_vector[i])

print 'start TESTING - ', n_xnodes * n_ynodes,'nodes'
start_testing=datetime.datetime.now()

ng.test(test_input_vector)

end_testing=datetime.datetime.now()
testing_time = end_testing-start_testing

print 'end testing - ', n_xnodes * n_ynodes,'nodes', ' -> Testing Time: ',testing_time

print 'Success rate:' , ng.success_rate, '%'

print 'Success rate on examples:' , ng.example_success_rate, '%'

print 'Success rate by category:'
for cat in ng.success_rate_by_category:
    print cat, '->', ng.success_rate_by_category[cat]


#draw ng.neurons_labels_matrix to glance at the labels assigned to each neuron
w,h = ng.n_xnodes,n_ynodes
data = np.zeros( (w,h,3), dtype=np.uint8)
for x in range(w):
#    print a.bmu_matrix[x]
    for y in range(h):
        data[x,y] = [0,30*ng.neurons_labels_matrix[x][y],0]
img = Image.fromarray(data, 'RGB')
out_folder = 'tests_'+str(n_cats)+'cat_'+str(n_reps)+'_rep'
out_filename = 'category_matrix_'+str(n_cats)+'cat_'+str(n_examples)+\
               '_rep_beta_'+str(beta)+'__'+str(n_xnodes*n_ynodes)+'_neurons__'+\
               str(n_epochs)+'epochs__'+str(ng.success_rate)+'p__'+\
               str(ng.example_success_rate)+'ex__lambda_s_'+str(init_lambda)+\
               '_e_'+str(final_lambda)+'__k_'+str(ng.k)+'__norm_'+str(normalize)+'.png'

print '*********************'
print 'saving '+out_folder+'/'+out_filename+' ...'
make_sure_path_exists(out_folder)
img.save(out_folder+'/'+out_filename)

