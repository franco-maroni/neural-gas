from numpy import *
from operator import itemgetter
from random import shuffle
'''
This implementation mainly follows the guidelines found in the paper:
Estevez, P. A., et al. "Gamma-filter self-organising neural networks
for unsupervised sequence processing."
Electronics Letters 47.8 (2011): 494-496.

The labelling procedure has been implemented taking as a reference
"Online labelling strategies for growing neural gas."
(Beyer, Oliver, and Philipp Cimiano. 2011)

'''

class NeuralGasNode(object):


     def         __init__(self,
                          n_xnodes,
                          n_ynodes,
                          k,
                          init_epsilon,
                          final_epsilon,
#                          epsilon,
                          init_lambda,
                          final_lambda,
#                          lambda_current,
                          beta,
                          n_epochs,
                          n_of_features,
                          #n_features_vectors,
                          categories,
                          alfa0,
                          alfak,
                          weight_matrix):

          self.n_xnodes=n_xnodes
          self.n_ynodes=n_ynodes
          self.k=k
          self.init_epsilon = init_epsilon
          self.final_epsilon = final_epsilon
 #         self.epsilon=epsilon                
          self.init_lambda = init_lambda
          self.final_lambda = final_lambda
#          self.lambda_current=lambda_current
          self.beta=beta           
          self.n_epochs=n_epochs
          self.n_of_features=n_of_features
          #self.n_features_vectors=n_features_vectors
          self.alfa0=alfa0
          self.alfak=alfak
          self.weight_matrix=weight_matrix
          self.categories=categories
          #structure used to simply represent the occurrences of each neuron as BMU(not relevant for ng algorithm)
          self.bmu_matrix = zeros((self.n_xnodes,self.n_ynodes))
          #structure used to count the occurrences of the categories "recognized" by each neuron
          self.neurons_category_counters_matrix = [[{k:0 for k in self.categories} for x in range(self.n_xnodes)] \
                              for y in range(self.n_ynodes)]
          #list containing the "labels" associated to each neuron representing the most frequently recognized category
          self.neurons_labels_matrix = zeros((self.n_xnodes,self.n_ynodes))

     '''   
     def getKey(self,item):
               return item[0]
     '''
     def  train(self, input):
          #nodes_matrix=zeros((self.n_xnodes,self.n_ynodes))

          #Create weight vectors and initialize to random values                 
          self.weight_matrix = [[[random.uniform(-1,1) for x in range (self.n_of_features)] \
                           for y in range(self.n_ynodes)] \
                           for x in range(self.n_xnodes)]
          

          #Create context vectors and initialize to random values
          self.context_matrix = [[[[random.uniform(-1,1) for x in range (self.n_of_features)] \
                           for z in range(self.k)]\
                           for y in range(self.n_ynodes)] \
                           for x in range(self.n_xnodes)]

          #tmp vectors initialization
          v1=zeros(self.n_of_features)
          v2=zeros(self.n_of_features)            

          print 'n_epochs', self.n_epochs
          max_epochs = float(self.n_epochs)

          #iterate on epochs
          for epoch in range(self.n_epochs + 1):
            print 'epoch', epoch
            #randomize input vector
            shuffle(input)            
            
            if epoch < self.n_epochs:
                epochs_ratio = epoch/max_epochs
            else:
                epochs_ratio = 1.
            #set/update epsilon and lambda_current
            self.epsilon = self.init_epsilon * ((self.final_epsilon/self.init_epsilon)**epochs_ratio)
            self.lambda_current = self.init_lambda * ((self.final_lambda/self.init_lambda)**epochs_ratio)
            print 'epochs_ratio', "%0.2f" % epochs_ratio
            print 'epsilon',self.epsilon
            print 'lambda', self.lambda_current

            #iterate on the examples contained in input_vector
            for example in input:
                for i in range(len(example)):
                  #separating features vectors from category vector(used for labeling)
                  features_vectors=example[:,(1,2,3)]
                  category_vector = example[:,0]            

#                  print category_vector
#                  print features_vectors

                  
                  v1=array(features_vectors[i])
                  cat=category_vector[i]
                  neurons_distances=[] # ready for append
                 # print i   
                  for m in range(self.n_xnodes):
                      for n in range(self.n_ynodes):


                          # Init global context vector,there is no past history use random values 
                          if i == 0: # first input vector
                              GContext=[[random.uniform(-1,1) for x in range (self.n_of_features)]\
                                        for y in range(self.k)]
                                               
                              GContext=array(GContext)
                              
                      #calculate  distances  
                      # get a weights vector
                          v2=array(self.weight_matrix[m][n])
                          diff=v1-v2  # difference
                          diff=diff*diff # squared
                          Wdist_sqr=sum(diff) # sum of squares
                          

                      #calculate context distances
                          Cdist_sqr=0.0
                          for k_count in range(self.k):
                               v3=array(self.context_matrix[m][n][k_count])
                               diff=GContext[k_count]-v3 #difference
                               diff=diff*diff  # squared
                               Cdist_sqr+=sum(diff)
                               
                      #calculate distance criterion or distortion error with alfa weights
                          dist_criterion=self.alfa0*Wdist_sqr+self.alfak*Cdist_sqr
                          #nodes_matrix[m][n]=dist_criterion
                          #print nodes_matrix
                      #save dist criterion and matrix coordinates
                          neurons_distances.append([dist_criterion,m,n])
               #           print neuron_distances,i,m,n
              #Calculate global context vector                
              #retrieve previous BMU context vector and weight
                  if i > 0: # for input vectors after the first
                      xc=BMU_xy[0]
                      yc=BMU_xy[1]
                      wk=array(self.weight_matrix[xc][yc])
                      for k_count in range(self.k):
                           ck=array(self.context_matrix[xc][yc][k_count])
                           if k_count == 0:
                                GContext[k_count]=self.beta*ck+(1-self.beta)*wk
                           else:
                                GContext[k_count]=self.beta*ck+(1-self.beta)*array(self.context_matrix[xc][yc][k_count-1])
                         
                                
              #rank distances
                  ranked_neurons=sorted(neurons_distances, key=itemgetter(0))
                #  print ranked_neurons
                  BMU_info=ranked_neurons[0]
                  #print BMU_info
                  # save coordinates of BMU for later use 
                  BMU_xy=[BMU_info[1],BMU_info[2]]
                  # print BMU_xy
                    
                  #increment the counter of the current BMU neuron
                  self.bmu_matrix[BMU_info[1]][BMU_info[2]] += 1

                  #increment the category 'cat' counter for the current BMU neuron
                  self.neurons_category_counters_matrix[BMU_info[1]][BMU_info[2]][cat] +=1
                  
                 
                  #compute the neighbourhood function
                  #lambda is neighbourhood size
                  N_neurons=len(ranked_neurons)
                  for r in range(N_neurons):
                      H=exp(-r/self.lambda_current)
                      # update weights and context
                      current_neuron = ranked_neurons[r]
                      xc=current_neuron[1]
                      yc=current_neuron[2]
                      #  v1  input vector  GContext[i] global vector at input i
                      delta_w=self.epsilon*H*(v1-array(self.weight_matrix[xc][yc]))
                      self.weight_matrix[xc][yc]=self.weight_matrix[xc][yc]+delta_w

                      delta_context=[zeros(self.n_of_features) for x in range(self.k)]
                      for k_count in range(self.k):
                           delta_context[k_count]=self.epsilon*H*(GContext[k_count] - self.context_matrix[xc][yc][k_count])
                           self.context_matrix[xc][yc][k_count]=self.context_matrix[xc][yc][k_count]+delta_context[k_count]     

                      
                      
                      


          for x in range(self.n_xnodes):
               for y in range(self.n_ynodes):
                    #assigned the most frequent category to each neuron as classification label
                    self.neurons_labels_matrix[x][y] = max(self.neurons_category_counters_matrix[x][y].iteritems(), key= itemgetter(1))[0]

                    
     def  test(self, input):
            
        self.num_errors = 0.0
        total_inputs=0.0

        self.num_errors_by_examples=0.0        
        total_examples=0.0

        self.num_errors_by_category = {k:0.0 for k in self.categories}
        total_examples_by_category = {k:0.0 for k in self.categories}
        self.success_rate_by_category = {k:0.0 for k in self.categories}
        
        v1=zeros(self.n_of_features)
        v2=zeros(self.n_of_features)
        

        self.test_labels_counter_vector = [{k:0 for k in self.categories} for x in range(len(input))]
        shuffle(input)
        
                
        for example_idx, example in enumerate(input):
            for i in range(len(example)):
              #input vector
              features_vectors=example[:,(1,2,3)]
              category_vector = example[:,0]            

#                  print category_vector
#                  print features_vectors

              
              v1=array(features_vectors[i])
              cat=category_vector[i]
              neurons_distances=[] # ready for append
             # print i   
              for m in range(self.n_xnodes):
                  for n in range(self.n_ynodes):
                    

                      # Init global context vector,there is no past history use random values 
                    if i == 0: # first input vector
                           GContext=[[random.uniform(-1,1) for x in range (self.n_of_features)]\
                                      for y in range(self.k)]
                           GContext=array(GContext)
                         
                  #calculate  distances  
                  # get a weights vector
                    v2=array(self.weight_matrix[m][n])
                    diff=v1-v2  # difference
                    diff=diff*diff # squared
                    Wdist_sqr=sum(diff) # sum of squares
                      

                  #calculate context distances  
                    Cdist_sqr=0.0
                    for k_count in range(self.k):
                         v3=array(self.context_matrix[m][n][k_count])
                         diff=GContext[k_count]-v3 #difference
                         diff=diff*diff  # squared
                         Cdist_sqr+=sum(diff)
                  #calculate distance criterion or distortion error with alfa weights
                    dist_criterion=self.alfa0*Wdist_sqr+self.alfak*Cdist_sqr
                      #nodes_matrix[m][n]=dist_criterion
                     # print nodes_matrix
                  #save dist criterion and matrix coordinates
                    neurons_distances.append([dist_criterion,m,n])
           #           print neuron_distances,i,m,n
          #Calculate global context vector                
          #retrieve previous BMU context vector and weight
                    if i > 0: # for input vectors after the first
                      xc=BMU_xy[0]
                      yc=BMU_xy[1]
                      wk=array(self.weight_matrix[xc][yc])
                      for k_count in range(self.k):
                           ck=array(self.context_matrix[xc][yc][k_count])
                           if k_count == 0:
                                GContext[k_count]=self.beta*ck+(1-self.beta)*wk
                           else:
                                GContext[k_count]=self.beta*ck+(1-self.beta)*array(self.context_matrix[xc][yc][k_count-1])
          #rank distances
              ranked_neurons=sorted(neurons_distances, key=itemgetter(0))
            #  print ranked_neurons
              BMU_info=ranked_neurons[0]
              #print BMU_info
              # save coordinates of BMU for later use 
              BMU_xy=[BMU_info[1],BMU_info[2]]
              classified = self.neurons_labels_matrix[BMU_info[1]][BMU_info[2]]
              if(classified != cat):
                self.num_errors = self.num_errors+1
              total_inputs+=1
              self.test_labels_counter_vector[example_idx][classified] += 1
              if i == (len(example)-1):
                classified_example = max(self.test_labels_counter_vector[example_idx].iteritems(), key= itemgetter(1))[0]
                if(classified_example != cat):
                    self.num_errors_by_examples += 1
                    self.num_errors_by_category[cat] += 1
                total_examples+=1
                total_examples_by_category[cat]+=1
              
#              self.bmu_matrix[BMU_info[1]][BMU_info[2]] = self.bmu_matrix[BMU_info[1]][BMU_info[2]] + 1
#              self.neurons_category_counters_matrix[BMU_info[1]][BMU_info[2]][cat] = self.neurons_category_counters_matrix[BMU_info[1]][BMU_info[2]][cat] + 1
             # print BMU_xy
              #compute the neighbourhood function
              #lambda is neighbourhood size
        self.success_rate= float("{0:.2f}".format(100 - self.num_errors/total_inputs*100))
        self.example_success_rate=float("{0:.2f}".format(100 - self.num_errors_by_examples/total_examples*100))
        for c in self.categories:
             self.success_rate_by_category[c] = float("{0:.2f}".format(100 - self.num_errors_by_category[c]/total_examples_by_category[c]*100))

             

'''

Example code for initialization and training of the neural gas

n_xnodes=10
n_ynodes=10
n_of_features=3
n_features_vectors=len(features_vector)
n_epochs=1
init_lambda=0.1*n_xnodes*n_ynodes 
final_lambda=0.01
lambda_current=init_lambda #lambda reserved keyword
init_epsilon=0.3
final_epsilon=0.05
epsilon=init_epsilon
init_beta=0.1
final_beta=0.9
beta=init_beta
# the alfa parameters weights express the importance of the current
# input signal over the past
alfa0=0.5  
alfak=0.5


a=NeuralGasNode(
                n_xnodes,
                n_ynodes,
                epsilon,
                lambda_current,
                beta,
                n_epochs,
                n_of_features,
                n_features_vectors,
                alfa0,
                alfak,
                weight_matrix=None)


a.train(features_vector)



'''





                




            
          
            
            
    
