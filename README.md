## Temporal-aware Neural Gas
Implementation of a temporal-aware (not growing) neural gas 
whose development mainly followed the description provided in the paper 
["Gamma-filter self-organising neural networks for unsupervised sequence processing."](ttp://link.springer.com/chapter/10.1007%2F978-3-642-21566-7_15)
( Estevez, P. A., et al.).

The neural gas is implemented to perform unsupervised learning on temporal sequences. 
Due to time constraints, only a simple training over the complete training set loaded from any of the provided training set files is performed. 
Most of parameters initialization values have been taken from the paper. In particular, K has been set to 1. 

The file read_file_and_launch_train.py allows to launch the train method on the neural gas providing as input the data extracted from a selected training file in the same directory and normalized.
A simple graphical representation of the “most frequent best matching units” is created at the end of the training and saved in the bmu_matrix.png file.

----------

UPDATES: 

The neural gas has now been extended in order to perform classification tasks. As suggested in the paper  
["Online labelling strategies for growing neural gas."](http://link.springer.com/chapter/10.1007/978-3-642-23878-9_10)
(Beyer, Oliver, and Philipp Cimiano. 2011), a labelling procedure has been added to the network. Following a simple frequency based method, the function assignes to each neuron a label corresponding to the category that has been most frequently "recognized" by the neuron.  

In order to perform the labelling two structures have been added to the neural gas: the category_counter_matrix, which stores the relative frequency of each category for each neuron (i.e. the number of times the neuron has been designed as the BMU for a data point of a specific category), and the neurons_labels_matrix, which stores the resulting label for each neuron.
The neurons_labels_matrix is then populated based of the frequencies stored in the category_counter_matrix.
The labels defined during the training phase are then used in the testing phase to classify category examples.

The user now can define the number of categories (2,4,8), the number of repetitions (20,30) and the number of examples to be used in the training phase.
The training is performed on a the user-defined subset of the training data file, while the testing is performed on the whole relative testing file. The examples are always provided in a random order.

Other improvements:
- Higher parametrization
- Possibility to define an arbitrary value for K and to disable input normalization. 