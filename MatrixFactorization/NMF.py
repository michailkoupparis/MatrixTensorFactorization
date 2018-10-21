import numpy as np
from numpy.random import RandomState

class NMF:
    distribution_beta = {'gaussian':2, 'poisson':1, 'gamma':0}
    def __init__(self,n_components=None, distribution = 'gaussian', error = 'frobenius', max_iterations = 100, random_state=None):
        ''' Constructor for this class. '''
        self.n_components = n_components
        self.distibution = distribution
        self.error_function = error
        self.max_iterations = max_iterations

        # Check if a random state is given correctly
        if random_state is not None and not type(random_state) is RandomState:
            raise TypeError('Random State attribute should be an instance of a numpy.random.RandomState')
        self.random_state = random_state

    def printType(self):
        print('NMF class')

    def fit(self, V):
        print('Fit the Model')
        self.V = V
        # Find if a random State is given

        
    def transform(self, V):
        print('Trnasformation')

    def fit_transform(self, V):
        print('Fit the Model and Transform')
        
              

        
