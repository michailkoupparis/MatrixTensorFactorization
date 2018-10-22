import numpy as np
from numpy.random import RandomState

distribution_beta = {'gaussian':2, 'poisson':1, 'gamma':0}

class NMF:
    
    def __init__(self,n_components=None, distribution = 'gaussian', error = 'frobenius', max_iterations = 100, random_state=None):        
        ''' Constructor for this class. '''
        self.n_components = n_components
        self.distibution = distribution
        print(self.distibution)
        self.error_function = error
        self.max_iterations = max_iterations

        # Check if a random state is given correctly
        if random_state is not None and not type(random_state) is RandomState:
            raise TypeError('Random State attribute should be an instance of a numpy.random.RandomState.')
        self.random_state = random_state
        self.H = None
        self.W = None

    def printType(self):
        print('NMF class')

    def update_rule(self,W, H):
        print('update rule')
        global distribution_beta
        b = distribution_beta[self.distibution]
        H = H * ( (W.T).dot( ( ( W.dot(H)**(b-2) ) *self.V) ) / ( W.T.dot( (W.dot(H))**(b-1) ) ) )
        W = W * ( ( ( ( W.dot(H)**(b-2) ) *self.V ).dot(H.T) ) / ( ( (W.dot(H))**(b-1) ).dot(H.T) ) )

        return W, H
        
    def fit(self, V):
        print('Fit the Model')
        self.V = V
        n_samples = V.shape[0]
        n_features = V.shape[1]

        # Check if n_components is given if not take all features
        if self.n_components is None:
            self.n_components = n_features

        # Find if a random State is given
        if self.random_state is not None:
            self.W = self.random_state.random_sample((n_samples,self.n_components))
            self.H = self.random_state.random_sample((self.n_components,n_features))
        else:
            self.W = np.random.random_sample((n_samples,n_features))
            self.H = np.random.random_sample((self.n_components,n_features))

        print(self.W)
        print(self.H)
        
    def transform(self, V):
        print('Trnasformation')
        if self.W is None or self.H is None:
            raise ValueError('You must first fit the NMF before tranforming the matrix.')

        W = self.W
        H = self.H
        print(self.W)
        print(self.H)
        for i in range(0,self.max_iterations):
            print("Iteration :" + str(i+1))

            W, H = self.update_rule(W,H)
            if np.array_equal(W,self.W) and np.array_equal(H,self.H):
                print('Update process Stabilize')
                break
            self.W = W
            self.H = H
                
        return W, H   

    def fit_transform(self, V):
        print('Fit the Model and Transform')
        self.fit(V)
        self.transform(V)
