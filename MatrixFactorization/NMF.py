import numpy as np
import math
from numpy.random import RandomState




distribution_beta = {'gaussian':2, 'poisson':1, 'gamma':0}



'''
Functions for updating the W and H Matrices given the distribution,
Each has a corresponding phi convex function, from this phi
the functions psi= partial_derivative(phi) and psi_inverse are found
Then they applied in the following equations:

wp = wp * psi_inverse([psi(v.T) H.T]p / [psi(w.T H) H.T]p )
hp = hp * psi_inverse([W.T psi(v)]p / [W.T psi(W h)]p)
'''

def gaussian_phi(V,W,H):


    H = np.multiply(H, np.divide(W.T.dot(V), W.T.dot(W.dot(H))))
    W = np.multiply(W, np.divide(V.dot(H.T), (W.dot(H)).dot(H.T)))

    return W, H

def poisson_phi(V,W,H):

    Vc = V + 0.00000000001
    H = np.multiply(H, np.power(math.e, np.divide(W.T.dot(np.log(np.divide(Vc,W.dot(H)))),W.T.dot(np.ones(Vc.shape)))))
    W = np.multiply ( W, np.power(math.e, np.divide( ((np.log(np.divide(Vc,W.dot(H)))).dot(H.T)) , (np.ones(Vc.shape).dot(H.T)   ) )  ))

    return W, H

def gamma_phi(V,W,H):

    Vc = V + 0.00000000001
    
    H = np.multiply(H, np.divide(W.T.dot(np.divide(W.dot(H),Vc)), W.T.dot(np.ones(Vc.shape))))
    W = np.multiply(W, np.divide(np.divide(W.dot(H),Vc).dot(H.T), np.ones(Vc.shape).dot(H.T)))

    return W, H

distribution_phi = {'gaussian' : gaussian_phi,
                    'poisson'  : poisson_phi,
                     'gamma'   : gamma_phi
                    }



class NMF:

    def __init__(self,n_components=None, distribution = 'gaussian', error = 'frobenius', max_iterations = 200, random_state=None, phi_update = False):
        ''' Constructor for this class. '''
        self.n_components = n_components
        if distribution not in list(distribution_phi.keys()):
            raise ValueError('The given distribution is not supported.')
        self.distibution = distribution
        self.error_function = error
        self.max_iterations = max_iterations

        # Check if a random state is given correctly
        if random_state is not None and not type(random_state) is RandomState:
            raise TypeError('Random State attribute should be an instance of a numpy.random.RandomState.')
        self.random_state = random_state
        self.H = None
        self.W = None

        self.phi_update = phi_update

    def printType(self):
        print('NMF class')

    def update_rule(self,W, H):

        #print('Normal Update')
        b = distribution_beta[self.distibution]
        V = self.V

        H = H * ( (W.T).dot( ( ( (W.dot(H))**(b-2) ) *V) ) / ( W.T.dot( (W.dot(H))**(b-1) ) ) )
        W = W * ( ( ( ( (W.dot(H))**(b-2) ) *V ).dot(H.T) ) / ( ( (W.dot(H))**(b-1) ).dot(H.T) ) )

        return W, H

    def phi_update_rule(self,W, H):

        #print('Phi Update')
        if self.V is None:
            raise ValueError('You should call the fit function first.')

        W, H = distribution_phi[self.distibution](self.V, W, H)
        return W, H

    def fit(self, V):
        #print('Fit the Model')
        self.V = V
        n_samples = V.shape[0]
        n_features = V.shape[1]

        # Check if n_components is given if not take all features
        if self.n_components is None:
            self.n_components = n_features

        minV = np.min(V)
        maxV = np.max(V)

        # Find if a random State is given
        if self.random_state is not None:
            self.W = self.random_state.random_sample((n_samples,self.n_components))
            self.H = self.random_state.random_sample((self.n_components,n_features))

        else:
            self.W = np.random.random_sample((n_samples,n_features))
            self.H = np.random.random_sample((self.n_components,n_features))


    def transform(self, V):
        #print('Transformation')
        if self.W is None or self.H is None:
            raise ValueError('You must first fit the NMF before tranforming the matrix.')

        W = self.W
        H = self.H

        for i in range(0,self.max_iterations):
            if self.phi_update:
                W, H = self.phi_update_rule(W,H)

            else:
                W, H = self.update_rule(W,H)
            if np.array_equal(W,self.W) and np.array_equal(H,self.H):
                print('Update process Stabilize')
                break
            self.W = W
            self.H = H

        return W, H

    def fit_transform(self, V):
        #print('Fit the Model and Transform')
        self.fit(V)
        W, H = self.transform(V)
        return W, H
