import numpy as np
import math
from numpy.random import RandomState




distribution_beta = {'gaussian':2, 'poisson':1, 'gamma':0}



'''
Functions for updating the W and H Matrices given the distribution,
Each has a corresponding phi convex function, from this phi
the functions psi= partial_derivative(phi) and zet= partial_derivative(psi) are found
Then they applied in the following equations:

wp = wp * ([(zet(WH) ⊙ V) H.T]p / [(zet(WH) ⊙ BC)  H.T]p )
hp = hp * ([W.T (zet(WH) ⊙ V)]p / [W.T ⊙ (zet(WH)  BC)]p )
'''



def gaussian_phi(V,W,H):


    H = np.multiply(H, np.divide(W.T.dot(V), W.T.dot(W.dot(H))))
    W = np.multiply(W, np.divide(V.dot(H.T), (W.dot(H)).dot(H.T)))

    return W, H

def poisson_phi(V,W,H):


    H = np.multiply(H, np.divide(W.T.dot(np.multiply(np.divide(1,W.dot(H)), V)), W.T.dot(np.multiply(np.divide(1,W.dot(H)), W.dot(H))) ) )
    W = np.multiply(W, np.divide( np.multiply(np.divide(1,W.dot(H)), V).dot(H.T), np.multiply(np.divide(1,W.dot(H)), W.dot(H)).dot(H.T) ) )

    return W, H

def gamma_phi(V,W,H):

    H = np.multiply(H, np.divide( W.T.dot(np.multiply(np.divide(1,(W.dot(H)))**2,V)) , W.T.dot(np.multiply(np.divide(1,(W.dot(H))**2),W.dot(H)))) )
    W = np.multiply(W, np.divide(np.multiply(np.divide(1,(W.dot(H))**2),V).dot(H.T) ,  np.multiply(np.divide(1,(W.dot(H))**2), W.dot(H)).dot(H.T)))

    return W, H

def bernoulli_phi(V,W,H):

    applied = bernoulli_apply_zeta(W,H,1)
    H = np.multiply(H, np.divide( W.T.dot(np.multiply( applied, V )) , W.T.dot(np.multiply( applied, W.dot(H) )) ) )

    applied = bernoulli_apply_zeta(W,H,1)
    W = np.multiply(W, np.divide(np.multiply( applied, V ).dot(H.T), np.multiply( applied, W.dot(H) ).dot(H.T)  ) )

    return W, H

'''
Apply 1/ ((WH)(1-WH))
'''
def bernoulli_apply_zeta(W,H,limit):

    dot_product = cast_to_limit(W,H,1)
    applied = np.divide(1,np.multiply(dot_product,1-dot_product))
    return applied

"Function for making the dot product of W with H satisfy the domain"
def cast_to_limit(W,H,limit):

    product = W.dot(H)
    indices = np.where(product>=limit)
    product[indices] = limit - 1e-5

    indices = np.where(product==0)
    product[indices] = 1e-5

    return product

distribution_phi = {'gaussian' : gaussian_phi,
                    'poisson'  : poisson_phi,
                     'gamma'   : gamma_phi,
                     'bernoulli' : bernoulli_phi
                     #'binomial'  : binomial_phi,
                     #'multinomial' : multinomial_phi
                    }

class NMF:

    def __init__(self,n_components=None, distribution = 'gaussian', error = 'frobenius', tol=1e-4, max_iterations = 200, random_state=None,  phi_update = False):
        ''' Constructor for this class. '''
        self.n_components = n_components
        if distribution not in list(distribution_phi.keys()):
            raise ValueError('The given distribution is not supported.')
        self.distibution = distribution
        self.error_function = error
        self.max_iterations = max_iterations
        self.tol = tol

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

        H =  H * ( (W.T).dot( ( ( (W.dot(H))**(b-2) ) *V) ) / ( W.T.dot( (W.dot(H))**(b-1) ) ) )
        W =  W * ( ( ( ( (W.dot(H))**(b-2) ) *V ).dot(H.T) ) / ( ( (W.dot(H))**(b-1) ).dot(H.T) ) )

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
