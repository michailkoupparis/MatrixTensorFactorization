import numpy as np
import math
from numpy.random import RandomState
from sklearn.decomposition import nmf as NMF_sk
EPSILON = np.finfo(np.float32).eps


distribution_beta = {'gaussian':2, 'poisson':1, 'gamma':0}


def gaussian_phi(V,W,H,l1_reg_W,l1_reg_H,l2_reg_W,l2_reg_H):
    #var = (V.std() ** 2) 
    var = 1
    #H = np.multiply(H, np.divide(W.T.dot(V/var), W.T.dot(W.dot(H)/var)+ l1_reg_H+l2_reg_H * H ))
    #W = np.multiply(W, np.divide((V/var).dot(H.T), (W.dot(H)/var).dot(H.T) + l1_reg_W + l2_reg_W * W ))
    H = np.multiply(H, np.divide(W.T.dot(V), W.T.dot(W.dot(H))+ l1_reg_H+l2_reg_H * H ))
    W = np.multiply(W, np.divide((V).dot(H.T), (W.dot(H)).dot(H.T) + l1_reg_W + l2_reg_W * W ))


    return W, H

def poisson_phi(V,W,H,l1_reg_W,l1_reg_H,l2_reg_W,l2_reg_H):

    zeta =  np.divide(1,W.dot(H))
    H = np.mutliply(H, np.divide(W.T.dot(np.multiply(zeta, V)), W.T.dot(np.multiply(zeta, W.dot(H))) + l1_reg_H +l2_reg_H ) )
    
    zeta = np.divide(1,(W.dot(H)))**2
    W = np.multiply(W, np.divide(np.multiply(zeta, V).dot(H.T), np.multiply(zeta, W.dot(H)).dot(H.T) + l1_reg_W+l2_reg_W) )

    return W, H

def gamma_phi(V,W,H,l1_reg_W,l1_reg_H,l2_reg_W,l2_reg_H):

    zeta = np.divide(1,(W.dot(H)))**2
    
    H = np.multiply(H, np.divide(W.T.dot(np.multiply(zeta,V+1e-5)) , W.T.dot(np.multiply(zeta,W.dot(H))) + l1_reg_H+l2_reg_H)) 
    
    zeta = np.divide(1,(W.dot(H)))**2

    W = np.multiply(W, np.divide(np.multiply(zeta,V+1e-5).dot(H.T) ,  np.multiply(zeta, W.dot(H)).dot(H.T)+ l1_reg_W+l2_reg_W))

    return W, H

def bernoulli_phi(V,W,H,l1_reg_W,l1_reg_H,l2_reg_W,l2_reg_H):

    applied = binomial_apply_zeta(W,H,1)
    H = np.multiply(H, np.divide( W.T.dot(np.multiply( applied, V )) , W.T.dot(np.multiply( applied, W.dot(H) )) + l1_reg_H+l2_reg_H) )

    applied = binomial_apply_zeta(W,H,1)
    W = np.multiply(W, np.divide(np.multiply( applied, V ).dot(H.T), np.multiply( applied, W.dot(H) ).dot(H.T)+ l1_reg_W+l2_reg_W  ) )


    indices = np.where(W.dot(H)>1)
    count = 0
    while len(indices[0]) >=1:
        #print(indices)
        count += 1
        #print("Loop interation" + str(count))
        for (i,c) in zip(indices[0],indices[1]):
            H[:,c] = H[:,c] / 1.001
            #W[i,:] = W[i,:] / 1.1
            indices = np.where(W.dot(H)>1)

    return W, H



def binomial_phi(V,W,H,N,l1_reg_W,l1_reg_H,l2_reg_W,l2_reg_H):

    applied = binomial_apply_zeta(W,H,N)
    H = np.multiply(H, np.divide( W.T.dot(np.multiply( applied, V )) , W.T.dot(np.multiply( applied, W.dot(H) )) + l1_reg_H+l2_reg_H ) )

    applied = binomial_apply_zeta(W,H,N)
    W = np.multiply(W, np.divide(np.multiply( applied, V ).dot(H.T), np.multiply( applied, W.dot(H) ).dot(H.T) + l1_reg_W+l2_reg_W ) )

    indices = np.where(W.dot(H)>N)
    count = 0
    while len(indices[0]) >=1:
        #print(indices)
        count += 1
        #print("Loop interation" + str(count))
        for (i,c) in zip(indices[0],indices[1]):
            H[:,c] = H[:,c] / 1.001
            #W[i,:] = W[i,:] / 1.1
            indices = np.where(W.dot(H)>N)

    return W, H

'''
Apply N/ ((WH)(N-WH))
'''
def binomial_apply_zeta(W,H,limit):

    dot_product = cast_to_limit(W,H,limit)
    applied = np.divide(limit,np.multiply(dot_product,limit-dot_product))
    return applied

"Function for making the dot product of W with H satisfy the domain"
def cast_to_limit(W,H,limit):

    product = W.dot(H)
    indices = np.where(product>=limit)
    product[indices] = limit - 1e-5

    indices = np.where(product==0)
    product[indices] = 1e-5

    return product

def multinomial_phi(V,W,H,N,l1_reg_W,l1_reg_H,l2_reg_W,l2_reg_H):

    applied = multinomail_apply_zeta(W,H,D)
    H = np.multiply(H, np.divide( W.T.dot(np.multiply( applied, V )) , W.T.dot(np.multiply( applied, W.dot(H) )) ) )

    applied = multinomial_apply_zeta(W,H,D)
    W = np.multiply(W, np.divide(np.multiply( applied, V ).dot(H.T), np.multiply( applied, W.dot(H) ).dot(H.T)  ) )

    return W, H

'''
Apply 1/ Sum from 1 to D(1/WH)
'''
def multinomial_apply_zeta(W,H,limit):

    dot_product = W.dot(H)

    applied = np.divide(1,dot_product)
    return applied



distribution_phi = {'gaussian' : gaussian_phi,
                    'poisson'  : poisson_phi,
                     'gamma'   : gamma_phi,
                     'bernoulli' : bernoulli_phi,
                     'binomial'  : binomial_phi,
                     'multinomial' : multinomial_phi
                    }

class NMF:
    
    
    def __init__(self,n_components=None, distribution = 'gaussian', error = 'frobenius', tol=1e-4, max_iterations = 200, init=None,
      phi_update = True, N= None, D=None, alpha=0.0, l1_ratio=0.0,random_state=None):

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

        if distribution == 'binomial' and N is  None:

            raise ValueError('Give the Number of trials N.')
        self.N = N

        if distribution == 'multinomial' and D is  None:

            raise ValueError('Give the Number of Dimensions')
        self.D = D
        
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.init = init


    def printType(self):
        print('NMF class')

    def update_rule(self,W, H):

        #print('Normal Update')
        b = distribution_beta[self.distibution]
        V = self.V

        H =  H * ( (W.T).dot( ( ( (W.dot(H))**(b-2) ) *V) ) / ( W.T.dot( (W.dot(H))**(b-1) ) ) )
        W =  W * ( ( ( ( (W.dot(H))**(b-2) ) *V ).dot(H.T) ) / ( ( (W.dot(H))**(b-1) ).dot(H.T) ) )

        return W, H

    def phi_update_rule(self,W,H,l1_reg_W,l1_reg_H,l2_reg_W,l2_reg_H):

        #print('Phi Update')
        if self.V is None:
            raise ValueError('You should call the fit function first.')

        if self.distibution == 'multinomial':
            W, H = distribution_phi[self.distibution](self.V, W, H, self.D,l1_reg_W,l1_reg_H,l2_reg_W,l2_reg_H)

        elif self.distibution == 'binomial':
            W, H = distribution_phi[self.distibution](self.V, W, H, self.N,l1_reg_W,l1_reg_H,l2_reg_W,l2_reg_H)

        else:

            W, H = distribution_phi[self.distibution](self.V, W, H,l1_reg_W,l1_reg_H,l2_reg_W,l2_reg_H)

        return W, H

    def fit(self, V, W=None, H=None):
        #print('Fit the Model')

        self.V = V
        n_samples = V.shape[0]
        n_features = V.shape[1]

        # Check if n_components is given if not take all features
        if self.n_components is None:
            self.n_components = n_features

        #minV = np.min(V)
        #maxV = np.max(V)

        multiply = 1
        if self.distibution is 'bernoulli':

            multiply = (1 / self.n_components) ** 2

        elif self.distibution is 'binomial':

            multiply = (self.N / self.n_components) ** 2

        # Find if a random State is given
        if self.init is None:
            if self.random_state is not None:
            
                self.W = self.random_state.random_sample((n_samples,self.n_components)) * multiply
                self.H = self.random_state.random_sample((self.n_components,n_features)) * multiply

            else:
                self.W = np.random.random_sample((n_samples,n_features)) * multiply
                self.H = np.random.random_sample((self.n_components,n_features)) * multiply
        
        elif self.init is 'custome':
            
            if W or H:
                raise ValueError('For custome intitialization give the matrices W and H.')
            self.W = W
            self.H = H
            
        else:
            
            self.W, self.H = NMF_sk._initialize_nmf(V, n_components = self.n_components, init=self.init,
                               random_state=self.random_state)

    def transform(self, V, max_iterations=None):
        #print('Transformation')
        if self.W is None or self.H is None:
            raise ValueError('You must first fit the NMF before tranforming the matrix.')

        W = self.W
        H = self.H
        
        if max_iterations:
            self.max_iterations = max_iterations

        for i in range(0,self.max_iterations):
            if self.phi_update:
              
                alpha_H = float(self.alpha)
                alpha_W = float(self.alpha)
                
                l1_reg_W = alpha_W * self.l1_ratio
                l1_reg_H = alpha_H * self.l1_ratio
                l2_reg_W = alpha_W * (1. - self.l1_ratio)
                l2_reg_H = alpha_H * (1. - self.l1_ratio)
                
                W, H = self.phi_update_rule(W,H,l1_reg_W,l1_reg_H,l2_reg_W,l2_reg_H)

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
    
    def get_components(self):
        
        return self.W , self.H
