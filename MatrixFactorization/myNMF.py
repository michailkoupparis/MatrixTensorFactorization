import numpy as np
import numbers

from numpy.random import RandomState
from sklearn.decomposition import NMF
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.extmath import safe_min
from sklearn.utils.validation import check_is_fitted, check_non_negative
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition.cdnmf_fast import _update_cdnmf_fast
from sklearn.decomposition.nmf import _check_string_param , _initialize_nmf,_fit_coordinate_descent,_fit_multiplicative_update,_beta_divergence,_compute_regularization, INTEGER_TYPES, EPSILON

distribution_beta = {'gaussian':2, 'poisson':1, 'gamma':0}
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

    H = np.multiply(H, np.divide( W.T.dot(np.multiply(np.divide(1,(W.dot(H)))**2,V)) , W.T.dot(np.multiply(np.divide(1,(W.dot(H))**2),W.dot(H)))) )
    W = np.multiply(W, np.divide(np.multiply(np.divide(1,(W.dot(H))**2),V).dot(H.T) ,  np.multiply(np.divide(1,(W.dot(H))**2), W.dot(H)).dot(H.T)))

    return W, H

def bernoulli_phi(V,W,H,N=None):

    applied = binomial_apply_zeta(W,H,1)
    H = np.multiply(H, np.divide( W.T.dot(np.multiply( applied, V )) , W.T.dot(np.multiply( applied, W.dot(H) )) ) )

    applied = binomial_apply_zeta(W,H,1)
    W = np.multiply(W, np.divide(np.multiply( applied, V ).dot(H.T), np.multiply( applied, W.dot(H) ).dot(H.T)  ) )

    return W, H


def binomial_phi(V,W,H,N):

    applied = binomial_apply_zeta(W,H,N)
    H = np.multiply(H, np.divide( W.T.dot(np.multiply( applied, V )) , W.T.dot(np.multiply( applied, W.dot(H) )) ) )

    applied = binomial_apply_zeta(W,H,N)
    W = np.multiply(W, np.divide(np.multiply( applied, V ).dot(H.T), np.multiply( applied, W.dot(H) ).dot(H.T)  ) )

    return W, H

'''
Apply 1/ ((WH)(1-WH))
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


def multinomial_phi(V,W,H,N):

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
                     'binomial' : binomial_phi,
                     'multinomial' : multinomial_phi

                    }

def update_rule(V, W, H, dist, N = None , D = None):

    global distribution_beta

    if dist is 'bernoulli' :

        W, H = distribution_phi[dist](V,W,H)

    elif dist is 'binomial':

        W, H = distribution_phi[dist](V,W,H,N)

    elif dist is 'multinomial':

        W, H = distribution_phi[dist](V,W,H,D)

    else:
        b = distribution_beta[dist]
        #if (b == 0):
        #    factor = 0.00001
        #else:
        #    factor = 0
        H = H * ( (W.T).dot( ( ( W.dot(H)**(b-2) ) *V) ) / ( W.T.dot( (W.dot(H))**(b-1) ) + 0) )
        W = W * ( ( ( ( W.dot(H)**(b-2) ) *V ).dot(H.T) ) / ( ( (W.dot(H))**(b-1) ).dot(H.T) + 0) )

    return W, H

def update(V,W,H,n_it,dist, N=None, D=None):

    Wold = W
    Hold = H

    for i in range(0,n_it):

            if dist == 'binomial':
                W, H = update_rule(V,W,H,dist,N)
            elif dist == 'multinomial':
                W, H = update_rule(V,W,H,dist,D)
            else:
                W, H = update_rule(V,W,H,dist)
            if np.array_equal(W,Wold) and np.array_equal(H,Hold):
                #print('Update process Stabilize')
                break
            Wold = W
            Hold = H

    return W,H,i

def non_negative_factorization(X, W=None, H=None, n_components=None,
                               init='random', update_H=True, solver='cd',
                               beta_loss='frobenius', tol=1e-4,
                               max_iter=200, alpha=0., l1_ratio=0.,
                               regularization=None, random_state=None,
                               verbose=0, shuffle=False, distribution = 'gaussian', N=None, D=None):
    r"""Compute Non-negative Matrix Factorization (NMF)
    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.
    The objective function is::
        0.5 * ||X - WH||_Fro^2
        + alpha * l1_ratio * ||vec(W)||_1
        + alpha * l1_ratio * ||vec(H)||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
        + 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2
    Where::
        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)
    For multiplicative-update ('mu') solver, the Frobenius norm
    (0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
    by changing the beta_loss parameter.
    The objective function is minimized with an alternating minimization of W
    and H. If H is given and update_H=False, it solves for W only.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant matrix.
    W : array-like, shape (n_samples, n_components)
        If init='custom', it is used as initial guess for the solution.
    H : array-like, shape (n_components, n_features)
        If init='custom', it is used as initial guess for the solution.
        If update_H=False, it is used as a constant, to solve for W only.
    n_components : integer
        Number of components, if n_components is not set all features
        are kept.
    init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' | 'custom'
        Method used to initialize the procedure.
        Default: 'random'.
        Valid options:
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)
        - 'custom': use custom matrices W and H
    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.
    solver : 'cd' | 'mu'
        Numerical solver to use:
        'cd' is a Coordinate Descent solver that uses Fast Hierarchical
            Alternating Least Squares (Fast HALS).
        'mu' is a Multiplicative Update solver.
        .. versionadded:: 0.17
           Coordinate Descent solver.
        .. versionadded:: 0.19
           Multiplicative Update solver.
    beta_loss : float or string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.
        .. versionadded:: 0.19
    tol : float, default: 1e-4
        Tolerance of the stopping condition.
    max_iter : integer, default: 200
        Maximum number of iterations before timing out.
    alpha : double, default: 0.
        Constant that multiplies the regularization terms.
    l1_ratio : double, default: 0.
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    regularization : 'both' | 'components' | 'transformation' | None
        Select whether the regularization affects the components (H), the
        transformation (W), both or none of them.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : integer, default: 0
        The verbosity level.
    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.
    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.
    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.
    n_iter : int
        Actual number of iterations.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import non_negative_factorization
    >>> W, H, n_iter = non_negative_factorization(X, n_components=2,
    ... init='random', random_state=0)
    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.
    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """

    #print('My Non negative Factorization')

    X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)
    check_non_negative(X, "NMF (input X)")
    beta_loss = _check_string_param(solver, regularization, beta_loss, init)

    if safe_min(X) == 0 and beta_loss <= 0:
        raise ValueError("When beta_loss <= 0 and X contains zeros, "
                         "the solver may diverge. Please add small values to "
                         "X, or use a positive beta_loss.")

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    # check W and H, or initialize them
    if init == 'custom' and update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        _check_init(W, (n_samples, n_components), "NMF (input W)")
    elif not update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        # 'mu' solver should not be initialized by zeros
        if solver == 'mu':
            avg = np.sqrt(X.mean() / n_components)
            W = np.full((n_samples, n_components), avg)
        else:
            W = np.zeros((n_samples, n_components))
    else:
        W, H = _initialize_nmf(X, n_components, init=init,
                               random_state=random_state)

    l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = _compute_regularization(
        alpha, l1_ratio, regularization)

    W, H, n_iter = update(X, W, H, max_iter, distribution, N , D)


    if n_iter == max_iter and tol > 0:
        warnings.warn("Maximum number of iteration %d reached. Increase it to"
                      " improve convergence." % max_iter, ConvergenceWarning)

    return W, H, n_iter

class myNMF(NMF):

    def __init__(self, n_components=None, init=None, solver='cd',
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None, alpha=0., l1_ratio=0., verbose=0,
                 shuffle=False,distribution = 'gaussian', N=None, D = None):
        ''' Constructor for this class. '''
        if (distribution not in distribution_phi.keys()):
            raise ValueError('This districution is not supported: ' + str(distribution))
        self.distribution = distribution
        if distribution == 'binomial' and N is  None:

            raise ValueError('Give the Number of trials N.')
        self.N = N

        if distribution == 'multinomial' and D is  None:

            raise ValueError('Give the Number of Dimensions')
        self.D = D

        #print(self.distribution)
        super().__init__(n_components=n_components, init=init, solver=solver,
                     beta_loss=beta_loss, tol=tol, max_iter=max_iter,
                     random_state=random_state, alpha=alpha, l1_ratio=l1_ratio, verbose=verbose,
                     shuffle=shuffle)

    def printType(self):
        print('NMF class')

    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.
        This is more efficient than calling fit followed by transform.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed
        y : Ignored
        W : array-like, shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.
        H : array-like, shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.
        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data.
        """
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)

        W, H, n_iter_ = non_negative_factorization(
            X=X, W=W, H=H, n_components=self.n_components, init=self.init,
            update_H=True, solver=self.solver, beta_loss=self.beta_loss,
            tol=self.tol, max_iter=self.max_iter, alpha=self.alpha,
            l1_ratio=self.l1_ratio, regularization='both',
            random_state=self.random_state, verbose=self.verbose,
            shuffle=self.shuffle, distribution = self.distribution, N=self.N, D=self.D)

        self.reconstruction_err_ = _beta_divergence(X, W, H, self.beta_loss,
                                                    square_root=True)

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter_

        return W

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed
        y : Ignored
        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def transform(self, X):
        """Transform the data X according to the fitted NMF model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be transformed by the model
        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data
        """
        check_is_fitted(self, 'n_components_')

        W, _, n_iter_ = non_negative_factorization(
            X=X, W=None, H=self.components_, n_components=self.n_components_,
            init=self.init, update_H=False, solver=self.solver,
            beta_loss=self.beta_loss, tol=self.tol, max_iter=self.max_iter,
            alpha=self.alpha, l1_ratio=self.l1_ratio, regularization='both',
            random_state=self.random_state, verbose=self.verbose,
            shuffle=self.shuffle , distribution = self.distribution)

        return W
