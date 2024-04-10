#bms
import numpy as np
from scipy.special import psi, gammaln, betainc
from scipy.stats import dirichlet,gamma
import warnings

def bms(lme):
    '''
    Adapted from matlab code by % Sam Gershman, July 2015
    Ham Huang, April 2024
    % Bayesian model selection for group studies.
    %
    % USAGE: [alpha,exp_r,xp,pxp,bor] = bms(lme)
    %
    % INPUTS:
    %   lme      - array of log model evidences
    %              rows: subjects
    %              columns: models (1..Nk)
    %
    % OUTPUTS:
    %   alpha   - vector of model probabilities
    %   exp_r   - expectation of the posterior p(r|y)
    %   xp      - exceedance probabilities
    %   pxp     - protected exceedance probabilities
    %   bor     - Bayes Omnibus Risk (probability that model frequencies
    %           are equal)
    %   g       - posterior belief g(i,k)=q(m_i=k|y_i) that model k generated
    %           the data for the i-th subject
    %
    % REFERENCES:
    %
    % Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009)
    % Bayesian Model Selection for Group Studies. NeuroImage 46:1004-1017
    %
    % Rigoux, L, Stephan, KE, Friston, KJ and Daunizeau, J. (2014)
    % Bayesian model selection for group studies—Revisited.
    % NeuroImage 84:971-85. doi: 10.1016/j.neuroimage.2013.08.065
    %__________________________________________________________________________
    % Based on the function spm_BMS.m in SPM12.
    
    '''
    Ni, Nk = lme.shape
    c = 1
    cc = 10e-4
    alpha0 = np.ones(Nk)
    alpha = alpha0.copy()

    # Iterative VB estimation
    while c > cc:
        log_u = np.zeros((Ni, Nk))
        g = np.zeros((Ni, Nk))
        for i in range(Ni):
            for k in range(Nk):
                log_u[i, k] = lme[i, k] + psi(alpha[k]) - psi(np.sum(alpha))
            u = np.exp(log_u[i, :] - np.max(log_u[i, :]))
            u_i = np.sum(u)
            g[i, :] = u / u_i

        beta = np.sum(g, axis=0)

        # Update alpha
        prev = alpha.copy()
        alpha = alpha0 + beta

        # Convergence?
        c = np.linalg.norm(alpha - prev)

    exp_r = alpha / np.sum(alpha)
    if Nk == 2:
        xp = np.array([bcdf(0.5, alpha[1], alpha[0]), bcdf(0.5, alpha[0], alpha[1])])
    else:
        xp = dirichlet_exceedance(alpha)

    posterior = {'a': alpha, 'r': g.T}
    priors = {'a': alpha0}
    bor, _, _ = BMS_bor(lme.T, posterior, priors)

    pxp = (1 - bor) * xp + bor / Nk

    return alpha, exp_r, xp, pxp, bor, g

def bcdf(x, v, w):
    # Convert inputs to NumPy arrays if they aren't already (this includes scalars)
    x, v, w = map(np.atleast_1d, [x, v, w])
    # Determine the number of dimensions of each input.
    ad = np.array([x.ndim, v.ndim, w.ndim])

    # Find the maximum number of dimensions.
    rd = ad.max()

    # Adjust the shapes to have the same number of dimensions by appending 1's where necessary.
    # This makes them compatible for broadcasting in numpy operations.
    as_ = np.array([
        np.concatenate([x.shape, np.ones(rd - ad[0])]),
        np.concatenate([v.shape, np.ones(rd - ad[1])]),
        np.concatenate([w.shape, np.ones(rd - ad[2])])
    ])
    
    # Find the size of the resulting array after broadcasting.
    rs = as_.max(axis=0)
    # Determine which arrays are not scalars (have more than one element).
    xa = np.prod(as_, axis=1) > 1
    if xa.sum() > 1 and np.any(np.diff(as_[xa, :], axis=0)):
        raise ValueError('non-scalar args must match in size')

    # Initialize result array F
    F = np.zeros_like(rs)
    # Define mask for valid input conditions
    valid_mask = (x >= 0) & (x <= 1) & (v > 0) & (w > 0)

    # Set NaN for out-of-range arguments
    if not np.all(valid_mask):
        F[~valid_mask] = np.nan
        warnings.warn('Returning NaN for out of range arguments')

    # Special case: F=1 when x=1

    F[(valid_mask) & (x == 1)] = 1
    
    # Find indices where conditions are met: md is True, x > 0, and x < 1.
    Q = np.where(valid_mask & (x > 0) & (x < 1))[0]

    # Return from the function if Q is empty. In Python, you would typically raise an exception or handle it differently depending on the context.
    if Q.size == 0:
        return
    # Assign Q to Qx, Qv, and Qw based on conditions in xa. In Python, '1' is used as an index, which does not directly translate from MATLAB's 1-based indexing,
    # so we adjust the logic to Python's 0-based indexing by using '0' or the actual condition array 'Q'.
    Qx = Q if xa[0] else 0  # Adjusted for Python's indexing
    Qv = Q if xa[1] else 0  # Adjusted for Python's indexing
    Qw = Q if xa[2] else 0  # Adjusted for Python's indexing

    F[Q] = betainc(v[Qv], w[Qw], x[Qx]) #scipy order of input differs from matlab
    return F

def BMS_bor(L, posterior, priors, C=None):
    """
    Compute Bayes Omnibus Risk (BOR).

    Parameters:
    - L: numpy array, log model evidences with shape (models, subjects)
    - posterior: dictionary with posterior distributions
    - priors: dictionary with prior distributions
    - C: optional, family priors if provided

    Returns:
    - bor: Bayes Omnibus Risk
    - F0: Free energy of the null hypothesis (equal model frequencies)
    - F1: Free energy of the alternative hypothesis
    """
    options = {'families': False}
    
    if C is None:
        F0, _ = FE_null(L, options)
    else:
        options['families'] = True
        options['C'] = C
        _, F0 = FE_null(L, options)

    F1 = FE(L, posterior, priors)
    bor = 1 / (1 + np.exp(F1 - F0))

    return bor, F0, F1

def FE(L, posterior, priors):
    K, n = L.shape
    a0 = np.sum(posterior['a'])
    Elogr = psi(posterior['a']) - psi(a0)
    Sqf = np.sum(gammaln(posterior['a'])) - gammaln(a0) - np.sum((posterior['a']-1) * Elogr)
    Sqm = 0
    for i in range(n):
        Sqm -= np.sum(posterior['r'][:, i] * np.log(posterior['r'][:, i] + np.finfo(float).eps))
    ELJ = gammaln(np.sum(priors['a'])) - np.sum(gammaln(priors['a'])) + np.sum((priors['a']-1) * Elogr)
    for i in range(n):
        for k in range(K):
            ELJ += posterior['r'][k, i] * (Elogr[k] + L[k, i])
    F = ELJ + Sqf + Sqm
    return F

def FE_null(L, options):
    K, n = L.shape
    F0m = 0
    F0f = []
    
    if options.get('families', False):
        C = options['C']
        f0 = C * np.sum(C, axis=1)[:, None] ** -1 / C.shape[1]
        F0f = 0

    for i in range(n):
        tmp = L[:, i] - np.max(L[:, i])
        g = np.exp(tmp) / np.sum(np.exp(tmp))
        for k in range(K):
            F0m += g[k] * (L[k, i] - np.log(K) - np.log(g[k] + np.finfo(float).eps))
            if options.get('families', False):
                F0f += g[k] * (L[k, i] - np.log(g[k] + np.finfo(float).eps) + np.log(f0[k]))

    return F0m, F0f

def dirichlet_exceedance(alpha):
    Nsamp = 1e6
    Nk = len(alpha)
    
    # Calculate block sizes for sampling in chunks to manage memory usage
    blk = int(np.ceil(Nsamp*Nk*8 / 2**28))
    blk_sizes = [Nsamp // blk] * blk
    blk_sizes[-1] = Nsamp - sum(blk_sizes[:-1])  # Adjust last block size
    
    xp = np.zeros(Nk)
    for size in blk_sizes:
        # Sample from univariate gamma distributions then normalize
        r = gamma.rvs(a=alpha, size=(int(size), Nk))
        r /= r.sum(axis=1, keepdims=True)
        
        # Exceedance probabilities: count the mode selections across samples
        max_indices = np.argmax(r, axis=1)
        for k in range(Nk):
            xp[k] += np.sum(max_indices == k)
    
    xp /= Nsamp
    return xp
