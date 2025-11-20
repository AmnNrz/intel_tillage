import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import pickle
import os.path

EPS_R = 1e-6   # minimum bandwidth
RIDGE = 1e-8   # kernel diagonal jitter # If anything still hiccups, bump RIDGE to 1e-7.

class Kernel(object):
    # Returns the submatrix of the kernel indexed by ps and qs
    def getKernel(self, ps, qs):
        return np.squeeze(ps[:,None]==qs[None,:]).astype(float)
        # Readable version: return np.array([[1. if p==q else 0. for q in qs] for p in ps])
    def __getitem__(self, args):
        ps, qs = args
        return self.getKernel(ps, qs)

class RBFKernel(Kernel):
    def __init__(self, R):
        self.R = R
    def getKernel(self, ps, qs):
        D = cdist(ps,qs)**2
        # Readable version: D = np.array([[np.dot(p-q, p-q) for q in qs] for p in ps])
        D = np.exp(-D/(2*self.R**2))
        return D
    
class ScoredKernel(Kernel):
    def __init__(self, R, space, scores, alpha=2, gamma=1):
        self.R = R
        self.space = space
        if alpha > 0:
            self.scores = scores**(float(gamma)/alpha)
        else:
            self.scores = scores
    '''
    def getKernel(self, p_ids, q_ids):
        p_ids = np.squeeze(np.array([p_ids]))
        if len(p_ids.shape)<1:
            p_ids = np.array([p_ids])
        q_ids = np.squeeze(np.array([q_ids]))
        if len(q_ids.shape)<1:
            q_ids = np.array([q_ids])
        
        ps = np.array(self.space)[p_ids]
        qs = np.array(self.space)[q_ids]
        D = cdist(ps,qs)**2
        
        # Readable version: D = np.array([[np.dot(p-q, p-q) for q in qs] for p in ps])
        D = np.exp(-D/(2*self.R**2))
        
        # I added the below line to have different sample scores
        D = ((D*self.scores[p_ids]).T * self.scores[q_ids]).T
        return D'''
    
    def getKernel(self, p_ids, q_ids):
        p_ids = np.squeeze(np.array([p_ids]))
        if len(p_ids.shape) < 1:
            p_ids = np.array([p_ids])
        q_ids = np.squeeze(np.array([q_ids]))
        if len(q_ids.shape) < 1:
            q_ids = np.array([q_ids])

        ps = np.array(self.space)[p_ids]
        qs = np.array(self.space)[q_ids]

        D2 = cdist(ps, qs)**2
        R_eff = max(float(self.R), EPS_R)
        K = np.exp(-D2 / (2.0 * R_eff * R_eff))

        # score weighting
        K = ((K * self.scores[p_ids]).T * self.scores[q_ids]).T

        # numeric cleanup
        np.nan_to_num(K, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return K

    
class ConditionedScoredKernel(Kernel):
    def __init__(self, R, space, scores, cond_ids, alpha=2, gamma=1):
        self.R = R
        self.space = np.array(space)
        self.cond_ids = np.sort(cond_ids)
        if alpha > 0:
            self.scores = np.array(scores**(float(gamma)/alpha)).reshape(-1)
        else:
            self.scores = scores
        self.kernel = self.computeFullKernel()
        
    '''    
    def computeFullKernel(self):
        # Construct kernel matrix
        D = cdist(self.space,self.space)**2
        D = np.exp(-D/(2*self.R**2))
        D = ((D*self.scores).T * self.scores).T
        
        # conditioning
        if len(self.cond_ids)>0:
            eye1 = np.eye(len(self.space))
            for id in self.cond_ids:
                eye1[id,id] = 0
            D = np.linalg.inv(D + eye1)
            mask = np.full(len(self.space), True, dtype=bool)
            mask[self.cond_ids] = False
            D = D[mask,:]
            D = D[:,mask]
            D = np.linalg.inv(D) - np.eye(len(self.space)-len(self.cond_ids))
            for id in self.cond_ids: # cond_ids is sorted
                D = np.insert(D, id, 0, axis=0)
                D = np.insert(D, id, 0, axis=1)
                
        return D
    '''
    def computeFullKernel(self):
        X = self.space
        D2 = cdist(X, X)**2
        R_eff = max(float(self.R), EPS_R)
        K = np.exp(-D2 / (2.0 * R_eff * R_eff))
        K = ((K * self.scores).T * self.scores).T

        # ridge for stability
        n = K.shape[0]
        K = K + RIDGE * np.eye(n)

        # conditioning
        if len(self.cond_ids) > 0:
            # zero-out selected diagonal entries then invert with ridge
            eye1 = np.eye(n)
            for idx in self.cond_ids:
                eye1[idx, idx] = 0.0

            # use pinv with ridge for robustness
            Kinv = np.linalg.pinv(K + RIDGE * np.eye(n))

            mask = np.ones(n, dtype=bool)
            mask[self.cond_ids] = False
            Kc = Kinv[mask][:, mask]
            # Schur complement style reconstitution
            Kc = np.linalg.pinv(Kc) - np.eye(mask.sum())

            # reinsert zeros for cond_ids rows/cols
            for idx in self.cond_ids:  # cond_ids sorted
                Kc = np.insert(Kc, idx, 0, axis=0)
                Kc = np.insert(Kc, idx, 0, axis=1)

            K = Kc

        np.nan_to_num(K, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return K


    def getKernel(self, p_ids, q_ids):
        p_ids = np.squeeze(np.array([p_ids]))
        if len(p_ids.shape)<1:
            p_ids = np.array([p_ids])
        q_ids = np.squeeze(np.array([q_ids]))
        if len(q_ids.shape)<1:
            q_ids = np.array([q_ids])
        
        D = self.kernel[p_ids,:]
        D = D[:,q_ids]
        return D

class Sampler(object):
    def __init__(self, kernel, space, k, cond_ids=[]):
        self.kernel = kernel
        self.space = space
        self.k = k
        self.cond_ids = cond_ids
        # norms will hold the diagonals of the kernel
        self.norms = np.array([self.kernel[p_id, p_id][0][0] for p_id in range(len(self.space))])
        self.clear()
    def clear(self):
        # S will hold chosen set of k points
        self.S = list(self.cond_ids)
        # M will hold the inverse of the kernel on S
        if len(self.cond_ids) == 0:
            self.M = np.zeros(shape=(0, 0))
        else:
            self.M = np.linalg.pinv(self.kernel[self.S, self.S])
    '''
    def makeSane(self):
        self.M = np.linalg.pinv(self.kernel[self.S, self.S])
    '''
    def makeSane(self):
        if len(self.S) == 0:
            self.M = np.zeros((0, 0))
        else:
            KSS = self.kernel[self.S, self.S] + RIDGE * np.eye(len(self.S))
            self.M = np.linalg.pinv(KSS)
    
    def testSanity(self):
        eps = 1e-4
        assert self.M.shape == (len(self.S), len(self.S))
        if len(self.S)>0:
            diff = np.abs(np.dot(self.M, self.kernel[self.S, self.S])-np.eye(len(self.S)))
            assert np.all(np.abs(diff)<=eps)
    '''
    def append(self, ind):
        if len(self.S)==0:
            self.S = [ind]
            self.M = np.array([[1./self.norms[ind]]])
        else:
            u = self.kernel[self.S, ind]
            # Compute Schur complement inverse
            v = np.dot(self.M, u)
            scInv = 1./(self.norms[ind]-np.dot(u.T, v))
            self.M = np.block([[self.M+scInv*np.outer(v, v), -scInv*v], [-scInv*v.T, scInv]])
            self.S.append(ind)
    '''
    def append(self, ind):
        if len(self.S) == 0:
            self.S = [ind]
            self.M = np.array([[1.0 / max(self.norms[ind], RIDGE)]])
        else:
            u = self.kernel[self.S, ind]
            v = np.dot(self.M, u)
            denom = float(self.norms[ind] - np.dot(u.T, v))
            if not np.isfinite(denom) or denom <= RIDGE:
                denom = RIDGE
            scInv = 1.0 / denom
            self.M = np.block([
                [self.M + scInv * np.outer(v, v), -scInv * v],
                [(-scInv * v).T,                  np.array([[scInv]])]
            ])
            self.S.append(ind)

    def remove(self, i):
        if len(self.S)==1:
            self.S = []
            self.M = np.zeros(shape=(0, 0))
        else:
            mask = [True]*len(self.S)
            mask[i] = False
            # Readable version: mask = [j!=i for j in range(len(self.S))]
            scInv = self.M[i, i]
            v = self.M[mask, i]
            self.M = self.M[mask, :][:, mask] - np.outer(v, v)/scInv
            self.S = self.S[:i] + self.S[i+1:]
            
    # Return array containing ratio of increase in kernel determinant after adding each point in space
    def ratios(self, item_ids=None):
        if item_ids is None:
            item_ids = np.arange(len(self.space))
        if len(self.S)==0:
            return self.norms[item_ids]
        else:
            U = self.kernel[item_ids, self.S]
            return self.norms[item_ids] - np.sum(np.dot(U, self.M)*U, axis=1)
        
    # Finds greedily the item to add to maximize the determinant of the kernel
    def addGreedy(self):
        self.append(np.argmax(self.ratios()))
    # Important step, because we need to start from a point whose probability is not too small
    def warmStart(self):
        self.clear()
        for i in range(self.k):
            self.addGreedy()
            
    def keepCurrentState(self):
        self.backup_S = self.S.copy()
        self.backup_M = self.M.copy()
    def restoreState(self):
        self.S = self.backup_S.copy()
        self.M = self.backup_M.copy()
        
    # Run one step of Markov chain
    def step(self, alpha=1.):
        temp = np.random.randint(len(self.cond_ids),len(self.S))
        remove_id = self.S[temp]
        self.remove(temp)
        
        add_id = np.random.randint(len(self.space))
        
        new_prob = np.maximum(self.ratios(add_id), 0.)**alpha
        old_prob = np.maximum(self.ratios(remove_id), 0.)**alpha
        
        if np.random.rand() < new_prob / old_prob:
            self.append(add_id)
        else:
            self.append(remove_id)
            
    def sample(self, alpha=2., steps=1000, makeSaneEvery=10):
        for iter in range(steps):
            self.step(alpha=alpha)
            if iter%makeSaneEvery==0:
                self.makeSane()
        return self.S

'''def setup_sampler(space, scores, k, alpha, gamma, cond_ids):
    sp = np.array(space)
    v = np.prod([np.max(sp[:,i])-np.min(sp[:,i]) for i in range(sp.shape[1])])
    d = sp.shape[1]
    R = np.exp(np.log(v)/d - np.log(2*(k+len(cond_ids)))/d) # 2 is an empirical factor
    s = Sampler(ScoredKernel(R, space, scores, alpha, gamma), space, k, cond_ids)
    s.warmStart()
    return s'''

def setup_sampler(space, scores, k, alpha, gamma, cond_ids):
    sp = np.array(space)
    n, d = sp.shape[0], max(1, sp.shape[1])

    # robust volume; avoid zeros
    ranges = np.maximum(np.ptp(sp, axis=0), 1e-12)
    v = float(np.prod(ranges))

    k_eff = max(1, min(int(k), n - len(cond_ids)))
    # try original heuristic
    R = np.exp(np.log(v)/d - np.log(2.0*(k_eff+len(cond_ids)))/d)
    if not np.isfinite(R) or R < EPS_R:
        # fallback for tiny/degenerate pools
        md = float(np.median(pdist(sp))) if n > 1 else 1.0
        R = max(md, EPS_R)

    s = Sampler(ScoredKernel(R, space, scores, alpha, gamma), space, k_eff, cond_ids)
    s.warmStart()
    return s

'''
def sample_ids_mc(points, scores, k, alpha=2., gamma=1., cond_ids=[], steps=1000):
    points = [p for p in points]
    scores = np.array(scores).reshape(-1,1)
    s = setup_sampler(points, scores, k, alpha, gamma, cond_ids)
    x = s.sample(alpha=alpha, steps=steps)
    return x[len(cond_ids):]'''

def sample_ids_mc(points, scores, k, alpha=2., gamma=1., cond_ids=[], steps=1000):
    points = [p for p in points]
    n = len(points)
    k = min(int(k), max(0, n - len(cond_ids)))
    if k <= 0 or n == 0:
        return np.array([], dtype=int)
    scores = np.array(scores).reshape(-1,1)
    s = setup_sampler(points, scores, k, alpha, gamma, cond_ids)
    x = s.sample(alpha=alpha, steps=steps)
    return x[len(cond_ids):]

def sample_mc(points, scores, k, alpha=2., gamma=1., cond_ids=[]):
    x = sample_ids_mc(points, scores, k, alpha, gamma)
    return np.array(points)[x]
