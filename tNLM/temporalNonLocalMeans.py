#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 21:37:45 2018

@author: kristianeschenburg
"""
import multiprocessing as mp
from joblib import Parallel,delayed

import numpy as np

NUM_CORES = mp.cpu_count()

class TNLM(object):
    
    """
    Class to implement temporal non-local means smoothing on resting state
    time series MRI data.

    Parameters:
    - - - - -
        h : smoothing kernel size

    """
    
    def __init__(self,h=0.7):

        self.h = h
        
    def smooth(self,darray,adj):
        
        """
        Parameters:
        - - - - -
            darray : resting state data array
            adj : pre-computed adjacency list, where adjacent vertices
                        are at most a distance, k, from the source vertex.
                        The source vertex is included in its own neighborhood.
        """

        [darray,mu,std] = self.standardize(darray)
        dv,dt = darray.shape
        nonzeros = std>0

        darray_tnlm = np.zeros((darray.shape))

        for v in np.arange(dv):
            if nonzeros[v]:
                
                # only get neighbors if they have time-series data associated
                # with them -- we don't want to bias the weights
                v_nbr = adj[v]
                v_nbr = list(set(v_nbr).intersection(nonzeros))
                darray_tnlm[v,:] = self.tnlm(darray[v,:],darray[v_nbr,:])

        return darray_tnlm
    
    def tnlm(self,source,neighbors):
        
        """
        Compute weights for each neighbor
        """
        
        print 'run'
        
        h = self.h
        
        vn,vt = neighbors.shape

        d = source[np.newaxis,:] - neighbors
        d = (d**2).sum(axis=1)/vt
        
        w = np.exp(-1.*d/(h**2))
        w = w/w.sum()
        
        source_out = (neighbors*w[:,np.newaxis]).sum(axis=0)

        return source_out
        
    def standardize(self,darray):
        
        """
        Method to standardize the data array.
        
        Parameters:
        - - - - -
            darray : resting state data array
        """

        mu = np.mean(darray,axis=1)
        darray = darray - mu[:,np.newaxis]
        std = np.std(darray,axis=1)
        darray = darray/std[:,np.newaxis]

        return [darray,mu,std]
    

"""
The below is similar to above, but allows for multiprocessing.
I don't think this is optimized as well as possible -- still in the works.
"""
def smooth(darray,adj,h):
        
    """
    Parameters:
    - - - - -
        darray : resting state data array
        adj : pre-computed adjacency list, where adjacent vertices
                    are at most a distance, k, from the source vertex.
                    The source vertex is included in its own neighborhood.
    """

    [darray,mu,std] = standardize(darray)
    dv,dt = darray.shape

    darray_tnlm = np.zeros((darray.shape))
    
    acceptedV = [k for k in adj.keys() if adj[k]]

    print 'Parallelizing tnlm'
    results = Parallel(n_jobs=NUM_CORES)(delayed(tnlm)(darray[i,:],
                       darray[adj[i],:],i,h) for i in acceptedV)
    
    print 'tnlm complete'
    print 'unzipping'
    r,s = zip(*results)
    
    print 'allocating space'
    darray_tnlm[list(s),:] = np.row_stack(r).squeeze()

    print 'returning array'
    return darray_tnlm
    
def tnlm(D,neighbors,source,h):
        
    """
    Compute weights for each neighbor
    """

    vn,vt = neighbors.shape

    d = D[np.newaxis,:] - neighbors
    d = (d**2).sum(axis=1)/vt
    
    w = np.exp(-1.*d/(h**2))
    w = w/w.sum()
    
    smoothed = (neighbors*w[:,np.newaxis]).sum(axis=0)

    return (smoothed,source)
    
    
def standardize(darray):
        
    """
    Method to standardize the data array.
    
    Parameters:
    - - - - -
        darray : resting state data array
    """

    mu = np.mean(darray,axis=1)
    darray = darray - mu[:,np.newaxis]
    std = np.std(darray,axis=1)
    darray = darray/std[:,np.newaxis]
    
    darray[np.isnan(darray)] = 1e-6

    return [darray,mu,std]
