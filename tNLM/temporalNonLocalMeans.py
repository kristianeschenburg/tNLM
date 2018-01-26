#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 21:37:45 2018

@author: kristianeschenburg
"""

import numpy as np

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
        
        [darray,mu,std] = self.whiten(darray)
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

        h = self.h
        vn,vt = neighbors.shape

        d = source[np.newaxis,:] - neighbors
        d = (d**2).sum(axis=1)/vt
        
        w = np.exp(-1.*d/(h**2))
        w = w/w.sum()
        
        source_out = (neighbors*w[:,np.newaxis]).sum(axis=0)

        return source_out
        
    def whiten(self,darray):
        
        """
        Method to whiten the data array.
        
        Parameters:
        - - - - -
            darray : resting state data array
        """

        mu = np.mean(darray,axis=1)
        darray = darray - mu[:,np.newaxis]
        std = np.std(darray,axis=1)
        darray = darray/std[:,np.newaxis]

        return [darray,mu,std]
