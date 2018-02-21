#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:13:33 2018

@author: kristianeschenburg
"""

import os
import sys
sys.path.append('../../niIO/')
sys.path.append('../../shortestpath/')
sys.path.append('../../tNLM/')

import niIO
import shortestpath
import tNLM

import networkx as nx
import h5py

if __name__ == "__main__":
    
    mainDir = '/mnt/parcellator/parcellation/parcellearning/Data/'
    subjectList = ''.join([mainDir,'Subjects.txt'])
    
    with open(subjectList,'r') as inSubj:
        subjects = inSubj.readlines()
    subjects = [x.strip() for x in subjects]
    
    cutoff = 10
    h = 0.7
    
    for subj in subjects:
        
        restFile = ''.join([mainDir,'RestingState/',subj,'.rfMRI_Z-Trans_merged_CORTEX_LEFT.mat'])
        surfFile = ''.join([mainDir,'Surfaces/',subj,'.L.inflated.32k_fs_LR.surf.gii'])
        outFile = ''.join([mainDir,'RestingState/',subj,'.L.TNLM.mat'])
        
        if os.path.exists(restFile) and os.path.exists(surfFile):
            
            rest = niIO.loaded.loadMat(restFile)
            S = shortestpath.SurfaceAdjacency(surfFile)
            S.generate_adjList()
            G = nx.from_dict_of_lists(S.adj_list)
            
            apsp = nx.all_pairs_shortest_path_length(G,cutoff=cutoff)
            apsp = dict(apsp)
            
            accepted = [k for k in apsp.keys() if apsp.keys()]
            APSP = {k: apsp[k].keys() for k in accepted}
            
            smoothed = tNLM.smooth(rest,APSP,h)
            outF = h5py.File(outFile,mode='w')
            outF.create_dataset('tnlm',data=smoothed)
            outF.close()    