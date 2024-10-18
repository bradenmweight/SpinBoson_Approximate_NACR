import numpy as np
import random

from numba import njit


class parameters():
    NCPUS = 24
    dtI = 0.005
    NSTEPS = int(20/dtI) # Spin boson models go to 20 a.u.
    NTRAJ = 24
    EStep = 200
    dtE = dtI/EStep

    M         = 1
    NSTATES   = 2
    initState = 0

    fs_to_au = 41.341 # a.u./fs

    NDOF    = 5
    dirName = "NDOF_%1.0f/" % NDOF

@njit
def initModelParams(NDOF):
    c = np.ones(( NDOF )) * 0.1
    w = np.ones(( NDOF )) * 3.0
    return c,w

@njit
def Hel(R):
    # SPIN BOSON MODEL PARAMETERS
    Delta = 1.0 # Off-digaonal coupling
    wm    = 2.0 #
    Eps   = 0.0 # Bias
    xi    = 0.09 # Friction
    beta  = 0.1  # Inverse temperature
    wc    = 5.0  # Cutoff frequency
    NDOF  = len(R)
    c, w  = initModelParams(NDOF)

    Hel = np.zeros((2,2))
    Hel[0,0] = Eps
    Hel[0,1] = Delta
    Hel[1,0] = Hel[0,1]
    Hel[1,1] = -Eps

    Hel[0,0]  +=  np.sum( c * R )
    Hel[1,1]  -=  np.sum( c * R )

    return Hel
  

def dHel0(R):
    c,w = initModelParams(len(R))
    return w**2 * R

@njit
def dHel(R):
    dHel = np.zeros(( 2,2,len(R) ))
    c,w = initModelParams(len(R))
    dHel[0,0,:] = c
    dHel[1,1,:] = -c
    return dHel

def initR():
    
    # SPIN BOSON MODEL PARAMETERS
    Delta = 1.0 # Off-digaonal coupling
    wm    = 2.0 #
    Eps   = 0.0 # Bias
    xi    = 0.09 # Friction
    beta  = 0.1  # Inverse temperature
    wc    = 5.0  # Cutoff frequency
    NSTEPS  = parameters.NSTEPS
    NDOF    = parameters.NDOF
    c, w    = initModelParams(NDOF)

    R = np.zeros(( NSTEPS, NDOF ))
    P = np.zeros(( NSTEPS, NDOF ))
    R0 = 0.0
    P0 = 0.0
    for d in range(NDOF):
        sigp = np.sqrt( w[d] / ( 2 * np.tanh( 0.5*beta*w[d] ) ) )
        sigq = sigp / w[d]

        R[0,d] = random.gauss(R0,sigq)
        P[0,d] = random.gauss(P0,sigp)
   
    return R,P

