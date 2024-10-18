import numpy as np
import multiprocessing as mp
import time, os, sys
import SB as model
import numpy.linalg as LA
import subprocess as sp
import random
from numba import njit


def getGlobalParams():
    global dtE, dtI, NSTEPS, NTRAJ, NSTATES, M
    global NCPUS, initstate, dirName
    global fs_to_au
    dtE = model.parameters.dtE
    dtI = model.parameters.dtI
    NSTEPS = model.parameters.NSTEPS
    NTRAJ = model.parameters.NTRAJ
    NSTATES = model.parameters.NSTATES
    M = model.parameters.M
    NCPUS = model.parameters.NCPUS
    initstate = model.parameters.initState
    dirName = model.parameters.dirName
    fs_to_au = 41.341 # a.u./fs

def cleanMainDir():
    if ( os.path.exists(dirName) ):
        sp.call("rm -r "+dirName,shell=True)
    sp.call("mkdir "+dirName,shell=True)

def cleanDir(traj):
    if ( os.path.exists(f"{dirName}/traj-{traj}" ) ):
        sp.call(f"rm -r {dirName}/traj-{traj}",shell=True)
    sp.call(f"mkdir {dirName}/traj-{traj}",shell=True)


def writeDATA(R,V,z,E,dE,NACR,traj):

    TIME = np.arange(0,NSTEPS*dtI,dtI)
    POP  = 0.500 * (np.abs(z)**2 - gw)
    np.savetxt(f"{dirName}/traj-{traj}/TIME.dat", np.c_[TIME,TIME/41.341], fmt="%1.5f")
    np.savetxt(f"{dirName}/traj-{traj}/population.dat", np.c_[TIME,TIME/41.341,POP], fmt="%1.5f")
    np.save(f"{dirName}/traj-{traj}/R.npy", R)
    np.save(f"{dirName}/traj-{traj}/V.npy", V)
    np.save( f"{dirName}/traj-{traj}/E.npy", E )
    np.save( f"{dirName}/traj-{traj}/dE.npy", dE )
    np.save( f"{dirName}/traj-{traj}/NACR.npy", NACR )



# Initialization of the mapping Variables
def initMapping():
    """
    Returns np.array z (complex)
    """
    global gw # Only depends on the number of states. So okay to be global

    Rw = 2*np.sqrt(NSTATES+1) # Radius of W Sphere
    gw = (2/NSTATES) * (np.sqrt(NSTATES + 1) - 1)

    r = np.ones((NSTATES)) * np.sqrt(gw)
    r[initstate] = np.sqrt( 2 + gw ) # Choose initial state

    z = np.zeros((NSTEPS,NSTATES),dtype=complex)
    for i in range(NSTATES):
        phi = random.random() * 2 * np.pi # Azimuthal Angle -- Always Random
        z[0,i] = r[i] * ( np.cos( phi ) + 1j * np.sin( phi ) )
    return z

@njit()
def propagateMapVars(z, V):
    """
    Updates mapping variables
    """
        
    Zreal = np.real(z).copy()
    Zimag = np.imag(z).copy()

    Zimag -= 0.5 * V @ Zreal * dtE
    Zreal +=       V @ Zimag * dtE
    Zimag -= 0.5 * V @ Zreal * dtE

    return  Zreal + 1j*Zimag

@njit
def Force(dHel, R, z, dHel0):
    F = -1 * dHel0
    RHO = 0.500 * ( np.outer(z.conj(), z) - gw*np.eye(NSTATES) ).real
    for j in range(NSTATES):
        F -= dHel[j,j,:] * RHO[j,j]
        for k in range(j+1,NSTATES): # Double counting off-diagonal
            F -= 2 * dHel[j,k,:] * RHO[j,k]
    return F

def get_dE_NACR( E, U, dHel, dHel0 ):
    dE   = np.zeros(( NSTATES, dHel.shape[-1] ))
    NACR = np.zeros_like(dHel)

    for i in range(NSTATES):
        dHel[i,i,:] += dHel0[:]

    dHel_ad = np.einsum("xj,xyR,yk->jkR", U, dHel, U)
    for i in range(NSTATES):
        for j in range(NSTATES):
            if ( i != j ):
                NACR[i,j,:] = dHel_ad[i,j,:] / (E[j] - E[i])
            else:
                dE[i,:] = dHel_ad[i,i,:]
    return dE, NACR

def VelVerF(R, V, z, F0):

    R += dtI * V + 0.500 * F0 * dtI**2 / M
    Hel,dHel,dHel0 = model.Hel(R), model.dHel(R), model.dHel0(R)
    F1 = Force(dHel, R, z, dHel0)
    V += 0.5000 * (F0+F1) * dtI / M
    for t in range( model.parameters.EStep ): z = propagateMapVars(z, Hel)

    E, U = LA.eigh(Hel)
    dE, NACR = get_dE_NACR( E, U, dHel, dHel0 )
    return R, V, z, E, dE, NACR, F1



def RunIterations( traj ): # This is parallelized already. "Main" for each trajectory.

    print("Running traj = %d" % traj)

    cleanDir(traj)

    R,P = model.initR() # Initialize nuclear DOF
    V   = P/M
    z   = initMapping() # Initialize mapping variables

    rho  = np.zeros(( NSTEPS,NSTATES,NSTATES ))
    E    = np.zeros(( NSTEPS,NSTATES ))
    dE   = np.zeros(( NSTEPS,NSTATES,len(R[0]) ))
    NACR = np.zeros(( NSTEPS,NSTATES,NSTATES,len(R[0]) ))

    VMat    = model.Hel(R[0])
    dHel    = model.dHel(R[0])
    dHel0   = model.dHel0(R[0])
    E[0], U = LA.eigh(VMat)
    dE[0], NACR[0] = get_dE_NACR( E[0], U, dHel, dHel0 )

    F0 = Force(dHel, R[0], z[0], dHel0)
    for step in range(NSTEPS-1):
        #print ("Step:", step)
        R[step+1], V[step+1], z[step+1], E[step+1], dE[step+1], NACR[step+1], F0 = VelVerF(R[step], V[step], z[step], F0)
    writeDATA(R, V, z, E, dE, NACR, traj)
    

def ComputeAverageDensity():
    
    # Extract printed data for each trajectory
    timeAU = np.zeros(( NSTEPS ))

    S = np.zeros(( NSTEPS,NSTATES )) # Population

    for t in range(NTRAJ):
        data    = np.loadtxt(f"{dirName}/traj-{t}/population.dat")
        timeAU  = data[:,0]
        timeFS  = data[:,1]
        S      += data[:,2:]

    # Take average over all trajectories for each step
    S /= NTRAJ

    # Normalize population at each step and print
    NORM = np.einsum( "tS->t", S )
    S    = np.einsum( "tS,t->tS", S, 1/NORM )

    np.savetxt(f"{dirName}/population.dat", np.c_[timeAU,timeFS,S], fmt="%1.5f")


### Start Main Program ###
if ( __name__ == "__main__"  ):

    getGlobalParams()
    cleanMainDir()
    start = time.time()

    print (f"There will be {NCPUS} cores with {NTRAJ} trajectories.")

    runList = np.arange(NTRAJ)
    with mp.Pool(processes=NCPUS) as pool:
        pool.map(RunIterations,runList)

    stop = time.time()
    print (f"Total Computation Time (Hours): {(stop - start) / 3600}")

    ComputeAverageDensity()
