import numpy as np
from matplotlib import pyplot as plt
from glob import glob


def read_data():
    global NDOF_LIST, NTRAJ, NSTEPS, NSTATES

    NDOF_FOLDERS = np.array(glob("NDOF_*"))
    NDOF_LIST    = np.array([ int(F.strip("NDOF_")) for F in NDOF_FOLDERS ])
    NDOF_FOLDERS = NDOF_FOLDERS[np.argsort(NDOF_LIST) ]
    NDOF_LIST    = NDOF_LIST[np.argsort(NDOF_LIST) ]

    #NDOF_FOLDERS = NDOF_FOLDERS[:2]
    #NDOF_LIST    = NDOF_LIST[:2]

    print("FOLDERS: ",NDOF_FOLDERS)
    print("DOF List:",NDOF_LIST)

    V    = {}
    dE   = {}
    NACR = {}
    for Fi,F in enumerate( NDOF_FOLDERS ):
        NDOF = NDOF_LIST[Fi]
        NTRAJ = len(glob(f"{F}/traj-*"))
        print( f"Reading data: {F}/, {NTRAJ} trajectories")
        V_TMP    = []
        dE_TMP   = []
        NACR_TMP = []
        for traj in range(1): #range( NTRAJ ):
            V_TMP   .append( np.load(f"{F}/traj-{traj}/V.npy"   ) )
            dE_TMP  .append( np.load(f"{F}/traj-{traj}/dE.npy"  ) )
            NACR_TMP.append( np.load(f"{F}/traj-{traj}/NACR.npy")[:,0,1,:] )
        V[NDOF]    = np.array(V_TMP)
        dE[NDOF]   = np.array(dE_TMP)
        NACR[NDOF] = np.array(NACR_TMP)
    print("V.npy")
    for key, item in V.items():
        print("DOF:", key, "SHAPE:", item.shape)
    print("dE.npy")
    for key, item in dE.items():
        print("DOF:", key, "SHAPE:", item.shape)
    print("NACR.npy")
    for key, item in NACR.items():
        print("DOF:", key, "SHAPE:", item.shape)

    NTRAJ, NSTEPS, NSTATES, _ = dE[NDOF_LIST[0]].shape

    NACT = {}
    for NDOF in NDOF_LIST:
        NACT[NDOF] = np.zeros(( NTRAJ, NSTEPS ))
        NACT[NDOF] = np.einsum("Ttd,Ttd->Tt", NACR[NDOF][:,:,:], V[NDOF])
    print("NACT")
    for key, item in NACT.items():
        print("DOF:", key, "SHAPE:", item.shape)

    DIFF_GRAD = {}
    for NDOF in NDOF_LIST:
        DIFF_GRAD[NDOF] = np.zeros(( NTRAJ, NSTEPS, NDOF ))
        DIFF_GRAD[NDOF][:,:,:] = dE[NDOF][:,:,0,:] - dE[NDOF][:,:,1,:]
    print("DIFF_GRAD")
    for key, item in DIFF_GRAD.items():
        print("DOF:", key, "SHAPE:", item.shape)

    return V, dE, NACR, NACT, DIFF_GRAD

def get_NACR_APPROX( V, NACT, DIFF_GRAD ):
    """
    G     = DIFF_GRAD + alpha . \\hat{V}
    alpha = NACT - \\hat{V} . DIFF_GRAD
    """
    NACR_APPROX = {}
    for NDOF in NDOF_LIST:
        V2    = np.einsum("Ttd->Tt", V[NDOF]**2)
        alpha = NACT[NDOF] - np.einsum("Ttd,Ttd->Tt", DIFF_GRAD[NDOF], V[NDOF])
        alpha = np.einsum("Tt,Tt->Tt", alpha, 1/V2)
        NACR_APPROX[NDOF] = DIFF_GRAD[NDOF] + np.einsum("Tt,Ttd->Ttd", alpha, V[NDOF])

    return NACR_APPROX

def get_VELOC_Localization( V ):
    IPR_V = {}

    for NDOF in NDOF_LIST:

        PROB        = V[NDOF]**2
        NUMER       = np.einsum( "Ttd->Tt", PROB )**2
        DENOM       = np.einsum( "Ttd->Tt", PROB**2 )
        IPR_V[NDOF] = np.einsum( "Tt,Tt->Tt", NUMER, 1 / DENOM )

    return IPR_V

def main():
    V, dE, NACR, NACT, DIFF_GRAD = read_data()
    NACR_APPROX                  = get_NACR_APPROX( V, NACT, DIFF_GRAD )
    IPR_V                        = get_VELOC_Localization( V )







    color_list = ["black", "red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "cyan"]*2

    for NDOFi,NDOF in enumerate( NDOF_LIST ):
        RMS_EXACT  = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", NACR[NDOF]**2 ) )
        RMS_APPROX = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", NACR_APPROX[NDOF]**2 ) )
        RMS_VELOC  = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", V[NDOF]**2 ) )
        print( np.max(RMS_EXACT), np.max(RMS_APPROX) )
        for traj in range(NTRAJ):
            if ( traj == 0 ):
                plt.plot( RMS_EXACT[traj,:500], alpha=0.5, lw=6, c=color_list[NDOFi], label=f"DOF={NDOF} (EXACT)")
                plt.plot( RMS_APPROX[traj,:500], c=color_list[NDOFi])#, label=f"DOF={NDOF} (APPROX)")
            else:
                plt.plot( RMS_EXACT[traj,:500], alpha=0.5, lw=6, c=color_list[NDOFi])
                plt.plot( RMS_APPROX[traj,:500], c=color_list[NDOFi] )
            #scale = 1 #np.max(RMS_APPROX[traj,:500]) - np.min(RMS_APPROX[traj,:500])
            #scale1 = 1 #np.max(RMS_VELOC[traj,:500]) - np.min(RMS_VELOC[traj,:500])
            #plt.plot( np.min(RMS_APPROX[traj,:500]) + scale * (RMS_VELOC[traj,:500] - np.average(RMS_VELOC[traj,:500])) / scale1, "--", c=color_list[NDOFi] )
            #plt.plot( RMS_VELOC[traj,:500], "--", c=color_list[NDOFi] )
    plt.legend()
    #plt.ylim(0.0)
    plt.savefig("RMS_EXACT_APPROX.jpg",  dpi=300)
    plt.clf()


    for NDOFi,NDOF in enumerate( NDOF_LIST ):
        RMS_EXACT  = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", NACR[NDOF]**2 ) )
        RMS_APPROX = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", NACR_APPROX[NDOF]**2 ) )
        for traj in range(NTRAJ):
            if ( traj == 0 ):
                plt.scatter( RMS_EXACT[traj,:500], RMS_APPROX[traj,:500], s=2, c=color_list[NDOFi], label=f"DOF={NDOF} (APPROX)")
            else:
                plt.scatter( RMS_EXACT[traj,:500], RMS_APPROX[traj,:500], s=2, c=color_list[NDOFi])#, label=f"DOF={NDOF} (APPROX)")
    plt.legend()
    #plt.ylim(0.0)
    plt.savefig("RMS_APPROX_vs_EXACT.jpg",  dpi=300)
    plt.clf()

    for NDOFi,NDOF in enumerate( NDOF_LIST ):
        RMSD       = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", (np.abs(NACR[NDOF]) - np.abs(NACR_APPROX[NDOF]))**2 ) )
        RMS_EXACT  = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", NACR[NDOF]**2 ) )
        for traj in range(NTRAJ):
            if ( traj == 0 ):
                plt.scatter( RMS_EXACT[traj,:500], RMSD[traj,:500], s=2, c=color_list[NDOFi], label=f"DOF={NDOF} (APPROX)")
            else:
                plt.scatter( RMS_EXACT[traj,:500], RMSD[traj,:500], s=2, c=color_list[NDOFi])#, label=f"DOF={NDOF} (APPROX)")
    plt.legend()
    #plt.ylim(0.0)
    plt.savefig("RMSD_vs_RMS_EXACT.jpg",  dpi=300)
    plt.clf()

    for NDOFi,NDOF in enumerate( NDOF_LIST ):
        RMSD       = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", (np.abs(NACR[NDOF]) - np.abs(NACR_APPROX[NDOF]))**2 ) )
        RMS_VELOC  = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", V[NDOF]**2 ) )
        for traj in range(NTRAJ):
            if ( traj == 0 ):
                plt.scatter( RMS_VELOC[traj,:500], RMSD[traj,:500], s=2, c=color_list[NDOFi], label=f"DOF={NDOF} (APPROX)")
            else:
                plt.scatter( RMS_VELOC[traj,:500], RMSD[traj,:500], s=2, c=color_list[NDOFi])#, label=f"DOF={NDOF} (APPROX)")
    plt.legend()
    #plt.ylim(0.0)
    plt.savefig("RMSD_vs_RMS_VELOC.jpg",  dpi=300)
    plt.clf()

    for NDOFi,NDOF in enumerate( NDOF_LIST ):
        RMSD       = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", (np.abs(NACR[NDOF]) - np.abs(NACR_APPROX[NDOF]))**2 ) )
        RMS_VELOC  = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", V[NDOF]**2 ) )
        for traj in range(NTRAJ):
            if ( traj == 0 ):
                plt.semilogx( 1/RMS_VELOC[traj,:500], RMSD[traj,:500], "o", ms=2, c=color_list[NDOFi], label=f"DOF={NDOF} (APPROX)")
            else:
                plt.semilogx( 1/RMS_VELOC[traj,:500], RMSD[traj,:500], "o", ms=2, c=color_list[NDOFi])#, label=f"DOF={NDOF} (APPROX)")
    plt.legend()
    #plt.ylim(0.0)
    plt.savefig("RMSD_vs_inv_RMS_VELOC.jpg",  dpi=300)
    plt.clf()

    for NDOFi,NDOF in enumerate( NDOF_LIST ):
        RMSD       = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", (np.abs(NACR[NDOF]) - np.abs(NACR_APPROX[NDOF]))**2 ) )
        for traj in range(NTRAJ):
            if ( traj == 0 ):
                plt.plot( IPR_V[NDOF][traj,:500], RMSD[traj,:500], "o", ms=2, c=color_list[NDOFi], label=f"DOF={NDOF} (APPROX)")
            else:
                plt.plot( IPR_V[NDOF][traj,:500], RMSD[traj,:500], "o", ms=2, c=color_list[NDOFi])#, label=f"DOF={NDOF} (APPROX)")
    plt.legend()
    #plt.ylim(0.0)
    plt.savefig("RMSD_vs_IPR_V.jpg",  dpi=300)
    plt.clf()


    RMSRMSD = np.zeros(( len(NDOF_LIST) ))
    AVERMSD = np.zeros(( len(NDOF_LIST) ))
    for NDOFi,NDOF in enumerate( NDOF_LIST ):
        RMSD           = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", (np.abs(NACR[NDOF]) - np.abs(NACR_APPROX[NDOF]))**2 ) )
        RMSRMSD[NDOFi] = np.sqrt( (1/NTRAJ/NSTEPS) * np.einsum("Tt->", (RMSD)**2 ) )
        AVERMSD[NDOFi] = np.sum( RMSD ) /NTRAJ/NSTEPS
    
    plt.plot( NDOF_LIST, RMSRMSD, "-o", label="RMS$[$ RMSD $]_\\mathrm{trajectory,time}$")
    plt.plot( NDOF_LIST, AVERMSD, "-o", label="$\\langle$ RMSD $\\rangle_\\mathrm{trajectory,time}$")
    plt.xlabel("N$_\\mathrm{DOF}$", fontsize=15)
    plt.ylabel("Error Estimate", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig("RMSRMSD_vs_NDOF.jpg",  dpi=300)
    plt.clf()


    RMSRMSD = np.zeros(( len(NDOF_LIST) ))
    AVERMSD = np.zeros(( len(NDOF_LIST) ))
    for NDOFi,NDOF in enumerate( NDOF_LIST ):
        RMSD           = np.sqrt( (1/NDOF) * np.einsum("Ttd->Tt", (np.abs(NACR[NDOF]) - np.abs(NACR_APPROX[NDOF]))**2 ) )
        RMSRMSD[NDOFi] = np.sqrt( (1/NTRAJ/NSTEPS) * np.einsum("Tt->", (RMSD)**2 ) )/np.sqrt(NDOF)
        AVERMSD[NDOFi] = np.sum( RMSD ) /NTRAJ/NSTEPS/np.sqrt(NDOF)
    
    plt.plot( NDOF_LIST, RMSRMSD, "-o", label="RMS$[$ RMSD $]_\\mathrm{trajectory,time}$")
    plt.plot( NDOF_LIST, AVERMSD, "-o", label="$\\langle$ RMSD $\\rangle_\\mathrm{trajectory,time}$")
    plt.xlabel("N$_\\mathrm{DOF}$", fontsize=15)
    plt.ylabel("Error Estimate / $\\sqrt{N_\\mathrm{DOF}}$", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig("RMSRMSD_vs_NDOF_NORMED.jpg",  dpi=300)
    plt.clf()


    for NDOFi,NDOF in enumerate( NDOF_LIST ):
        INNER      = np.einsum( "Ttd,Ttd->Tt", NACR[NDOF], NACR_APPROX[NDOF] )
        NORM1      = np.sqrt( np.einsum( "Ttd->Tt", NACR[NDOF]**2 ) )
        NORM2      = np.sqrt( np.einsum( "Ttd->Tt", NACR_APPROX[NDOF]**2 ) )
        ANGLE      = np.arccos( INNER/NORM1/NORM2 ) / np.pi * 180
        for traj in range(NTRAJ):
            if ( traj == 0 ):
                plt.plot( ANGLE[traj,:500], "o", ms=2, c=color_list[NDOFi], label=f"DOF={NDOF}")
            else:
                plt.plot( ANGLE[traj,:500], "o", ms=2, c=color_list[NDOFi])
    plt.xlabel("Time Step", fontsize=15)
    plt.ylabel("Angle, $\\theta_\\mathrm{Exact,Approx}$", fontsize=15)
    plt.legend()
    #plt.ylim(0.0)
    plt.savefig("ANGLE_EXACT_APPROX.jpg",  dpi=300)
    plt.clf()


if ( __name__ == "__main__" ):
    main()