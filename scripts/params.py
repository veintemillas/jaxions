# =============================================================================================
#      params.py 
#      
#      This scripts check the parameters of simulation before running. Parameters to parse are:
#      
#      - Axion mass index: n
#      - Lattice size: N
#      - Physical size: L
#      - String widht/tension: msa
#      - Final time: zf
#      - Reference momentum scale: kref
#
#      In addition, if you parse --stdn [float] the program will display some standard 
#      simulations with a chosen axion mass index n 
#      
#      Examples : $ python3 params.py --n 4.0 --N 256 --L 4.0 --zf 2.0 --msa 1.0 --kref 100.0
#                 $ python3 params.py --stdn 7.0  
#                 $ python3 params.py --help                (for options)
#          
# =============================================================================================

import numpy as np
from scipy.integrate import quad

def LambdaPRS(N,L,msa):
	LambdaPRS = 0.5*(msa*N/L)**2
	return LambdaPRS

def findtau2(n,N):
	kappa = np.log(N/2)
	tau2 = (np.pi*kappa/4)**(2/(n+4))
	return tau2

def findzdoom(n,N,L,msa):
	zdoom = (LambdaPRS(N,L,msa)/40)**(1/(n+2))
	return zdoom

def findtauNy(n,N,L):
	tauNy = (N*np.pi/L)**(2/(n+2))
	return tauNy

def findtauNy4(n,N,L):
	tauNy4 = (N*np.pi/(4*L))**(2/(n+2))
	return tauNy4

def findmaa(n,N,L,zf):
    maa = zf**(n/2+1)*L/N
    return maa

def findtau3(n,N,L):
    tau3 = (N/L)**(2/(n+2))
    return tau3

def findtauref(k,n):
    tauref = k**(2/(n+2))
    return tauref

def findfsmax(n,N,L):
	fsmax = (1.55+n)/n*(np.pi*N/L)**(2/(n+2))
	return fsmax

def findfsref(n,kref):
	fsref = (1.55+n)/n*kref**(2/(n+2))
	return fsref

def conf_vel(x,n,kref):
    return (1/(1 + x**(n + 2)/kref**2))**(1/2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="---------- CHECK YOUR SIMULATION SETTINGS ----------------")
    
    parser.add_argument(
	    "--n",
	    dest="n",
	    help="choose axion mass power law index n",
	    default="4.0",
	    type=float,
	    required=False
	)

    parser.add_argument(
	    "--N",
	    dest="N",
	    help="choose lattice size N",
	    default="128",
	    type=float,
	    required=False
	)

    parser.add_argument(
	    "--L",
	    dest="L",
	    help="choose physical size L",
	    default="4.0",
	    type=float,
	    required=False
	)

    parser.add_argument(
	    "--zf",
	    dest="zf",
	    help="choose a final evolution time zf",
	    default="2.0",
	    type=float,
	    required=False
	)

    parser.add_argument(
	    "--msa",
	    dest="msa",
	    help="choose string width msa",
	    default="1.0",
	    type=float,
	    required=False
	)
	
    parser.add_argument(
	    "--kref",
	    dest="kref",
	    help="choose a reference momentum",
	    default="100",
	    type=float,
	    required=False
	)

    parser.add_argument(
		"--stdn",
		dest="stdn",
		help="parse n to find out a standard simulation",
		type=float,
		required=False
	)

    args = parser.parse_args()

# ##############################################################################################################################
# SOME STANDARD SIMULATIONS (to complete)
# ##############################################################################################################################

    if args.stdn == 7.0:

        print("\nSTANDARD SIMULATION WITH n = 7.0")
        print("=================================")
        print("\nN = 4096,  L=6,  zf=4.5,  msa=1.0")
        print("N = 8192,  L=6,  zf=4.5,  msa=1.0\n ")
        exit(0)

    if args.stdn == 4.0:

        print("\nSTANDARD SIMULATIONS WITH n = 4.0")
        print("=================================")
        print("\nN = 2048,  L=10,  zf=5.5,  msa=1.0 ")
        print("N = 3072,  L=12,  zf=8.0,  msa=1.0\n ")
        print("High resolution/memory:\n ")
        print("N = 4096,  L=10,  zf=5.5,  msa=1.0\n ")

        exit(0)


# ##############################################################################################################################
# QCD AXION QUICK WARNINGS
# ##############################################################################################################################

    if args.n == 7.0:
    	
    	print("\nQCD axion chosen!\n")

    	if args.N < 3072:

    		print("WARNING! The lattice size is too small to avoid unphysical effects, the parameter N must be at least 3072. \n")
    		exit(1)

    	if args.N >= 3072:

    		if args.L < 6:

    			print("WARNING! The physical size is too small to avoid finite volume effects, the parameter L must be larger than 6. \n")
    			exit(1)


# ##############################################################################################################################
# OUTPUT 
# ############################################################################################################################## 
    I = quad(conf_vel, 0, args.zf, args=(args.n,args.kref))
    
    fsscale = I[0]
    tau2 = findtau2(args.n,args.N)
    tau3 = findtau3(args.n,args.N,args.L)
    tauNy = findtauNy(args.n,args.N,args.L)
    tauNy4 = findtauNy4(args.n,args.N,args.L)




    print("\n---------------------------------------------------------------------------")
    print("           CHOSEN PARAMETERS                                                 ")
    print("---------------------------------------------------------------------------\n")

    print("Axion mass index: n =",(args.n))
    print("Lattice size: N =",(args.N))
    print("Physical size: L =",(args.L))
    print("Final evolution time: zf =",(args.zf))
    print("String width : msa =",(args.msa))
    print("Reference momentum : kref =",(args.kref),"/ L1\n")

    print("\nMemory requirements:", format(24*args.N**3*1e-6,'.2f'), "MB =", format(24*args.N**3*1e-9,'.2f'),"GB.\n")

    print("\nSTRINGS AND WALLS")
    print("=================  \n")
    print("String tension:", "{:e}".format((LambdaPRS(args.N,args.L,args.msa)),'.2f'))
    print("\nDomain walls expected to dominate at: t =", format(findtau2(args.n,args.N),'.2f'))
    print("String expected to disappear at: t =", format(2*findtau2(args.n,args.N),'.2f'))
    print("\nUnphysical effects for strings/walls expected at t =", format(findzdoom(args.n,args.N,args.L,args.msa),'.2f'),"\n")
    print("\nAXION FREE-STREAMING")
    print("====================\n")
    print("Nyquist frequency: kL1 =" , format(np.pi*args.N/args.L,'.2f'))
    print("Fastest axion is NR at : t = ", format((findtauNy(args.n,args.N,args.L)),'.2f'))
    #print("Free-streaming scale of fastest axion: \u03BB =", format((findfsmax(args.n,args.N,args.L)),'.2f'))
    print("\nAxions with momentum k_Ny/4 are NR at: t =",  format((findtauNy4(args.n,args.N,args.L)),'.2f'))
    print("Axions with reference momentum are NR at: t =",  format((findtauref(args.kref,args.n)),'.2f'))
    #print("Free-streaming scale of k_Ny/4 axion: \u03BB =", format((findfsmax(args.n,args.N,args.L*2)),'.2f'))
    print("\nFree-streaming scale of reference momentum axion: \u03BB =", format(fsscale,'.2f'),"\n")
    print("\nAXITONS")
    print("======= \n")
    print("Axiton core size at final simulation time selected is: 1/ma*a =", format(1/(findmaa(args.n,args.N,args.L,args.zf)),'.2f'))
    print("Axiton core reaches ma*a = 1.00 at: t =",format((findtau3(args.n,args.N,args.L)),'.2f'))

    print("\nEVOLUTION")
    print("========= \n")
    print("Suggested final evolution time:", format((findtauNy4(args.n,args.N,args.L)),'.2f'), "< t <", 
             format((findtau3(args.n,args.N,args.L)),'.2f'))
    print("Minimum WKB time:", format((findtauNy(args.n,args.N,args.L)),'.2f'))

    print("\n---------------------------------------------------------------------------")
    print("            WARNINGS                                                         ")
    print("---------------------------------------------------------------------------\n")

    nowarnings = True

    if 2*findtau2(args.n,args.N) > findzdoom(args.n,args.N,args.L,args.msa):
        nowarnings = False
        print("- Unphysical destruction appears when the strings/walls are still present!\n  To solve this issue increase the parameter --N or the parameter --msa\n")

    if args.zf < args.L/2:
        if args.zf < findtauNy(args.n,args.N,args.L):
            if args.zf < tau3:
                nowarnings = False
                if args.L/2 < findtauNy(args.n,args.N,args.L):
                    print("- You could evolve at least to zf =", 0.5*args.L,"\n")
                else:
                    print("- You could evolve to zf =", format(min(tau3,tauNy4),'.2f'),"\n")
    
    if args.zf > tau3:
        print("- Your axiton radius overcomes the lattice resolution!\n  The core reaches the lattice spacing at: t =",format(tau3,'.2f'),"\n")

    if args.zf > findtauNy(args.n,args.N,args.L):
    	nowarnings = False
    	print("- Your final evolution time goes beyond the time where all axions are NR,\n  you could be wasting computational resources!\n")

    if args.L < 2*fsscale:
        if args.L == int(2*fsscale):
            nowarnings = True
        else:
            nowarnings = False
            print("- Compare L with the reference momentum free-streaming scale.\n  Ideally L should be larger than twice that scale, i.e. L >"
    		,format(2*fsscale,'.1f'),"\n")

    
    

    if nowarnings:
    	print("No warnings found, your simulation is ready to run!\n")









