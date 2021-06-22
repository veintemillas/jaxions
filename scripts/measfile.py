from pyaxions import jaxions as pa
import numpy as np

a = pa.mli()

a.N   = 256
a.L   = 6.0
a.msa = 1.0
a.ctend= 3.5

# Scaling measurements
a.clear()
a.me_adds("string")
a.me_adds("energy")
a.nmt_prt()
a.addset(9*4,0.0,9.0,0,'logi')

# Refine close to DM annihilation
a.clear()
a.me_adds("string")
a.me_adds("energy")
a.me_adds("bin theta")
a.me_adds("bin rho")
a.me_adds("bin logtheta")
a.me_adds("bin contrast")
a.nmt_prt()
a.addset(20,1.8,2.8,0,'lin')

# A spectrum from time to time
a.clear()
a.me_adds("string")
a.me_adds("bin theta")
a.me_adds("bin rho")
a.me_adds("bin logtheta")
a.me_adds("bin contrast")
a.me_adds("plot2D")
# default is mask flat KGVS
a.me_adds("NSP_A")
# default is mask flat
a.me_adds("PSP_A")
a.me_addmask("FLAT")
a.me_addmask("AXITV")
a.nmt_prt()
a.addset(7*2,2.0,9.0,0,'logi')

# Refine spectra during the NR period
a.clear()
a.me_adds("string")
a.me_adds("bin theta")
a.me_adds("bin rho")
a.me_adds("bin logtheta")
a.me_adds("bin contrast")
a.me_addmap("slices XY")
a.me_adds("NSP_A")
a.me_adds("PSP_A")
a.me_addmask("FLAT")
a.me_addmask("AXITV")
a.nmt_prt()
a.addset(25,1.5,4.0,0,'lin')

# Final dump
a.clear()
#a.me_adds("plot2D")
a.me_adds("bin theta")
a.me_adds("bin rho")
a.me_adds("bin logtheta")
a.me_adds("bin contrast")
a.me_adds("NSP_A")
a.me_adds("PSP_A")
a.me_addmask("FLAT")
a.me_addmask("AXITV")
a.nmt_prt()
a.addset(1,a.ctend,a.ctend,0,'lin')

a.give()

# 'MEAS_NOTHING': 0,
# 'MEAS_BINTHETA': 1,
# 'MEAS_BINRHO': 2,
# 'MEAS_BINLOGTHETA2': 4,
# 'MEAS_BINDELTA': 8,
# 'MEAS_EMPTY': 16,
# 'MEAS_STRING': 32,
# 'MEAS_STRINGMAP': 64,
# 'MEAS_STRINGCOO': 128,
# 'MEAS_ENERGY': 256,
# 'MEAS_ENERGY3DMAP': 512,
# 'MEAS_REDENE3DMAP': 1024,
# 'MEAS_2DMAP': 2048,
# 'MEAS_3DMAP': 4096,
# 'MEAS_MASK': 8192,
# 'MEAS_PSP_A': 16384,
# 'MEAS_PSP_S': 32768,
# 'MEAS_NSP_A': 65536,
# 'MEAS_NSP_S': 131072,
# 'MEAS_NNSPEC': 262144
