from pyaxions import jaxions as pa
import numpy as np

a = pa.mli()

a.N   = 256
a.L   = 6.0
a.msa = 1.0
a.ctend= 4.0

# Scaling measurements
a.me  = 0
a.me_adds("string")
a.me_adds("energy")
a.nmt_prt()
a.addset(7*4,1.0,8.0,0,'logi')

# Refine close to DM annihilation
a.me  = 0
a.me_adds("string")
a.me_adds("energy")
a.nmt_prt()
a.addset(1.0*10,1.8,2.8,0,'lin')

# A spectrum from time to time
a.me  = 0
a.me_adds("string")
a.me_adds("plot2D")
a.me_adds("NSP_A")
a.me_adds("PSP_A")
a.nmt_prt()
a.addset(6*2,2.0,8.0,0,'logi')

# Refine spectra during the NR period
a.me  = 0
a.me_adds("string")
a.me_adds("plot2D")
a.me_adds("NSP_A")
a.me_adds("PSP_A")
a.nmt_prt()
a.addset(20,1.5,4.0,0,'lin')

# Final dump
a.me  = 0
a.me_adds("plot2D")
# a.me_adds("plot3D") # better specify this with --p3D 2 (final) or --p3D 6 (final+wkb)
a.me_adds("reduced energy map")
a.me_adds("NSP_A")
a.me_adds("PSP_A")
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
