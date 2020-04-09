#!/usr/bin/python3


prefix='V_'
l_QCD=['0','1','2','C','V','L']
l_PQ=['1','2']
l_ex=['RHO','DRHO','DRHO_RHO','DALL']

PR={'PROP_OMELYAN2':'PropOmelyan2', 'PROP_OMELYAN4':'PropOmelyan4',
    'PROP_LEAP':'PropLeap', 'PROP_MLEAP':'PropMLeap', 'PROP_RKN4':'PropRKN4'}

def tempCPU(vqcd,classname):
    print('\t\tcase %s: '%vqcd)
    print('\t\t\t prop = std::make_unique<%s<%s>>(field, propclass);'%(classname,vqcd))
    print('\t\tbreak; \n ')

def multicase(prop):
    print('case %s: '%prop)
    print('LogMsg(VERB_NORMAL,"[ip] %s chosen");'%prop[5:])
    print('\tswitch (pot) {')
    for q in l_QCD:
        for p in l_PQ:
            vqcd=prefix+'QCD'+q+'_PQ'+p
            tempCPU(vqcd,PR[prop])
            for e in l_ex:
                vqcd=prefix+'QCD'+q+'_PQ'+p+'_'+e
                tempCPU(vqcd,PR[prop])
    print('} \n break; \n')

def CASECPU(vqcd,classname):
    print('\t CASE(aPT,%s) \\'%vqcd)

def multiCASECPU(prop):
    print('case PT1:          \\')
    print('switch (pot) {     \\')
    for q in l_QCD:
        for p in l_PQ:
            vqcd=prefix+'QCD'+q+'_PQ'+p
            CASECPU(vqcd,PR[prop])
            for e in l_ex:
                vqcd=prefix+'QCD'+q+'_PQ'+p+'_'+e
                CASECPU(vqcd,PR[prop])
    print('} \\\n break; \n')

# run this to get multiCASECPU
# multiCASECPU('PROP_LEAP')

# tun this to get all explicit templates
# for prop in [k for k in PR.keys()]:
#     multicase(prop)
