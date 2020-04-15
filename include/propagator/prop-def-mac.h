#ifndef	_PDM_
	#define	_PDM_

	#include "scalar/scalarField.h"
	#include "enum-field.h"

	/* Used in Cpu propagators */

	#define CASE(PT,x)  \
		case x:																							\
		prop = std::make_unique<PT<x>>		(field, propclass);	\
		break;

	#define DEFALLPROPTEM(aPT)  \
	switch (pot) {     \
	 CASE(aPT,V_QCD0_PQ1) \
	 CASE(aPT,V_QCD0_PQ1_RHO) \
	 CASE(aPT,V_QCD0_PQ1_DRHO) \
	 CASE(aPT,V_QCD0_PQ1_DRHO_RHO) \
	 CASE(aPT,V_QCD0_PQ1_DALL) \
	 CASE(aPT,V_QCD0_PQ2) \
	 CASE(aPT,V_QCD0_PQ2_RHO) \
	 CASE(aPT,V_QCD0_PQ2_DRHO) \
	 CASE(aPT,V_QCD0_PQ2_DRHO_RHO) \
	 CASE(aPT,V_QCD0_PQ2_DALL) \
	 CASE(aPT,V_QCD1_PQ1) \
	 CASE(aPT,V_QCD1_PQ1_RHO) \
	 CASE(aPT,V_QCD1_PQ1_DRHO) \
	 CASE(aPT,V_QCD1_PQ1_DRHO_RHO) \
	 CASE(aPT,V_QCD1_PQ1_DALL) \
	 CASE(aPT,V_QCD1_PQ2) \
	 CASE(aPT,V_QCD1_PQ2_RHO) \
	 CASE(aPT,V_QCD1_PQ2_DRHO) \
	 CASE(aPT,V_QCD1_PQ2_DRHO_RHO) \
	 CASE(aPT,V_QCD1_PQ2_DALL) \
	 CASE(aPT,V_QCD2_PQ1) \
	 CASE(aPT,V_QCD2_PQ1_RHO) \
	 CASE(aPT,V_QCD2_PQ1_DRHO) \
	 CASE(aPT,V_QCD2_PQ1_DRHO_RHO) \
	 CASE(aPT,V_QCD2_PQ1_DALL) \
	 CASE(aPT,V_QCD2_PQ2) \
	 CASE(aPT,V_QCD2_PQ2_RHO) \
	 CASE(aPT,V_QCD2_PQ2_DRHO) \
	 CASE(aPT,V_QCD2_PQ2_DRHO_RHO) \
	 CASE(aPT,V_QCD2_PQ2_DALL) \
	 CASE(aPT,V_QCDC_PQ1) \
	 CASE(aPT,V_QCDC_PQ1_RHO) \
	 CASE(aPT,V_QCDC_PQ1_DRHO) \
	 CASE(aPT,V_QCDC_PQ1_DRHO_RHO) \
	 CASE(aPT,V_QCDC_PQ1_DALL) \
	 CASE(aPT,V_QCDC_PQ2) \
	 CASE(aPT,V_QCDC_PQ2_RHO) \
	 CASE(aPT,V_QCDC_PQ2_DRHO) \
	 CASE(aPT,V_QCDC_PQ2_DRHO_RHO) \
	 CASE(aPT,V_QCDC_PQ2_DALL) \
	 CASE(aPT,V_QCDV_PQ1) \
	 CASE(aPT,V_QCDV_PQ1_RHO) \
	 CASE(aPT,V_QCDV_PQ1_DRHO) \
	 CASE(aPT,V_QCDV_PQ1_DRHO_RHO) \
	 CASE(aPT,V_QCDV_PQ1_DALL) \
	 CASE(aPT,V_QCDV_PQ2) \
	 CASE(aPT,V_QCDV_PQ2_RHO) \
	 CASE(aPT,V_QCDV_PQ2_DRHO) \
	 CASE(aPT,V_QCDV_PQ2_DRHO_RHO) \
	 CASE(aPT,V_QCDV_PQ2_DALL) \
	 CASE(aPT,V_QCDL_PQ1) \
	 CASE(aPT,V_QCDL_PQ1_RHO) \
	 CASE(aPT,V_QCDL_PQ1_DRHO) \
	 CASE(aPT,V_QCDL_PQ1_DRHO_RHO) \
	 CASE(aPT,V_QCDL_PQ1_DALL) \
	 CASE(aPT,V_QCDL_PQ2) \
	 CASE(aPT,V_QCDL_PQ2_RHO) \
	 CASE(aPT,V_QCDL_PQ2_DRHO) \
	 CASE(aPT,V_QCDL_PQ2_DRHO_RHO) \
	 CASE(aPT,V_QCDL_PQ2_DALL) \
	}




	// #define DEFALLPROPTEM(PT1,aPT,pf)  \
	// case PT1:           \
	// 	switch (pf) {     \
	// 		CAZ2(aPT,QCD0) \
	// 		CAZ2(aPT,QCD1) \
	// 		CAZ2(aPT,QCDV) \
	// 		CAZ2(aPT,QCD2) \
	// 		CAZ2(aPT,QCDL) \
	// 		CAZ2(aPT,QCDC) \
	// 	}                 \
	// 	break;
	//
	// #define CAZ2(pts,qcds)  \
	// CAZ3(pts,qcds,PQ1)  \
	// CAZ3(pts,qcds,PQ2)
	//
	// #define CAZ3(pt,qcd,pq)  \
	// CASE(pt,COCA(qcd,pq))  \
	// CASE(pt,COCA(qcd,pq,_RHO))  \
	// CASE(pt,COCA(qcd,pq,_DRHO))  \
	// CASE(pt,COCA(qcd,pq,_DALL))  \
	// CASE(pt,COCA(qcd,pq,_DRHO_RHO))
	//
	// #define COCO(a,b,c)	a##_##b##c
	// #define COCA(a,b)	a##_##b
	//
	// #define CASE(pt,X)  \
	// case V_##X:         \
	// prop = std::make_unique<pt<V_##X>>		(field, propclass);	\
	// break;






	/* Used in Gpu propagators */
	#define PK_GPU(preci,X)  \
	case	V_X:							\
	propagateKernel<preci, V_X>		<<<gridSize,blockSize,0,stream>>> ((const complex<preci> *) m, (complex<preci> *) v, (complex<preci> *) m2, \
											zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (preci) LL, Lx, Sf, Vo, Vf); \
	break;

	#define CAZ3_GPU(pk,preci,qcd,pq)  \
	PK_GPU(pk,preci,qcd##_##pq))  \
	PK_GPU(pk,preci,qcd##_##pq##_RHO))  \
	PK_GPU(pk,preci,qcd##_##pq##_DRHO))  \
	PK_GPU(pk,preci,qcd##_##pq##_DALL))  \
	PK_GPU(pk,preci,qcd##_##pq##_DRHO_RHO))

	#define CAZ2_GPU(pk,preci,qcds)  \
	CAZ3_GPU(pk,preci,qcds,PQ1))  \
	CAZ3_GPU(pk,preci,qcds,PQ2))

	#define DEFALLPROPTEM_K_GPU(pk,preci)  \
	CAZ2_GPU(pk,preci,QCD0) \
	CAZ2_GPU(pk,preci,QCD1) \
	CAZ2_GPU(pk,preci,QCDV) \
	CAZ2_GPU(pk,preci,QCD2) \
	CAZ2_GPU(pk,preci,QCDL) \
	CAZ2_GPU(pk,preci,QCDC)


	#define UVK_GPU(preci,X)  \
	case	V_X: \
	updateVKernel<preci, V_X><<<gridSize,blockSize,0,stream>>> ((const complex<preci> *) m, (complex<preci> *) v, \
									zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (preci) LL, Lx, Sf, Vo, Vf); \
	break;

	#define UCAZ3_GPU(pk,preci,qcd,pq)  \
	UVK_GPU(pk,preci,qcd##_##pq))  \
	UVK_GPU(pk,preci,qcd##_##pq##_RHO))  \
	UVK_GPU(pk,preci,qcd##_##pq##_DRHO))  \
	UVK_GPU(pk,preci,qcd##_##pq##_DALL))  \
	UVK_GPU(pk,preci,qcd##_##pq##_DRHO_RHO))

	#define UCAZ2_GPU(pk,preci,qcds)  \
	UCAZ3_GPU(pk,preci,qcds,PQ1))  \
	UCAZ3_GPU(pk,preci,qcds,PQ2))

	#define DEFALLPROPTEM_U_GPU(pk,preci)  \
	UCAZ2_GPU(pk,preci,QCD0) \
	UCAZ2_GPU(pk,preci,QCD1) \
	UCAZ2_GPU(pk,preci,QCDV) \
	UCAZ2_GPU(pk,preci,QCD2) \
	UCAZ2_GPU(pk,preci,QCDL) \
	UCAZ2_GPU(pk,preci,QCDC)




#endif
