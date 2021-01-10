#include <cmath>
#include <complex>

#include <omp.h>

#include "enum-field.h"
#include "scalar/scalarField.h"

#include "utils/profiler.h"
#include "utils/kgvops.h"
#include "utils/kgvopsXeon.h"
#include "spectrum/LLUTs.h"


/* Define template functions that build the velocity, gradient or potential
in m2 typically for spectra or dumping.
It allows masks */

/* String velocity correction function */

using namespace profiler;

template<typename Float>
void	stringcorre	( Float *da, Float *re, int nr)
{
	// Float sigma = SIGMALUT;
	// Float mr = da[0];
	// Float mi = da[1];
	// Float vr = da[2];
	// Float vi = da[3];
	// Float mrx = da[4];Float mry = da[5];Float mrz = da[6];
	// Float mix = da[7];Float miy = da[8];Float miz = da[9];
	// Gradients already premultiplied by sigma

	/* LUT function */
	// 1 - LUTs is somehow loaded already
	// Lc and Ls
	// LogOut(">>> %f %f %f %f %f %f %f %f %f %f\n",da[0], da[1], da[2], da[3], da[4], da[5], da[6], da[7], da[8], da[9], da[10]);
	Float ma  = std::sqrt(da[4]*da[4]+da[5]*da[5]+da[6]*da[6]);
	Float mb  = std::sqrt(da[7]*da[7]+da[8]*da[8]+da[9]*da[9]);
	bool flip = false;
	if (ma < mb)
		flip=true;
	Float e   = flip ? ma/mb : mb/ma;
	Float cb  = (da[4]*da[7]+da[5]*da[8]+da[6]*da[9])/(ma*mb);
	Float sb  = std::sqrt(1-cb*cb);
	Float A   = flip ? da[1]/mb : da[0]/ma;
	Float B   = flip ? da[0]/mb : da[1]/ma;
	Float s1  = 1;
	Float s2  = 1;
	// LogOut("ma %.2e mb %.2e \n",ma,mb);
	// LogOut("A %.2f B %.2f E %.2f cb %.2f\n",A,B,e,cb);
	if (A < 0){ A = -A; B = -B; s1 = -s1; s2 = -s2;};
	if (B < 0){ B = -B; cb = -cb; s2 = -s2;};
	if ( (A <= 5) && (B <= 5) && ( e <= 1 ) && (cb <= 1) && (cb >= -1))
	{
		/* the LUT has N_A*N_B*N_E*N_C
		in the range (0,5)(0,5)(0,1),(-1,1)
		it misses a factor of e sqrt(1-cb^2)
		but will be compensated at the end */
		// LogOut("A %.2f B %.2f E %.2f cb %.2f\n",A,B,e,cb);
		int iA = A*(N_A-1)/5;
		int iB = B*(N_B-1)/5;
		int iE = e*(N_E-1);
		int iC = (cb+1)*(N_C-1)/2;
		int i0 = iC + N_C*iE + N_CE*iB + N_CEB*iA;
		// LogOut("iA %d iB %d iE %d iC %d \n",iA,iB,iE,iC);
		Float dA  = A*(N_A-1)/5-iA;
		Float dB  = B*(N_B-1)/5-iB;
		Float dE  = e*(N_E-1)-iE;
		Float dC  = (cb+1)*(N_C-1)/2-iC;
		Float dA1  = 1-dA;
		Float dB1  = 1-dB;
		Float dE1  = 1-dE;
		Float dC1  = 1-dC;
		// trilinear interpolation
		// face C 16->8 proyection
		Float L000 = (Float) (LUTc[i0]*dC1 + LUTc[i0+1]*dC);
		Float L001 = (Float) (LUTc[i0+N_C]*dC1 + LUTc[i0+N_C+1]*dC);
		Float L010 = (Float) (LUTc[i0+N_E]*dC1 + LUTc[i0+N_E+1]*dC);
		Float L011 = (Float) (LUTc[i0+N_E+N_C]*dC1 + LUTc[i0+N_E+N_C+1]*dC);
		Float L100 = (Float) (LUTc[i0+N_B]*dC1 + LUTc[i0+N_B+1]*dC);
		Float L101 = (Float) (LUTc[i0+N_B+N_C]*dC1 + LUTc[i0+N_B+N_C+1]*dC);
		Float L110 = (Float) (LUTc[i0+N_B+N_E]*dC1 + LUTc[i0+N_B+N_E+1]*dC);
		Float L111 = (Float) (LUTc[i0+N_B+N_E+N_C]*dC1 + LUTc[i0+N_B+N_E+N_C+1]*dC);
		// face E 8->4
		L000 = L000*dE1 + L001*dE;
		L010 = L010*dE1 + L011*dE;
		L100 = L100*dE1 + L101*dE;
		L110 = L110*dE1 + L111*dE;
		// face B 4->2
		L000 = L000*dB1 + L010*dB;
		L100 = L100*dB1 + L110*dB;
		// face A 2->1
		Float Lc = s1*(L000*dA1 + L100*dA);

		L000 = (Float) (LUTs[i0]*dC1             + LUTs[i0+1]*dC);
		L001 = (Float) (LUTs[i0+N_C]*dC1         + LUTs[i0+N_C+1]*dC);
		L010 = (Float) (LUTs[i0+N_E]*dC1         + LUTs[i0+N_E+1]*dC);
		L011 = (Float) (LUTs[i0+N_E+N_C]*dC1     + LUTs[i0+N_E+N_C+1]*dC);
		L100 = (Float) (LUTs[i0+N_B]*dC1         + LUTs[i0+N_B+1]*dC);
		L101 = (Float) (LUTs[i0+N_B+N_C]*dC1     + LUTs[i0+N_B+N_C+1]*dC);
		L110 = (Float) (LUTs[i0+N_B+N_E]*dC1     + LUTs[i0+N_B+N_E+1]*dC);
		L111 = (Float) (LUTs[i0+N_B+N_E+N_C]*dC1 + LUTs[i0+N_B+N_E+N_C+1]*dC);
		// face E 8->4
		L000 = L000*dE1 + L001*dE;
		L010 = L010*dE1 + L011*dE;
		L100 = L100*dE1 + L101*dE;
		L110 = L110*dE1 + L111*dE;
		// face B 4->2
		L000 = L000*dB1 + L010*dB;
		L100 = L100*dB1 + L110*dB;
		// face A 2->1
		Float Ls = s2*(L000*dA1 + L100*dA);

		// Float L0 = LUTc[i0];
		// Float LpA = LUTc[i0+N_CEB];
		// Float LpB = LUTc[i0+N_CE];
		// Float LpE = LUTc[i0+N_C];
		// Float LpC = LUTc[i0+1];
		// LogOut("dA  %f dB  %f dE  %f dC  %f\n", dA,dB,dE,dC);
		// LogOut("LpA %f LpB %f LpE %f LpC %f\n", LpA, LpB, LpE, LpC);
		// Float Lc = L0 + (LpA-L0)*dA + (LpB-L0)*dB + (LpE-L0)*dE + (LpC-L0)*dC;
		// Lc *= s1;

		// LogOut("L0 %f Lc %f \n",L0,Lc);
		// L0  = LUTs[i0];
		// LpA = LUTs[i0+N_CEB];
		// LpB = LUTs[i0+N_CE];
		// LpE = LUTs[i0+N_C];
		// LpC = LUTs[i0+1];
		// // LogOut("LpA %f LpB %f LpE %f LpC %f\n", LpA, LpB, LpE, LpC);
		// Float Ls = L0 + (LpA-L0)*dA + (LpB-L0)*dB+ (LpE-L0)*dE + (LpC-L0)*dC;
		// Ls *= s2;
		// LogOut("L0 %f Ls %f \n",L0,Ls);

		if (nr == 1)
			re[0] = flip ? ((Float) Ls*da[3]-Lc*da[2])/mb : ((Float) Lc*da[3]-Ls*da[2])/ma;
		else {
			re[1] = flip ? ((Float) Ls*re[1]-Lc*re[0])/mb : ((Float) Lc*re[1]-Ls*re[0])/ma;
			re[0] = flip ? ((Float) Ls*da[3]-Lc*da[2])/mb : ((Float) Lc*da[3]-Ls*da[2])/ma;
		}
	}
	else
	{
		if (nr == 1)
			re[0] = (da[0]*da[3]-da[1]*da[2])/(da[0]*da[0]+da[1]*da[1]);
		else {
			re[1] = (da[0]*re[1]-da[1]*re[0])/(da[0]*da[0]+da[1]*da[1]);
			re[0] = (da[0]*da[3]-da[1]*da[2])/(da[0]*da[0]+da[1]*da[1]);
		}
	}
	// LogOut("r0 %.3f r1 %.3f\n",re[0],re[1]);
}


/* Template expansion */

void	stringcorre	(void *data, void *result, FieldPrecision fPrec, int nr) // parms is a pointer to m,r and grads, result written to result
{
	switch (fPrec)
	{
		case FIELD_SINGLE :
		stringcorre<float> ( (float *) data, (float *) result, nr);
		break;

		case FIELD_DOUBLE :
		stringcorre<double> ( (double *) data, (double *) result, nr);
		break;

		default :
		LogError("[stringcorre] precision not reconised.");
		break;
	}
}


/* Builds conformal Axion velocity w/wo LUT correction */


template<typename Float, SpectrumMaskType mask, bool LUTcorr, bool padded>
size_t buildc_k(Scalar *field, PadIndex pi, Float zaskaFF)
{
	/* Builds the axion velocity from a complex scalar
	with the integral softening method.
	Options
	PadIndex = PFIELD_M2 ? copy to m2
	PadIndex = PFIELD_M2S ? copy to m2Start
	PadIndex = PFIELD_M2H ? copy to m2half
	PadIndex = PFIELD_M2HS ? copy to m2half

	Actually we first compute in m2 and then move
	*/


	/* vectorised version for AVX, does not buy much */
	// if (LUTcorr){
	// 	buildc_k_KernelXeon(field->mCpu(),field->vCpu(),field->m2Cpu(),field);
	// 	return 0;
	// }

	size_t Lz = field->Depth();
	size_t Ly = field->Length();
	size_t LyLy = Ly*Ly;
	FieldPrecision fPrec = field->Precision();
	Float Rscale = *field->RV();

	std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mCpu())+(field->getNg()-1)*field->Surf();
	std::complex<Float> *va     = static_cast<std::complex<Float>*>(field->vCpu());
	char *strdaa                = static_cast<char *>(static_cast<void *>(field->sData()));
	Float *m2sa                 = static_cast<Float *>(field->m2Cpu());
	Float *m2sax                = static_cast<Float *>(field->m2half());

	// Float sigma = SIGMALUT;
	Float pre = 0.5*SIGMALUT;
	size_t luts = 0;
	LogMsg(VERB_HIGH,"[bck] Build conformal Axion velocity %s", LUTcorr? "with LUT":"");LogFlush();
	#pragma omp parallel for schedule(static) collapse(2) reduction(+:luts)
	for (size_t iz=0; iz < Lz; iz++) {
		for (size_t iy=0; iy < Ly; iy++) {
			size_t zo = Ly*(Ly+2)*(iz) ;
			size_t zi = LyLy*(iz+1) ;
			size_t zp = LyLy*(iz+2) ;
			size_t zm = LyLy*(iz) ;
			size_t yo = (Ly+2)*iy ;
			size_t yi = Ly*iy ;
			size_t yp = Ly*((iy+1)%Ly) ;
			size_t ym = Ly*((Ly+iy-1)%Ly) ;
			for (size_t ix=0; ix < Ly; ix++) {
				size_t odx = ix + yo + zo; size_t idx = ix + yi + zi; size_t oodx;
				Float vA;
				Float da[10];
				bool  skipLUT = false;
				da[0] = ma[idx].real()-zaskaFF;
				da[1] = ma[idx].imag();
				da[2] = va[idx-LyLy].real();
				da[3] = va[idx-LyLy].imag();
				if (LUTcorr) {
					da[4] = pre*(ma[((ix + 1) % Ly) + yi + zi].real()-ma[((Ly + ix - 1) % Ly) + yi + zi].real());
					if (abs(da[0]) > abs(5*da[4]))
						{ skipLUT = true; goto jmp;}
					da[7] = pre*(ma[((ix + 1) % Ly) + yi + zi].imag()-ma[((Ly + ix - 1) % Ly) + yi + zi].imag());
					if (abs(da[1]) > abs(5*da[7]))
						{ skipLUT = true; goto jmp;}
					da[5] = pre*(ma[ix + yp + zi].real()-ma[ix + ym + zi].real());
					if (abs(da[0]) > abs(5*da[5]))
						{ skipLUT = true; goto jmp;}
					da[8] = pre*(ma[ix + yp + zi].imag()-ma[ix + ym + zi].imag());
					if (abs(da[1]) > abs(5*da[8]))
						{ skipLUT = true; goto jmp;}
					da[6] = pre*(ma[ix + yi + zp].real()-ma[ix + yi + zm].real());
					if (abs(da[0]) > abs(5*da[6]))
						{ skipLUT = true; goto jmp;}
					da[9] = pre*(ma[ix + yi + zp].imag()-ma[ix + yi + zm].imag());
					if (abs(da[1]) > abs(5*da[9]))
						{ skipLUT = true; goto jmp;}
					stringcorre(static_cast<void*>(da),static_cast<void*>(&vA), fPrec, 1);
					luts++;
				}
				jmp:
				if (!LUTcorr || skipLUT)
					vA = (da[0]*da[3]-da[1]*da[2])/(da[0]*da[0]+da[1]*da[1]);

				// if (idx == 256*256){
				// LogOut("x %d %d %d y %d %d z %d %d \n",ix,((ix + 1) % Ly),((Ly + ix - 1) % Ly),yp,ym,zp,zm);
				// LogOut("%f %f %f %f %f %f %f %f %f %f\n ", da[0],da[1],da[2],da[3],da[4],da[5],da[6],da[7],da[8],da[9]);
				// LogOut("%f %f \n ", re[0],re[1]);
				// }
				oodx = padded ? odx : idx-LyLy;
				switch(mask){
					case SPMASK_FLAT:
								m2sa[oodx] = Rscale*(vA);
							break;
					case SPMASK_REDO:
					case SPMASK_BALL:
								if (strdaa[idx-LyLy] & STRING_MASK)
										m2sa[oodx] = 0 ;
								else
										m2sa[oodx] = Rscale*(vA);
							break;
					case SPMASK_GAUS:
					case SPMASK_DIFF:
								m2sa[oodx] = m2sax[idx-LyLy]*Rscale*(vA);
							break;
					case SPMASK_VIL:
								m2sa[oodx] = std::sqrt(da[0]*da[0]+da[1]*da[1])*(vA);
							break;
					case SPMASK_VIL2:
								m2sa[oodx] = (da[0]*da[0]+da[1]*da[1])/Rscale*(vA);
							break;
					case SPMASK_SAXI:
								m2sa[oodx]   =  da[2] ;
								m2sax[oodx]  =  da[3] ;
					break;

				} //end mask
			}
		}
	}
	return luts;
}

void	buildc_k_map	(Scalar *field, bool LUTcorr)
{
	double zasi = field->Saskia()* *(field->RV());
	if (LUTcorr)
		size_t luts = buildc_k<float,SPMASK_FLAT,true, false> (field, PFIELD_M2, (float) zasi);
	else
		size_t luts = buildc_k<float,SPMASK_FLAT,false, false> (field, PFIELD_M2, (float) zasi);
}

/* Float Template expansion */

template <SpectrumMaskType mask, bool LUTcorr>
void	buildc_k	(Scalar *field, PadIndex pi, double zaskaFF)
{
	Profiler &prof = getProfiler(PROF_SPEC);
	prof.start();
	size_t luts;
	if (LUTcorr)
		switch (field->Precision())
		{
			default :
			case FIELD_SINGLE :
				luts = buildc_k<float,mask,true, true> (field, pi, (float) zaskaFF);
				break;
			case FIELD_DOUBLE :
				luts = buildc_k<double,mask,true, true> (field, pi, (double) zaskaFF);
				break;
		}
	else
	switch (field->Precision())
	{
		default:
		case FIELD_SINGLE :
			luts = buildc_k<float,mask,false, true> (field, pi, (float) zaskaFF);
			break;
		case FIELD_DOUBLE :
			luts = buildc_k<double,mask,false, true> (field, pi, (double) zaskaFF);
			break;
	}

	char LABEL[256];
	sprintf(LABEL, "BuildK %d %s", mask, LUTcorr? "LUT" : "-");
	prof.stop();
	/* Flop/Bytes counter
	-
	if no lut
		FLOP *-* / *-* = 7
		BYTE mr,mi,vr,vi->m2 4 reads 1 write
	if lut
		FLOP sqrt(*+*+*)x2,/,*+*+* /*,sqrt-*,/x4,****,* /-*-/-*-/-+*-/-----,
		FLOP *+*x(8+4+2+1)x2, *-* / (possibly 2)
		= 6x2 + 1 + 7 + 3 + 4 + 4+ 20 + 3x15x2 + 3 =
		BYTE (mr,mi)x(7),vr,vi->m2 16 reads 1 write
	if FLAT, REDO *
	if GAUS,DIFF  **
	if VIL  sqrt*+*** VIL2 I *+* / *
	*/
	LogMsg(VERB_PARANOID,"[bck] BuildK Reporting %lu LUTs",luts);
	double flops  = 1e-9 * (144*luts + 7*(field->Size()-luts));
	double mbytes = 1e-9 * (17*luts + 5*(field->Size()-luts));
	prof.add(std::string(LABEL), flops, mbytes);
}

void	buildc_k	(Scalar *field, PadIndex pi, double zaskaFF, SpectrumMaskType mask, bool LUTcorr)
{
	if (LUTcorr)
	switch (mask){
		default:
		case SPMASK_FLAT:
			buildc_k<SPMASK_FLAT,true> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_REDO:
		case SPMASK_BALL:
			buildc_k<SPMASK_REDO,true> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_GAUS:
		case SPMASK_DIFF:
			buildc_k<SPMASK_DIFF,true> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_VIL:
			buildc_k<SPMASK_VIL,true> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_VIL2:
			buildc_k<SPMASK_VIL2,true> (field, PFIELD_M2, zaskaFF);
			break;
	}
	else {
		switch (mask){
			default:
			case SPMASK_FLAT:
				buildc_k<SPMASK_FLAT,false> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_REDO:
			case SPMASK_BALL:
				buildc_k<SPMASK_REDO,false> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_GAUS:
			case SPMASK_DIFF:
				buildc_k<SPMASK_DIFF,false> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_VIL:
				buildc_k<SPMASK_VIL,false> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_VIL2:
				buildc_k<SPMASK_VIL2,false> (field, PFIELD_M2, zaskaFF);
				break;
		}
	}

}



/* Builds conformal Axion gradient X w/wo LUT correction,
is almost a copy of the K function
it only substitutes
v_real and v_imag for d_x phi_real and d_y phi_imag */


template<typename Float, SpectrumMaskType mask, bool LUTcorr>
size_t buildc_gx(Scalar *field, PadIndex pi, Float zaskaFF)
{

	size_t Lz            = field->Depth();
	size_t Ly            = field->Length();
	size_t LyLy          = Ly*Ly;
	FieldPrecision fPrec = field->Precision();
	Float Rscale         = *field->RV();
	Float Rdepta2        = Rscale/field->Delta()/2; // prefactor
	Float idepta2        = 1/field->Delta()/2; // prefactor

	std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mCpu())+(field->getNg()-1)*field->Surf();
	std::complex<Float> *va     = static_cast<std::complex<Float>*>(field->vCpu());
	char *strdaa                = static_cast<char *>(static_cast<void *>(field->sData()));
	Float *m2sa                 = static_cast<Float *>(field->m2Cpu());
	Float *m2sax                = static_cast<Float *>(field->m2half());

	// Float sigma = 0.4;
	Float pre = 0.5*SIGMALUT;
	size_t luts = 0;
	LogMsg(VERB_HIGH,"[bcgx] Build conformal Axion gradient X %s", LUTcorr? "with LUT":"");
	#pragma omp parallel for schedule(static) collapse(2) reduction(+:luts)
	for (size_t iz=0; iz < Lz; iz++) {
		for (size_t iy=0; iy < Ly; iy++) {
			size_t zo = Ly*(Ly+2)*(iz) ;
			size_t zi = LyLy*(iz+1) ;
			size_t zp = LyLy*(iz+2) ;
			size_t zm = LyLy*(iz) ;
			size_t yo = (Ly+2)*iy ;
			size_t yi = Ly*iy ;
			size_t yp = Ly*((iy+1)%Ly) ;
			size_t ym = Ly*((Ly+iy-1)%Ly) ;
			for (size_t ix=0; ix < Ly; ix++) {
				size_t odx = ix + yo + zo; size_t idx = ix + yi + zi;
				Float gAx;
				Float da[10];
				bool  skipLUT = false;
				da[0] = ma[idx].real()-zaskaFF;
				da[1] = ma[idx].imag();
				da[2] = ma[((ix + 1) % Ly) + yi + zi].real()-ma[((Ly + ix - 1) % Ly) + yi + zi].real(); // gradient*2delta
				da[3] = ma[((ix + 1) % Ly) + yi + zi].imag()-ma[((Ly + ix - 1) % Ly) + yi + zi].imag(); // gradient*2delta
				if (LUTcorr) {
					da[4] = pre*da[2];
					if (abs(da[0]) > abs(5*da[4]))
						{ skipLUT = true; goto jmp;}
					da[7] = pre*da[3];
					if (abs(da[1]) > abs(5*da[7]))
						{ skipLUT = true; goto jmp;}
					da[5] = pre*(ma[ix + yp + zi].real()-ma[ix + ym + zi].real());
					if (abs(da[0]) > abs(5*da[5]))
						{ skipLUT = true; goto jmp;}
					da[8] = pre*(ma[ix + yp + zi].imag()-ma[ix + ym + zi].imag());
					if (abs(da[1]) > abs(5*da[8]))
						{ skipLUT = true; goto jmp;}
					da[6] = pre*(ma[ix + yi + zp].real()-ma[ix + yi + zm].real());
					if (abs(da[0]) > abs(5*da[6]))
						{ skipLUT = true; goto jmp;}
					da[9] = pre*(ma[ix + yi + zp].imag()-ma[ix + yi + zm].imag());
					if (abs(da[1]) > abs(5*da[9]))
						{ skipLUT = true; goto jmp;}
					stringcorre(static_cast<void*>(da),static_cast<void*>(&gAx), fPrec, 1);
					luts++;
				}
				jmp:
				if (!LUTcorr || skipLUT)
					gAx = (da[0]*da[3]-da[1]*da[2])/(da[0]*da[0]+da[1]*da[1]);

				// if (idx == 256*256){
				// LogOut("x %d %d %d y %d %d z %d %d \n",ix,((ix + 1) % Ly),((Ly + ix - 1) % Ly),yp,ym,zp,zm);
				// LogOut("%f %f %f %f %f %f %f %f %f %f\n ", da[0],da[1],da[2],da[3],da[4],da[5],da[6],da[7],da[8],da[9]);
				// LogOut("%f %f \n ", re[0],re[1]);
				// }
				switch(mask){
					case SPMASK_FLAT:
								m2sa[odx] = Rdepta2*(gAx);
							break;
					case SPMASK_REDO:
					case SPMASK_BALL:
								if (strdaa[idx-LyLy] & STRING_MASK)
										m2sa[odx] = 0 ;
								else
										m2sa[odx] = Rdepta2*(gAx);
							break;
					case SPMASK_GAUS:
					case SPMASK_DIFF:
								m2sa[odx] = m2sax[idx-LyLy]*Rdepta2*(gAx);
							break;
					case SPMASK_VIL:
								m2sa[odx] = idepta2*std::sqrt(da[0]*da[0]+da[1]*da[1])*(gAx);
							break;
					case SPMASK_VIL2:
								m2sa[odx] = idepta2*(da[0]*da[0]+da[1]*da[1])/Rscale*(gAx);
							break;
					case SPMASK_SAXI:
								m2sa[odx]   =  idepta2*da[2] ;
								m2sax[odx]  =  idepta2*da[3] ;
					break;

				} //end mask
			}
		}
	}
	return luts;
}

/* Float Template expansion */

template <SpectrumMaskType mask, bool LUTcorr>
void	buildc_gx	(Scalar *field, PadIndex pi, double zaskaFF)
{
	Profiler &prof = getProfiler(PROF_SPEC);
	prof.start();
	size_t luts;
	if (LUTcorr)
		switch (field->Precision())
		{
			default :
			case FIELD_SINGLE :
				luts = buildc_gx<float,mask,true> (field, pi, (float) zaskaFF);
				break;
			case FIELD_DOUBLE :
				luts = buildc_gx<double,mask,true> (field, pi, (double) zaskaFF);
				break;
		}
	else
	switch (field->Precision())
	{
		default:
		case FIELD_SINGLE :
			luts = buildc_gx<float,mask,false> (field, pi, (float) zaskaFF);
			break;
		case FIELD_DOUBLE :
			luts = buildc_gx<double,mask,false> (field, pi, (double) zaskaFF);
			break;
	}

	char LABEL[256];
	sprintf(LABEL, "BuildGX %d %s", mask, LUTcorr? "LUT" : "-");
	prof.stop();
	/* Flop/Bytes counter
	-
	if no lut
		FLOP -*-*- / *-* = 9
		BYTE mr,mi+(mr,mi)x2 neighbours ->m2 6 reads 1 write
	if lut
		aprox the same as before
		BYTE (mr,mi)x(7),vr,vi->m2 16 reads 1 write
	if FLAT, REDO *
	if GAUS,DIFF  **
	if VIL  sqrt*+*** VIL2 I *+* / *
	*/

	LogMsg(VERB_PARANOID,"[bck] BuildGX Reporting %lu LUTs",luts);
	double flops  = 1e-9 * (144*luts + 9*(field->Size()-luts));
	double mbytes = 1e-9 * (15*luts + 7*(field->Size()-luts));
	prof.add(std::string(LABEL), flops, mbytes);
}


void	buildc_gx	(Scalar *field, PadIndex pi, double zaskaFF, SpectrumMaskType mask, bool LUTcorr)
{
if (LUTcorr)
	switch (mask){
		default:
		case SPMASK_FLAT:
			buildc_gx<SPMASK_FLAT,true> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_REDO:
		case SPMASK_BALL:
			buildc_gx<SPMASK_REDO,true> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_GAUS:
		case SPMASK_DIFF:
			buildc_gx<SPMASK_DIFF,true> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_VIL:
			buildc_gx<SPMASK_VIL,true> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_VIL2:
			buildc_gx<SPMASK_VIL2,true> (field, PFIELD_M2, zaskaFF);
			break;
		}
else
	switch (mask){
		default:
		case SPMASK_FLAT:
			buildc_gx<SPMASK_FLAT,false> (field, PFIELD_M2, zaskaFF);
		break;
		case SPMASK_REDO:
		case SPMASK_BALL:
			buildc_gx<SPMASK_REDO,false> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_GAUS:
		case SPMASK_DIFF:
			buildc_gx<SPMASK_DIFF,false> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_VIL:
			buildc_gx<SPMASK_VIL,false> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_VIL2:
			buildc_gx<SPMASK_VIL2,false> (field, PFIELD_M2, zaskaFF);
			break;
	}
}


/* Builds conformal Axion gradient X w/wo LUT correction,
is almost a copy of the K function
it only substitutes
v_real and v_imag for d_x phi_real and d_y phi_imag */


template<typename Float, SpectrumMaskType mask, bool LUTcorr>
size_t buildc_gyz(Scalar *field, PadIndex pi, Float zaskaFF)
{

	size_t Lz            = field->Depth();
	size_t Ly            = field->Length();
	size_t LyLy          = Ly*Ly;
	FieldPrecision fPrec = field->Precision();
	Float Rscale         = *field->RV();
	Float Rdepta2        = Rscale/field->Delta()/2; // prefactor
	Float idepta2        = 1/field->Delta()/2; // prefactor

	std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mCpu())+(field->getNg()-1)*field->Surf();
	std::complex<Float> *va     = static_cast<std::complex<Float>*>(field->vCpu());
	char *strdaa                = static_cast<char *>(static_cast<void *>(field->sData()));
	Float *m2sa                 = static_cast<Float *>(field->m2Cpu());
	Float *m2sax                = static_cast<Float *>(field->m2half());

	// Float sigma = 0.4;
	Float pre = 0.5*SIGMALUT;
	size_t luts = 0;
	LogMsg(VERB_HIGH,"[bcgyz] Build conformal Axion gradients YZ %s", LUTcorr? "with LUT":"");
	#pragma omp parallel for schedule(static) collapse(2) reduction(+:luts)
	for (size_t iz=0; iz < Lz; iz++) {
		for (size_t iy=0; iy < Ly; iy++) {
			size_t zo = Ly*(Ly+2)*(iz) ;
			size_t zi = LyLy*(iz+1) ;
			size_t zp = LyLy*(iz+2) ;
			size_t zm = LyLy*(iz) ;
			size_t yo = (Ly+2)*iy ;
			size_t yi = Ly*iy ;
			size_t yp = Ly*((iy+1)%Ly) ;
			size_t ym = Ly*((Ly+iy-1)%Ly) ;
			for (size_t ix=0; ix < Ly; ix++) {
				size_t odx = ix + yo + zo; size_t idx = ix + yi + zi;
				Float gAyz[2];
				Float da[10];
				bool  skipLUT = false;
				da[0] = ma[idx].real()-zaskaFF;
				da[1] = ma[idx].imag();
				da[2] = ma[ix + yp + zi].real()-ma[ix + ym + zi].real(); // gradienty*2delta
				da[3] = ma[ix + yp + zi].imag()-ma[ix + ym + zi].imag(); // gradienty*2delta
				gAyz[0] = ma[ix + yi + zp].real()-ma[ix + yi + zm].real();
				gAyz[1] = ma[ix + yi + zp].imag()-ma[ix + yi + zm].imag();
				if (LUTcorr) {
					da[4] = pre*(ma[((ix + 1) % Ly) + yi + zi].real()-ma[((Ly + ix - 1) % Ly) + yi + zi].real());
					if (abs(da[0]) > abs(5*da[4]))
						{ skipLUT = true; goto jmp;}
					da[7] = pre*(ma[((ix + 1) % Ly) + yi + zi].imag()-ma[((Ly + ix - 1) % Ly) + yi + zi].imag());
					if (abs(da[1]) > abs(5*da[7]))
						{ skipLUT = true; goto jmp;}
					da[5] = pre*da[2];
					if (abs(da[0]) > abs(5*da[5]))
						{ skipLUT = true; goto jmp;}
					da[8] = pre*da[3];
					if (abs(da[1]) > abs(5*da[8]))
						{ skipLUT = true; goto jmp;}
					da[6] = pre*gAyz[0];
					if (abs(da[0]) > abs(5*da[6]))
						{ skipLUT = true; goto jmp;}
					da[9] = pre*gAyz[1];
					if (abs(da[1]) > abs(5*da[9]))
						{ skipLUT = true; goto jmp;}
					stringcorre(static_cast<void*>(da),static_cast<void*>(gAyz), fPrec, 2);
					luts++;
				}
				jmp:
				if (!LUTcorr || skipLUT) {
					gAyz[1] = (da[0]*gAyz[1]-da[1]*gAyz[0])/(da[0]*da[0]+da[1]*da[1]);
					gAyz[0] = (da[0]*da[3]-da[1]*da[2])/(da[0]*da[0]+da[1]*da[1]);
				}

				switch(mask){
					case SPMASK_FLAT:
								m2sa[odx]  = Rdepta2*gAyz[0];
								m2sax[odx] = Rdepta2*gAyz[1];
							break;
					case SPMASK_REDO:
					case SPMASK_BALL:
								if (strdaa[idx-LyLy] & STRING_MASK) {
										m2sa[odx]  = 0 ;
										m2sax[odx] = 0 ;}
								else {
										m2sa[odx]  = Rdepta2*gAyz[0];
										m2sax[odx] = Rdepta2*gAyz[1]; }
							break;
					case SPMASK_GAUS:
					case SPMASK_DIFF:
								m2sa[odx]       = m2sax[idx-LyLy]*Rdepta2*gAyz[0];
								m2sax[idx-LyLy] = m2sax[idx-LyLy]*Rdepta2*gAyz[1];
							break;
					case SPMASK_VIL:
								m2sa[odx]  = idepta2*std::sqrt(da[0]*da[0]+da[1]*da[1])*gAyz[0];
								m2sax[odx] = idepta2*std::sqrt(da[0]*da[0]+da[1]*da[1])*gAyz[1];
							break;
					case SPMASK_VIL2:
								m2sa[odx]  = idepta2*(da[0]*da[0]+da[1]*da[1])/Rscale*gAyz[0];
								m2sax[odx] = idepta2*(da[0]*da[0]+da[1]*da[1])/Rscale*gAyz[1];
							break;
					case SPMASK_SAXI:
								m2sa[odx]   =  idepta2*da[2] ;
								m2sax[odx]  =  idepta2*da[3] ;
					break;

				} //end mask
			}
		}
	}
	return luts;
}

/* Float Template expansion */

template <SpectrumMaskType mask, bool LUTcorr>
void	buildc_gyz	(Scalar *field, PadIndex pi, double zaskaFF)
{
	Profiler &prof = getProfiler(PROF_SPEC);
	prof.start();
	size_t luts;
	if (LUTcorr)
		switch (field->Precision())
		{
			default :
			case FIELD_SINGLE :
				luts = buildc_gyz<float,mask,true> (field, pi, (float) zaskaFF);
				break;
			case FIELD_DOUBLE :
				luts = buildc_gyz<double,mask,true> (field, pi, (double) zaskaFF);
				break;
		}
		else
		switch (field->Precision())
		{
			default:
			case FIELD_SINGLE :
				luts = buildc_gyz<float,mask,false> (field, pi, (float) zaskaFF);
				break;
			case FIELD_DOUBLE :
				luts = buildc_gyz<double,mask,false> (field, pi, (double) zaskaFF);
				break;
		}

	char LABEL[256];
	sprintf(LABEL, "BuildGYZ %d %s", mask, LUTcorr? "LUT" : "-");
	prof.stop();
	/* Flop/Bytes counter
	-
	if no lut
		FLOP (-*-*- / *-*) + *  = 10 x 2 = 18
		BYTE mr,mi+(mr,mi)x4 neighbours ->m2 10 reads 2 write
	if lut
		aprox the same as before
		BYTE (mr,mi)x(7)->m2 14 reads 2 write
	if FLAT, REDO *
	if GAUS,DIFF  **
	if VIL  sqrt*+*** VIL2 I *+* / *
	*/
	LogMsg(VERB_PARANOID,"[bck] BuildGYZ Reporting %lu LUTs",luts);
	double flops  = 1e-9 * (144*luts + 20*(field->Size()-luts));
	double mbytes = 1e-9 * (16*luts + 12*(field->Size()-luts));
	prof.add(std::string(LABEL), flops, mbytes);
}


void	buildc_gyz	(Scalar *field, PadIndex pi, double zaskaFF, SpectrumMaskType mask, bool LUTcorr)
{
	if (LUTcorr)
		switch (mask){
			default:
			case SPMASK_FLAT:
				buildc_gyz<SPMASK_FLAT,true> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_REDO:
			case SPMASK_BALL:
				buildc_gyz<SPMASK_REDO,true> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_GAUS:
			case SPMASK_DIFF:
				buildc_gyz<SPMASK_DIFF,true> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_VIL:
				buildc_gyz<SPMASK_VIL,true> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_VIL2:
				buildc_gyz<SPMASK_VIL2,true> (field, PFIELD_M2, zaskaFF);
				break;
		}
	else
		switch (mask){
			default:
			case SPMASK_FLAT:
				buildc_gyz<SPMASK_FLAT,false> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_REDO:
			case SPMASK_BALL:
				buildc_gyz<SPMASK_REDO,false> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_GAUS:
			case SPMASK_DIFF:
				buildc_gyz<SPMASK_DIFF,false> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_VIL:
				buildc_gyz<SPMASK_VIL,false> (field, PFIELD_M2, zaskaFF);
				break;
			case SPMASK_VIL2:
				buildc_gyz<SPMASK_VIL2,false> (field, PFIELD_M2, zaskaFF);
				break;
		}
}


/* Builds conformal Axion potential equivalent */


template<typename Float, SpectrumMaskType mask, bool LUTcorr>
void buildc_v(Scalar *field, PadIndex pi, Float zaskaFF)
{
	/* If cosine potential the energy is
	R^-4 [ (psi')^2/2 + (grad phi)^2/2 + m2 R^4 (1-cos(psi/R)) ]
	in the linear regime the potential term is simply
	m2 R^2 psi^2/2
	the non0-linear term can be written as
	m2 R^4 (2 sin^2(psi/2R))
	which suggests to generalise
	m2 R^2/2 psi^2 -> m2 R^2/2 (4R^2 sin^2(psi/2R))
	and compute the FT not of psi, but of 2R sin(psi/2R)

	psi -> 2R sin(psi/2R) = 2R sin(theta/2)

	If we  use the binning method SPECTRUM_VV which requires to multiply
	by the mass-prefactor as well (conformal mass)

	(m_A * R) psi -> (m_A * R) 2R sin(theta/2)

	But we will use V and VNL which do it automatically

	We are served with m_real and m_imag = mod (cos(theta),sin(theta))
	so we can use the equality

	sin(theta/2) = +sqrt(0.5(1-m_real re/|m|))
	2 R sin(theta/2) = R sqrt(2(1-m_real re/|m|))

	we chose the positive sign, and no alternations because it must be
	periodic.

	Indeed the generalisation is
	theta^2/2 -> 1-Cos[theta]
	theta -> sqrt(2(1-Cos[theta]))
	*/

	size_t Lz = field->Depth();
	size_t Ly = field->Length();
	size_t LyLy = Ly*Ly;
	FieldPrecision fPrec = field->Precision();
	Float Rscale = *field->RV();

	std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mStart());
	std::complex<Float> *va     = static_cast<std::complex<Float>*>(field->vCpu());
	char *strdaa                = static_cast<char *>(static_cast<void *>(field->sData()));
	Float *m2sa                 = static_cast<Float *>(field->m2Cpu());
	Float *m2sax                = static_cast<Float *>(field->m2half());

	// Float sigma = 0.4;
	Float pre = 0.5*SIGMALUT;
	LogMsg(VERB_HIGH,"[nRun] Build conformal Axion velocity") ;
	#pragma omp parallel for schedule(static)
	for (size_t iz=0; iz < Lz; iz++) {
		size_t zo = Ly*(Ly+2)*(iz) ;
		size_t zi = LyLy*iz ;
		for (size_t iy=0; iy < Ly; iy++) {
			size_t yo = (Ly+2)*iy ;
			size_t yi = Ly*iy ;
			for (size_t ix=0; ix < Ly; ix++) {
				size_t odx = ix + yo + zo; size_t idx = ix + yi + zi;
				Float da[2] = {ma[idx].real()-zaskaFF, ma[idx].imag()};
				Float m2    = da[0]*da[0]+da[1]*da[1];
				switch(mask){
					case SPMASK_FLAT:
								// m2sa[odx]  = Rscale*std::sqrt(2*(1-da[0]/std::sqrt(da[0]*da[0]+da[1]*da[1])));
								m2sa[odx]  = 2.*Rscale * std::sin(0.5*std::atan2(da[1],da[0]));
								m2sax[odx] = Rscale   * std::atan2(da[0],da[1]);
							break;
					case SPMASK_REDO:
					case SPMASK_BALL:
								if (strdaa[idx] & STRING_MASK) {
										m2sa[odx] = 0 ;
										m2sax[odx]= 0 ; }
								else {
										m2sa[odx] = 2.*Rscale * std::sin(0.5*std::atan2(da[1],da[0]));
										m2sax[odx] = Rscale*std::atan2(da[0],da[1]); }
							break;
					case SPMASK_GAUS:
					case SPMASK_DIFF:
								m2sa[odx] = m2sax[idx]*2.*Rscale * std::sin(0.5*std::atan2(da[1],da[0]));
								/* Little problem here we do not want to break the mask */
							break;
					case SPMASK_VIL: {
								Float mskk = std::sqrt(da[0]*da[0]+da[1]*da[1])/Rscale;
								m2sa[odx]  = mskk*2.*Rscale * std::sin(0.5*std::atan2(da[1],da[0]));
								m2sax[odx] = mskk*Rscale*std::atan2(da[0],da[1]); }
							break;
					case SPMASK_VIL2: {
								Float mskk = (da[0]*da[0]+da[1]*da[1])/Rscale/Rscale;
								m2sa[odx]  = mskk*2.*Rscale * std::sin(0.5*std::atan2(da[1],da[0]));
								m2sax[odx] = mskk*Rscale*std::atan2(da[0],da[1]); }
							break;
				} //end mask
			}
		}
	}
}

/* Float Template expansion */

template <SpectrumMaskType mask, bool LUTcorr>
void	buildc_v	(Scalar *field, PadIndex pi, double zaskaFF)
{
	Profiler &prof = getProfiler(PROF_SPEC);
	prof.start();

	if (LUTcorr)
		switch (field->Precision())
		{
			default :
			case FIELD_SINGLE :
				buildc_v<float,mask,true> (field, pi, (float) zaskaFF);
				break;
			case FIELD_DOUBLE :
				buildc_v<double,mask,true> (field, pi, (double) zaskaFF);
				break;
		}
	else
	switch (field->Precision())
	{
		default:
		case FIELD_SINGLE :
			buildc_v<float,mask,false> (field, pi, (float) zaskaFF);
			break;
		case FIELD_DOUBLE :
			buildc_v<double,mask,false> (field, pi, (double) zaskaFF);
			break;
	}

	char LABEL[256];
	sprintf(LABEL, "BuildV %d %s", mask, LUTcorr? "LUT" : "-");
	prof.stop();
	prof.add(std::string(LABEL), 0.0, 0.0);
}

void	buildc_v	(Scalar *field, PadIndex pi, double zaskaFF, SpectrumMaskType mask)
{
	switch (mask){
		default:
		case SPMASK_FLAT:
			buildc_v<SPMASK_FLAT,false> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_REDO:
		case SPMASK_BALL:
			buildc_v<SPMASK_REDO,false> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_GAUS:
		case SPMASK_DIFF:
			buildc_v<SPMASK_DIFF,false> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_VIL:
			buildc_v<SPMASK_VIL,false> (field, PFIELD_M2, zaskaFF);
			break;
		case SPMASK_VIL2:
			buildc_v<SPMASK_VIL2,false> (field, PFIELD_M2, zaskaFF);
			break;
	}
}
