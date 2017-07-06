#ifndef _RKPARMS_
	#define _RKPARMS_

	/*	Runge-Kutta-Nystrom 4th order	*/

	#define C1  0.1344961992774310892
	#define D1  0.5153528374311229364
	#define C2  -0.2248198030794208058
	#define D2  -0.085782019412973646
	#define C3  0.7563200005156682911
	#define D3  0.4415830236164665242
	#define C4  0.3340036032863214255
	#define D4  0.1288461583653841854

	/*		Omelyan			*/
/*
	constexpr double xi  = +0.16449865155757600;
	constexpr double lb  = -0.02094333910398989;
	constexpr double chi = +1.23569265113891700;

	constexpr double oC1 = xi; 
	constexpr double oD1 = 0.5*(1.-2.*lb);
	constexpr double oC2 = chi;
	constexpr double oD2 = lb;
	constexpr double oC3 = 1.-2.*(xi+chi);
	constexpr double oD3 = lb;
	constexpr double oC4 = chi;
	constexpr double oD4 = 0.5*(1.-2.*lb);
	constexpr double oC5 = xi;
*/
	constexpr double chi = +0.19318332750378360;
	constexpr double oC3 = chi;
	constexpr double oD3 = 0.5;
	constexpr double oC4 = 1.-2.*chi;
	constexpr double oD4 = 0.5;
	constexpr double oC5 = chi;

#endif
