#include<cstdlib>
#include<cstring>

#include "utils/utils.h"

#include <cosmos/cosmos.h>
#include <WKB/spline.h>


using namespace std;

void Cosmos::Setup()
{

  /*Only if ueCosmo */

  if (!ueCosm)
    return ;
  LogMsg(VERB_NORMAL,"[Cos] Cosmos Setup");
  LogFlush();

  /*Read Cosmology*/

  char cosName[2048];
  if (const char *cosPath = std::getenv("JAXIONS_DIR")) {
    if (strlen(cosPath) < 1022) {
      struct stat tStat;
      if (stat(cosPath, &tStat) == 0 && S_ISDIR(tStat.st_mode)) {
        strcpy(cosName, cosPath);
      } else {
        printf("Path %s doesn't exist, using default\n", cosPath);
      }
    }
  }
  sprintf(cosName, "%s%s", cosName,  "/include/cosmos/jaxi-cosmo.txt");
  std::vector<double>	etav, Rv, Tv, Rppv, chiv, pfv;
  FILE *cFile = nullptr;
  if (((cFile  = fopen(cosName, "r")) == nullptr)){
    LogMsg(VERB_NORMAL,"[Cos] No %s !",cosName);
    return ;
  }
  else
  {
    double eta, R, T, Rpp, chi ,pf;

    LogMsg(VERB_NORMAL,"[Cos] Reading cosmology files from list");
    LogFlush();
    char buffer[200];
    fgets(buffer, 200, cFile); // reads header
    LogMsg(VERB_NORMAL,"[Cos] %s",buffer);
    fgets(buffer, 200, cFile); //reads 2nd line space

    int line = 1;
    fscanf (cFile ,"%lf %lf %lf %lf %lf %lf", &eta, &R, &T, &Rpp, &chi , &pf);
    while(!feof(cFile)){
      etav.push_back(eta);
      Rv.push_back(R);
      Tv.push_back(T);
      Rppv.push_back(Rpp);
      chiv.push_back(chi);
      pfv.push_back(pf);
      // LogMsg(VERB_NORMAL,"[VAX] i_meas=%d read z=%f meas=%d", i_meas, meas_zlist[i_meas], meas_typelist[i_meas]);
      fscanf (cFile ,"%lf %lf %lf %lf %lf %lf", &eta, &R, &T, &Rpp, &chi , &pf);
      if (feof(cFile)){
        LogMsg (VERB_PARANOID ,"I break");
        break;
      }

      LogMsg (VERB_PARANOID ,"%d %lf %lf %lf %lf %lf %lf", line, eta, R, T, Rpp, chi , pf);
      line ++;
    }
    LogMsg (VERB_PARANOID ,"eta %lf eta %lf ", etav[etav.size()-1], etav[etav.size()-2]);
    // for (int i =etav.size()-1; i>0;i--)
    //   if (etav[i] == etav[i-1]){
    //     LogMsg (VERB_PARANOID ,"%d eta %lf eta-1 %lf merged", i, etav[i],etav[i-1]);
    //     etav.erase(etav.begin()+i);
    //     Rv.erase(Rv.begin()+i);
    //     Tv.erase(Tv.begin()+i);
    //     Rppv.erase(Rppv.begin()+i);
    //     chiv.erase(chiv.begin()+i);
    //     pfv.erase(pfv.begin()+i);
    //   }
  }
  double mA = std::sqrt(chiv.back())/fA;

  /* Find eta 1 */

  double eta1, R1, chi1;
  LogMsg(VERB_NORMAL,"[Cos] Finding eta1");
  LogFlush();
  {
    std::vector<double>	letav;
    for (int i; i < etav.size();i++)
      letav.push_back(std::log10(etav[i]));

    double lfA = std::log10(fA);
    sR.set_points(letav,pfv);
    double leta0 = letav.front() ;
    double leta2 = letav.back() ;
    double lfA0  = pfv.front();
    double lfA2  = pfv.back();
    double errr = 1.0 ;
    double leta1, lfA1, lslo;
    while (std::abs(errr) > 0.00001)
    {
      lslo  = (lfA2-lfA0)/(leta2-leta0);
      leta1 = leta0 + (lfA-lfA0)/lslo;
      lfA1  = sR(leta1);
      errr  = 1.0 - lfA1/lfA;
      if (std::abs(lfA2-lfA) < std::abs(std::abs(lfA0-lfA))){
        leta0 = leta1; lfA0 = lfA1;
      } else {
        leta2 = leta1; lfA2 = lfA1;
      }
      // LogOut("err %e \n",errr);
    }

    eta1 = pow(10.,leta1);
    LogMsg(VERB_NORMAL,"[Cos] eta1 found %e",eta1);
    sR.set_points(etav,Rv);
    R1 = sR(eta1);
    LogMsg(VERB_NORMAL,"[Cos] R1   found %e",R1);
    schi.set_points(etav,chiv);
    chi1 = schi(eta1);
    LogMsg(VERB_NORMAL,"[Cos] chi1 found %e",chi1);

  }

  /* Normalise R, chi,
    but not Rpp, because we have tabulated
    Rpp = eta^2 R''/R which does not depend on the units of c-time
    in ADM units we thus have R_ctct/R = Rpp/ct^2
    should we tabulate however the log, which extrapolates safely to 0? */

  for (int i; i < etav.size();i++){
    etav[i] /= eta1;
    Rv[i]   /= R1;
    chiv[i] /= chi1;
  }
  /* Create Splines */
  sR.set_points(etav,Rv);
  sT.set_points(etav,Tv);
  sRpp.set_points(etav,Rppv);
  schi.set_points(etav,chiv);
  LogMsg(VERB_NORMAL,"[Cos] Setup of Cosmos finished",chi1);

  if (commRank() == 0 ){
    FILE *file_co ;
    file_co = NULL;
    char baseco[256];
    sprintf(baseco, "out/cosmos.txt");
    file_co = fopen(baseco,"w+");
    for (size_t i=0; i<etav.size(); i++)
      fprintf(file_co,"%e %e %e %e %e %e\n",etav[i], Rv[i], Tv[i], Rppv[i], chiv[i], pfv[i]);
    fclose(file_co);
  }
  LogMsg(VERB_NORMAL,"[Cos] Setup of Cosmos printed\n",chi1);

  LogOut("\n");
  LogOut("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n");
  LogOut("*-*        QCD jaxi-cosmo.txt       *-*\n");
  LogOut("*-*                                 *-*\n");
  LogOut("*-*      fA = %.2e GeV          *-*\n",fA);
  LogOut("*-*      mA = %.2e  eV          *-*\n",1.0e+9*mA);
  LogOut("*-*      T1 = %.2e GeV          *-*\n",sT(1.0)/1000.);
  LogOut("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n");
}



double  Cosmos::TopSus     (const double ct)
{
  return schi(ct);
}

double  Cosmos::Rpp  (const double ct)
{
  double rpp;
  if (ueCosm)
    rpp = sRpp(ct)/(ct*ct);
  else
  {
    // R = ct^frw
    // R''/R = frw(frw-1)/ct^2
  	//except in the case where frw = 0,1
  	rpp = frw*(frw-1.0)/(ct*ct) ;
  }
LogMsg(VERB_PARANOID,"[Cos:Rpp] Rpp %.2e ",rpp);
  return rpp;
}

double  Cosmos::Rp  (const double ct)
{
  double rp;
  if (ueCosm) {
    rp = (R(ct+1.e-3)-R(ct-1.e-3))/(2.e-3 * R(ct));
    LogMsg(VERB_NORMAL,"[Cos] calculating Rp =R(ct+e)-R(ct-e)/2eR = %e",rp);
  }
  else {
  // R = ct^frw
  // R'/R = frw/ct
	rp = frw/(ct) ;
  LogMsg(VERB_PARANOID,"[Cos] Rp = %e ",rp);
  }
  return rp;
}

double  Cosmos::R       (const double ct)
{
  double r ;
  if (ueCosm)
    r = sR(ct);
  else
    if (frw == 1 && icdatastruc.grav > 0.0)
      r = ct * (1 + 0.25 * ct * icdatastruc.grav);
    else
      r = std::pow(ct,frw);

  LogMsg(VERB_PARANOID,"[Cos:R] R %.2e ",r);
  return r;
}

double  Cosmos::T       (const double ct)
{
LogMsg(VERB_PARANOID,"[Cos:T] T %.2e ",sT(ct));
  return sT(ct);
}

double  Cosmos::AxionMass2 (const double ct)
{
  double RNow = R(ct);
  double aMass2;

  if (ueCosm)
    aMass2 = schi(ct);
  else
  {
    if (zThRes <= zRestore) /* mode restore */
    {
      if (RNow < zThRes)
        aMass2 = indi3*indi3*pow(RNow, nQcd);

      if (RNow >= zThRes && RNow <= zRestore)
        aMass2 = indi3*indi3*pow(zThRes, nQcd);

      if (RNow > zRestore)
        aMass2 = indi3*indi3*pow(zThRes, nQcd)*pow(RNow/zRestore, nQcdr);

    } else { /* mode saturate */

      if (RNow < zThRes)
        aMass2 = indi3*indi3*pow(RNow, nQcd);
      else
        aMass2 = indi3*indi3*pow(zThRes, nQcd);
    }
  }
LogMsg(VERB_PARANOID,"[Cos:mA2] mA2 %.2e ",aMass2);
    return aMass2;
}


double  Cosmos::DAxionMass2Dct (const double ct)
{
  double RNow = R(ct);
  double dMass2;

  if (ueCosm){
    dMass2 = (schi(ct+1.e-6)-schi(ct-1.e-6))/2.e-6;
  }
  else
  {
    if (zThRes <= zRestore) /* mode restore */
    {
      if (RNow < zThRes)
        dMass2 = nQcd*frw*AxionMass2(ct)/ct;

      if (RNow >= zThRes && RNow <= zRestore)
        dMass2 = 0;

      if (RNow > zRestore)
        dMass2 = nQcdr*frw*AxionMass2(ct)/ct;

    } else { /* mode saturate */
      if (RNow < zThRes)
        dMass2 = nQcd*frw*AxionMass2(ct)/ct;
      else
        dMass2 = 0.0;
    }
  }
LogMsg(VERB_PARANOID,"[Cos:] dmA2dt %.2e ",dMass2);
  return dMass2;
}

/*logarithmic derivative of m_AR with respect to time */
double  Cosmos::DlogMARDlogct (const double ct)
{
  if (ueCosm){
    double e = 1.e-6;
    return (ct*std::sqrt(schi(ct+e))*R(ct+e)-std::sqrt(schi(ct-e))*R(ct-e))/(2.*e*std::sqrt(schi(ct))*R(ct));
  }
  else {
    /*(nqcd/2 + 1)*frw*/
    double RNow = R(ct);
    double dlmRlct;

    if (zThRes <= zRestore) /* mode restore */
    {
      if (RNow < zThRes)
        dlmRlct = frw*(nQcd/2+1);

      if (RNow >= zThRes && RNow <= zRestore)
        dlmRlct = frw;

      if (RNow > zRestore)
        dlmRlct = frw*(nQcdr/2+1);

    } else { /* mode saturate */
      if (RNow < zThRes)
        dlmRlct = frw*(nQcd/2+1);
      else
        dlmRlct = frw;
    }

LogMsg(VERB_PARANOID,"[Cos:] DlogMARDlogct %.2e ",dlmRlct);
    return dlmRlct;
    }
}

double	Cosmos::LambdaP (double ct)
{
LogMsg(VERB_PARANOID,"[Cos:LambdaP] LambdaPhysical %e Le %e",lambda,lz2e);
  return  lambda/pow(R(ct),lz2e);
}

double  Cosmos::SaxionMass2  (double ct)
{
	double lbd   = LambdaP(ct);

	switch  (pot & V_PQ) {
		case    V_PQ1:
			lbd *= 2.;
			break;

		case    V_PQ2:
      lbd *= 8.;
			break;

		default :
			lbd *= 0;
			break;
	}
  LogMsg(VERB_PARANOID,"[Cos:ms2] ms2 %.2e ",lbd);
	return  lbd;
}
