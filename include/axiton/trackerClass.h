#ifndef	_TRACKERCLASS_
	#define	_TRACKERCLASS_

	#include <string>
	#include <complex>
	#include <memory>
	#include "scalar/scalarField.h"
	#include "scalar/folder.h"
	#include "enum-field.h"

	#ifdef	USE_GPU
		#include <cuda.h>
		#include <cuda_runtime.h>
		#include <cuda_device_runtime_api.h>
	#endif

	#include "utils/utils.h"
	#include "fft/fftCode.h"
	#include "comms/comms.h"


	#include "io/readWrite.h"

	class	Axiton
	{
		private:
		size_t idx;
		int i0;
		std::vector<double>	m;
		std::vector<double>	v;
		std::vector<double>	e;

		public:

			Axiton(){ };
			~Axiton(){ };

			void	SetIdx	  (size_t id, int i) {idx = id; i0=i;};
			void	AddPoint	(double m1, double v1) {m.push_back(m1); v.push_back(v1);};
			double* Field	() { return m.data();};
			double* Veloc	() { return v.data();};
			size_t 	Idx	()   { return idx;};
			size_t 	Size	() { return m.size();};
			int	I0	()       { return i0;};
	};


	class Group 
	{
		/* This will contain group of axiton candidates or minicluster candidates 
		   On Axion mode groups into axitons, on Paxion mode gropus into miniclusters
		*/
		private:
		// an ID
		// a vector containing the positions
		// a vector containing the density
		// a double to store the total mass
		// a double to store the maximum density
		// a double to store an estimation of the size
		 
		public:
			Group (){ };
			~Group(){ };

			void setID (){};
			void addGroup(){};
	};



	class	Tracker : public Tunable
	{
		private:

		Scalar			*afield;

		const FieldPrecision	precision;
		const size_t		Lx;
		const size_t		Lz;
		const size_t		Tz;
		const size_t		S;
		const size_t		V;
		const size_t		Ng;
		const size_t		BO;

		const double k0, norm;
		const size_t hLx;
		const size_t hLy;
		const size_t hTz;

		const size_t nModeshc;
		const size_t dl;
		const size_t pl;
		const size_t ss;

		size_t zBase;
		size_t fSize;
		size_t shift;

		PropParms ppar;

		std::vector<Axiton>	Axilist;

		std::vector<size_t> idxlist;
		std::vector<double>	ct;

		int limit;
		int index;
		double ct_th;
		double th_th;
		double ve_th;
		int printradius;
		bool gradients;

		bool addifm;
		bool addifv;
		void	report	(int index, int red, int pad, int fft, int slice, char *name);

		template<typename Float>
		int	ReadAxitons	() ;

		template<typename Float>
		int	SearchAxitons ();

		public:

			/* Constructor */

			Tracker (Scalar *field) : afield(field), precision(afield->Precision()), Lx(afield->Length()), Lz(afield->Depth()), Tz(afield->TotalDepth()),
								S(afield->Surf()), V(afield->Size()), Ng(afield->getNg()), BO(Ng*S),
								k0((2.0*M_PI/ (double) afield->BckGnd()->PhysSize())),  hLx((Lx >> 1)+1), hLy(Lx >> 1), hTz(Tz >> 1), nModeshc(Lx*hLx*Lz),
								norm( -1./(k0*k0*((double) afield->TotalSize()))), dl(precision*Lx), pl(precision*(Lx+2)), ss(Lz*Lx),
								fSize(afield->DataSize()), shift(afield->DataAlign()/fSize) {

					/* generic; check if needed TODO */
					zBase = Lx/commSize()*commRank();
					ppar.Ng    = Ng;
					ppar.ood2a = 1.0;
					ppar.PC    = afield->getCO();
					ppar.Lx    = Lx;
					ppar.Lz    = Lz;

					/* Specific */
					addifm = false;
					addifv = false;
					{
					LogMsg(VERB_HIGH,"[AT] Axiton Tracker config");
					index  = 0;
					LogMsg(VERB_HIGH,"     index %d",index);

					int iaux = afield->BckGnd()->ICData().axtinfo.nMax;
					if (iaux == -1){
						LogError("Axiton tracker should not have initialised: closing.");
						return;
					}
					else if (iaux == -2)
						iaux = 100;
					limit  = iaux/commSize();
					LogMsg(VERB_HIGH,"     nMax  %d",limit);

					double aux = afield->BckGnd()->ICData().axtinfo.th_threshold;
					if (aux == -1.)
						th_th = M_PI;
					else
						th_th = aux;
					addifm = true;
					LogMsg(VERB_HIGH,"     thetha_threshold  %.2f",th_th);

					aux = afield->BckGnd()->ICData().axtinfo.ve_threshold;
					if (aux == -1.){
						ve_th  = 1.0;
						addifv = false;
						LogMsg(VERB_HIGH,"     cvelocity_threshold (disabled) but report if v > %.2f [mAR^2]",ve_th);
					}
					else{
						ve_th  = aux;
						addifv = true;
						LogMsg(VERB_HIGH,"     cvelocity_threshold  %.2f [mAR^2]",ve_th);
					}

					aux = afield->BckGnd()->ICData().axtinfo.ct_threshold;
					if (aux == -1.)
						ct_th = 2.5;
					else
						ct_th = aux;
					LogMsg(VERB_HIGH,"     ct_threshold  mA*R*ct > %.2f",ct_th);

					LogMsg(VERB_HIGH,"     printradius ignored",ct_th);
					LogMsg(VERB_HIGH,"     gradients ignored",ct_th);
					}

			}

		 	~Tracker()  {};

			void	SetLimit	(int newlim) {limit = newlim;} ;
			int	  SearchAxitons	() ;
			bool	AddAxiton	(size_t id) ;
			void	Update () ;
			void	PrintAxitons () ;
	};


	/* Loops over the idx list, reads, Field values and velocities of axitons and stores them  */


	void	Tracker::Update ()
	{
		if (afield->Device() == DEV_GPU)
			return;

		if (precision == FIELD_DOUBLE)
		{
			this->ReadAxitons<double>();
		}
			else
		{
			this->ReadAxitons<float>();
		}
	}

	template<typename Float>
	int	Tracker::ReadAxitons ()
	{

		ct.push_back(*afield->zV());

		Float *m = static_cast<Float*>(afield->mStart());
		Float *v = static_cast<Float*>(afield->vStart());

		if (afield->Folded()){
			#pragma omp parallel for schedule(static)
			for (int iidx = 0 ; iidx < idxlist.size(); iidx++)
			{
					size_t idx = Axilist[iidx].Idx();
					Axilist[iidx].AddPoint( (double) m[idx], (double) v[idx]);
					/* energy? */
			}
		} else {
			#pragma omp parallel for schedule(static)
			for (int iidx = 0 ; iidx < idxlist.size(); iidx++)
			{
					size_t fidx = Axilist[iidx].Idx();
					size_t X[3];
					indexXeon::idx2Vec (fidx, X, Lx);
					size_t sy  = X[1]/(Lx/shift);
					size_t iiy = X[1] - (sy*Lx/shift);
					size_t idx = X[2]*S + shift*(iiy*Lx+X[0]) +sy;
					Axilist[iidx].AddPoint( (double) m[idx], (double) v[idx]);
					/* energy? */
			}

		}
		index++;
	}



	/* scans the grid for occurrences of a criterion and creates an axiton if needed to be tracked */

	int	Tracker::SearchAxitons ()
	{
		if (afield->Device() == DEV_GPU)
			return -1;

		if (afield->Field() == FIELD_AXION)
		{
			if (afield->AxionMass()*(*afield->RV())*(*afield->zV()) < ct_th)
			return -1;
		}

		// ADD TEMPORAL CHECK FOR PAXION (MINICLUSTERS)

		int co ;
		if (precision == FIELD_DOUBLE)
		{
			co = this->SearchAxitons<double>();
		}
			else
		{
			co = this->SearchAxitons<float>();
		}
		return co;
	}

	template<typename Float>
	int	Tracker::SearchAxitons ()
	{
		const int nThreads = commThreads(); // IS THIS USEFUL?

		if (afield->Field() == FIELD_AXION)
		{
			Float *m = static_cast<Float*>(afield->mStart());
			Float *v = static_cast<Float*>(afield->vStart());

			Float mlim = (Float) (th_th)* (*afield->RV());
			Float vlim = (Float) (ve_th)* afield->AxionMass()*(*afield->RV())*(*afield->RV());

			int axitt = 0;
			int axitm = 0;
			int axitv = 0;
			int axitn = 0;
			#pragma omp parallel for schedule(static) reduction (+:axitm,axitt,axitv, axitn)
			for (size_t iidx = 0 ; iidx < V; iidx++)
			{
				int max = false;
				int vax = false;
				if ( std::abs(m[iidx]) > mlim )
					max = true;
				if ( std::abs(v[iidx]) > vlim )
					vax = true;
				if ( (max && addifm) || (vax && addifv))
				{
						size_t esta = iidx;
						if (!afield->Folded())
						{
							/* fidx = iZ*n2 + iiy*shift*Lx +ix*shift +sy
								idx  = iZ*n2 + [iy+sy*(n1/shift)]*Lx +ix
								ix   = rix*ref , etc... */
							size_t X[3];
							indexXeon::idx2Vec (iidx, X, Lx);
							size_t sy  = X[1]/(Lx/shift);
							size_t iiy = X[1] - (sy*Lx/shift);
							esta = X[2]*S + shift*(iiy*Lx+X[0]) +sy;
						}

						axitt++;

						if (max)
							axitm++;

						if (vax)
							axitv++;

						bool bola = false;

						#pragma omp critical (writeaxiton)
							bola = AddAxiton(esta);

						if (bola)
							axitn++;
				}
			}
			LogMsg(VERB_PARANOID,"[AT] Search axitons returned %d (%d/%d with m/v criterion) but only %d new",axitt,axitm,axitv,axitn);
			return axitm;
		}
		else if (afield->Field() == FIELD_PAXION)
		{
			/* HALO CODE HERE */
			 

			Float *m2 = static_cast<Float*>(afield->m2Start());
			Float *m2h = static_cast<Float*>(afield->m2hStart());
			char *tag = static_cast<char *>(static_cast<void *>(field->sData()));

			memset(afield->m2half(),0,afield->eSize()*afield->Precision());
			memset(tag,0,afield->Size());

			Float m2lim = (Float) (en_th); //en_th will be an element of AxitInfo Struct
			
			int halott = 0;

			#pragma omp parallel for schedule(static) reduction (+:halott)
			for (size_t iidx = 0 ; iidx < V; iidx++)
			{
				int max = false;
				if ( std::abs(m2[iidx]) > m2lim )
				{
					max = true;
					tag[iidx] = STRING_MASK;

					size_t esta = iidx;
					if (!afield->Folded())
					{
						/* fidx = iZ*n2 + iiy*shift*Lx +ix*shift +sy
							idx  = iZ*n2 + [iy+sy*(n1/shift)]*Lx +ix
							ix   = rix*ref , etc... */
						size_t X[3];
						indexXeon::idx2Vec (iidx, X, Lx);
						size_t sy  = X[1]/(Lx/shift);
						size_t iiy = X[1] - (sy*Lx/shift);
						esta = X[2]*S + shift*(iiy*Lx+X[0]) +sy;
					}

					// if (max)
					// 	halott++;

					bool bola = false;
					#pragma omp critical (writeaxiton)
					{
						m2h[iidx] = halott;
						bola = AddAxiton(esta);
						if (bola)
							halott++;
					}

				}
				
			}
		}	
		else 
		{
			LogError("Tracker class is only valid in Axion/Paxion mode");
			return -1;
		}
	}

	// Running serial?
	int Tracker::GroupAxitons()
	{
	
		if (afield->Folded()){
			//#pragma omp parallel for schedule(static)
			for (int iidx = 0 ; iidx < idxlist.size(); iidx++)
			{
				size_t idx = Axilist[iidx].Idx();
				//Axilist[iidx].AddPoint( (double) m[idx], (double) v[idx]);
				/* energy? */
			}
		} else {
			//#pragma omp parallel for schedule(static)
			for (int iidx = 0 ; iidx < idxlist.size(); iidx++)
			{
				size_t fidx = Axilist[iidx].Idx();
				size_t X[3];
				indexXeon::idx2Vec (fidx, X, Lx);
				size_t sy  = X[1]/(Lx/shift);
				size_t iiy = X[1] - (sy*Lx/shift);
				size_t idx = X[2]*S + shift*(iiy*Lx+X[0]) +sy;

				size_t idxPx, idxMx, idxPy, idxMy, idxPz, idxMz,X[3],O[4];
				indexXeon::idx2VecNeigh(idx,X,O,totlX);				
				idxPx = O[0];
				idxMx = O[1];
				idxPy = O[2];
				idxMy = O[3];
				idxPz = idx + S;
				idxMz = idx - S;

				if (m2h[idxPx] > 0)
					m2h[idxPx] = min(m2h[idx],m2h[idxPx]);

				if (m2h[idxMx] > 0)
					m2h[idxMx] = min(m2h[idx],m2h[idxMx]);

				if (m2h[idxPy] > 0)
					m2h[idxPy] = min(m2h[idx],m2h[idxPy]);

				if (m2h[idxMy] > 0)
					m2h[idxMy] = min(m2h[idx],m2h[idxMy]);
				
				if (m2h[idxPz] > 0)
					m2h[idxPz] = min(m2h[idx],m2h[idxPz]);

				if (m2h[idxMz] > 0)
					m2h[idxMz] = min(m2h[idx],m2h[idxMz]);

				// We might need to check if neighbours have index smaller than center

	

				



				//Axilist[iidx].AddPoint( (double) m[idx], (double) v[idx]);
				/* energy? */
			}



			vector<Float> groups;
			for (int iidx = 0 ; iidx < idxlist.size(); iidx++)
			{
				Float preid = m2h[idx];
				
				{
					if ( std::find(idxlist.begin(), idxlist.end(), id) != idxlist.end()  )
						return false;
					else 
					{
						Group newgroup;
						newaxiton.SetIdx(id,index);
						Axilist.push_back(newaxiton);
						idxlist.push_back(id);
						return true;
					}	
				}
			}
			

		}

	



		return group_id;
	}



	/* Creates an axiton and pushes it into the axiton-list together with the index */


	bool	Tracker::AddAxiton (size_t id)
	{
		if (afield->Device() == DEV_GPU)
			return false;


		if ( idxlist.size() >= limit){
			LogMsg(VERB_PARANOID,"[AT] Axiton limit (%d) exceeded rejecting axiton %d",limit, id);
			return false;
		}

		if ( std::find(idxlist.begin(), idxlist.end(), id) != idxlist.end()  )
			return false;
		else {
			Axiton newaxiton;
			newaxiton.SetIdx(id,index);
			Axilist.push_back(newaxiton);
			idxlist.push_back(id);
			return true;
		}
	}




    void	Tracker::PrintAxitons (){

	if (afield->Device() == DEV_GPU)
		return;

	LogMsg(VERB_NORMAL,"[PA] Print %d axitons",Axilist.size());LogFlush();
	// char outName[128] = "axion\0";
	commSync();
	char oldName[128] ;
	sprintf(oldName, outName);
	sprintf(outName, "notixa\0");

	createMeas(afield, 0);

	char group[256];
	sprintf(group, "/");
	writeArray(&ct.data()[0], ct.size(), group, "ct",0);

	int nAx[commSize()];
	int nAx_l = Axilist.size();
	MPI_Allgather (&nAx_l, 1, MPI_INT, nAx, 1, MPI_INT, MPI_COMM_WORLD);

	nAx_l = 0;
	for (int i = 0; i < commSize(); i++){
		LogMsg(VERB_HIGH,"rank %d found %d Axitons",i,nAx[i]);
		nAx_l += nAx[i];
	}
LogMsg(VERB_HIGH,"Total %d",nAx_l);
	LogFlush();

	/* each rank writes his own stuff but all have to create groups */
	int writtingrank = 0;
	int locax = 0;
	for (int i = 0 ; i < nAx_l ; i++  )
	{
		if (locax > nAx[writtingrank]-1){
			writtingrank++;
			locax = 0;
		}


		LogMsg(VERB_HIGH,"Rank %d will write Axiton %d / %d/%d",writtingrank,i,locax,nAx[writtingrank]);LogFlush();
		sprintf(group, "/axiton%3d",i);

		double *basura = 0;
		double *vasura = 0;
		size_t X[5] = {0,0,0,0,0};

		if (commRank() == writtingrank){
			/* might be that other ranks do not have the same number of axitons */
			basura = &(Axilist[locax].Field()[0]);
			vasura = &(Axilist[locax].Veloc()[0]);

			/* i0 and size */
			X[3] = Axilist[locax].I0();
			X[4] = Axilist[locax].Size();

			/* unfold coordinates */
			size_t idx = Axilist[locax].Idx();
			X[2] = idx/S;
			size_t xy  = idx - X[2]*S;
			size_t iiy = xy/(Lx*shift);
			xy  -= iiy*(Lx*shift);
			X[0] = xy/shift;
			size_t sy = xy - X[0]*shift;
			X[1] = iiy + sy*(Lx/shift);
			X[2] += writtingrank*(Lz);
		}

		MPI_Bcast (&X[0], 5, MPI_UNSIGNED_LONG, writtingrank, MPI_COMM_WORLD);
		LogMsg(VERB_HIGH,"Bcast %d %d %d i0 %d size %d",X[0], X[1], X[2], X[3], X[4]);LogFlush();
		// writeArray(Axilist[i].Field(), Axilist[i].Size(), group, "field", writtingrank);
		// writeArray(Axilist[i].Veloc(), Axilist[i].Size(), group, "veloc", writtingrank);
		writeArray(basura, X[4], group, "field", writtingrank);
		writeArray(vasura, X[4], group, "veloc", writtingrank);

		writeAttributeg	(&X[0], group, "x", H5T_NATIVE_INT);
		writeAttributeg	(&X[1], group, "y", H5T_NATIVE_INT);
		writeAttributeg	(&X[2], group, "z", H5T_NATIVE_INT);
		writeAttributeg	(&X[3], group, "i0", H5T_NATIVE_INT);

		locax++;
		commSync();
	}

	destroyMeas();

	commSync();
	sprintf(outName, oldName);


	}


	template<typename Float>
	int Tracker::SearchHalos()
	{
		/* Assumes density contrast in m2 
			TO INTRODUCE:
			- halo_thr	
			- halom2 as a varibale of the tracjer Class
		*/
		Float *m2 = static_cast<Float*>(afield->m2Start());
		Float dens_thr = (Float) (halo_th); //halo_thr might be element of HaloInfo Struct

		const int nThreads = commThreads();

		int halott = 0;
		Float halo_thr = ;//ADD_HERE!

		#pragma omp parallel for schedule(static) reduction (+:halott)
		for (size_t iidx = 0 ; iidx < V; iidx++)
		{
			int max = false;
			if ( std::abs(m2[iidx]) > halo_thr )
				max = true;

			if (max && addifm)
			{
				size_t esta = iidx;
				if (!afield->Folded())
				{
					/* fidx = iZ*n2 + iiy*shift*Lx +ix*shift +sy
						idx  = iZ*n2 + [iy+sy*(n1/shift)]*Lx +ix
						ix   = rix*ref , etc... */
					size_t X[3];
					indexXeon::idx2Vec (iidx, X, Lx);
					size_t sy  = X[1]/(Lx/shift);
					size_t iiy = X[1] - (sy*Lx/shift);
					esta = X[2]*S + shift*(iiy*Lx+X[0]) +sy;
				}

				halott++;

				if (max)
					halom2++;

				bool bola = false;

				#pragma omp critical (writeaxiton)
					bola = AddHaloPoint(esta);

				if (bola)
					halonew++;
			}
		}
		LogMsg(VERB_PARANOID,"[AT] Search Halos returned %d (%d with m2 criterion) but only %d new",halott,halom2,halonew);
		return axitm;

	}


#endif
