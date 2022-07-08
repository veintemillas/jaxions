#ifndef	_TRACKERCLASS_
	#define	_TRACKERCLASS_

	#include <string>
	#include <iostream>
	#include <complex>
	#include <memory>
	#include "scalar/scalarField.h"
	#include "scalar/folder.h"
	#include "enum-field.h"
	#include "gravity/gravityPaxionXeon.h"

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

		size_t haloid;

		public:
			/* Constructor */
			Axiton() : haloid(-1){};

			~Axiton(){ };

			void	SetIdx	  (size_t id, int i) {idx = id; i0=i;};
			void	SethaloIdx (int id) {haloid = id; };

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
		   On Axion mode groups into axitons, on Paxion mode groups into miniclusters
		*/
		private:
		int gidx;    	// group id
		int gidxB;    // group id of the group with points in the previus rank
		int gidxF;    // group id of the group with points in the next rank
		int npoints; // number of points
		std::vector<size_t> pfidxlist ; // a vector containing the positions foldedIds
		std::vector<double> pmlist ;    // a vector containing a field value
		std::vector<double> pvlist ;    // a vector containing another field value
		std::vector<double> pm2list ;   // a vector containing density contrast

		// a double to store the total mass
		// a double to store the maximum density
		// a double to store an estimation of the size

		public:
			Group () : gidx(-1),npoints(0) { };
			~Group() { };

			void    setID (size_t gidx){gidx=gidx;};
			void    setIDF (size_t gidxF){gidxF=gidxF;};
			void    setIDB (size_t gidxB){gidxB=gidxB;};
			void    addGroup(){};
			size_t* IdxList() {return pfidxlist.data();};

			int     NPoints() {return npoints;};
			int     gID()     {return gidx;};
			bool    AddPoint (size_t fidx)
			{
				if ( std::find(pfidxlist.begin(), pfidxlist.end(), fidx) != pfidxlist.end()  )
					return false;
				else {
					pfidxlist.push_back(fidx);
					npoints++;
					return true;
				}
			};

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

		std::vector<Axiton>	Axilist;  // axiton list
		std::vector<Group*>	Halolist; //group/halo list


		std::vector<size_t> idxlist;
		std::vector<size_t> cidxlist; // candidate list
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

		double en_th;

		void	report	(int index, int red, int pad, int fft, int slice, char *name);

		template<typename Float>
		int	ReadAxitons	() ;

		template<typename Float>
		int	SearchAxitons ();

    	template<typename Float>
		int GroupTags ();

		// template<typename Float>
		// int SearchHalos();

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

					aux = afield->BckGnd()->ICData().axtinfo.con_threshold;
					if (aux == -1.)
						en_th = 100.0;
					else
						en_th = aux;
					LogMsg(VERB_HIGH,"     con_threshold  en/<en> > %.2f",en_th);

					LogMsg(VERB_HIGH,"     printradius ignored",ct_th);
					LogMsg(VERB_HIGH,"     gradients ignored",ct_th);
					}

			}

		 	~Tracker()  {};

			void	SetLimit	(int newlim) {limit = newlim;} ;
			void	SetEThreshold	(double lola) {en_th = lola;} ;
			int   SearchAxitons	() ;
			int   GroupTags	() ;
			bool  PatchGroups () ;
			bool	AddAxiton	(size_t fidx) ;
			bool	AddGroup	(Group* newg) ;
			void	AddTagPoint	(size_t idx) {cidxlist.push_back(idx);}; // saved unfolded to simplify neighbours a bit

			void	  Update () ;
			void	  PrintAxitons () ;
			size_t  foldidx(size_t idx);
			size_t  unfoldidx(size_t fidx);
	};


	/* Loops over the idx list, reads, Field values and velocities of axitons and stores them  */


	void	Tracker::Update ()
	{
		LogMsg(VERB_HIGH," [UA] Update Axitons");
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

		LogMsg(VERB_HIGH," [UA] Read Axitons");
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
		commSync();
		LogMsg(VERB_HIGH," [UA] Axitons read, index = %d (number of times?)",index);

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
		LogMsg(VERB_NORMAL,"[SA] Searching axitons");
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
			LogMsg(VERB_NORMAL,"[SA] Paxion Mode");
			/* HALO CODE HERE
			Make sure the paxion energy is in m2S */
			void *nada;
			PropParms ppar;
			ppar.Ng    = Ng;
			ppar.ood2a = 1.0;
			ppar.PC    = afield->getCO();
			ppar.Lx    = Lx;
			ppar.Lz    = Lz;
			LogMsg(VERB_NORMAL,"[SA] Energy to m2Start");
			LogMsg(VERB_NORMAL,"[SA] Check B0 %d V %d Precision %d xB %d yB %d zB %d",BO, V, afield->Precision(), xBlock, yBlock, zBlock);
			graviPaxKernelXeon<KIDI_ENE>(afield->mCpu(), afield->vCpu(), nada, afield->m2Cpu(), ppar, BO, V+BO, afield->Precision(), xBlock, yBlock, zBlock);

			Float *m2 = static_cast<Float*>(afield->m2Start());
			Float *m2h = static_cast<Float*>(afield->m2hStart());
			char *tag = static_cast<char *>(static_cast<void *>(afield->sData()));

			LogMsg(VERB_NORMAL,"[SA] Resseting");
			memset(afield->m2half(),0,afield->eSize()*afield->Precision());
			memset(tag,0,afield->Size());

			Float m2lim = (Float) (en_th);

			/* For candidate point IDs */
			int candipointcounter = 0;
			int accepted = 1;

			LogMsg(VERB_NORMAL,"[SA] Mainloop, contrast limit %f",m2lim);
			#pragma omp parallel for schedule(static) reduction(+:candipointcounter,accepted)
			for (size_t iidx = 0 ; iidx < V; iidx++)
			{

				if ( std::abs(m2[iidx]) > m2lim )
				{
					// LogMsg(VERB_PARANOID,"[AT] Paxitons %f > %f ",m2[iidx],m2lim);
					candipointcounter++;
					tag[iidx] = STRING_MASK;

					/* Canditates are saved unfolded */
					#pragma omp critical 
					AddTagPoint(afield->Folded() ? unfoldidx(iidx) : iidx);
					m2h[iidx] = accepted;

						// if follow in time is wanted
						// bool bola = false;
						// size_t esta = afield->Folded() ? iidx : foldidx(iidx);
						// #pragma omp critical (candypoint)
						// 	bola = AddAxiton(esta); // Axitons track evolution in time, ID is fidx
						// if (bola)
					accepted++;


				}

			} // end main loop
			LogMsg(VERB_NORMAL,"[AT] Search Candidate Points returned/accepted %d/%d ",candipointcounter,accepted);
			return accepted;
		} // end case PAXION
		else
		{
			LogError("Tracker class is only valid in Axion/Paxion mode");
			return -1;
		}
	}

	int	Tracker::GroupTags ()
	{
		LogMsg(VERB_NORMAL,"[GA] Group Candidates...");
		if (afield->Device() == DEV_GPU)
			return -1;

		// if (afield->Field() == FIELD_AXION)
		// {
		// 	if (afield->AxionMass()*(*afield->RV())*(*afield->zV()) < ct_th)
		// 	return -1;
		// }

		// ADD TEMPORAL CHECK FOR PAXION (MINICLUSTERS)

		int co ;
		if (precision == FIELD_DOUBLE)
		{
			co = this->GroupTags<double>();
		}
			else
		{
			co = this->GroupTags<float>();
		}
		return co;
	}

	// Running serial?
	template<typename Float>
	int Tracker::GroupTags()
	{
		LogMsg(VERB_NORMAL,"[GA] Group Tagged points CPU (con_th %.1f)",en_th);
		/* Make neighbouring points have the same tag in m2h */

		/* If groups already exist, do something else */

		/* If groups do not exist and we need to process all info */

		/* Exchange Ghosts in m2h */

		int groupId = 1;

		/* It should work for folded and unfolded fields */
		{

			LogMsg(VERB_NORMAL,"[GA] Group Axitons (Unfolded)");
			Float *m2h = static_cast<Float*>(afield->m2hStart());
			char  *tag = static_cast<char*>(afield->sData());

			/* Loop over points (Axitons),

			copies them into a newgroup
			tags in sData:
			STRING_WALL        = point in the group, already checked for neighbours
			STRING_XY_POSITIVE = point in the group
			STRING_MASK        = point belonging to some group (from SearchAxitons)*/

			#pragma omp parallel for schedule(static) shared(groupId)
			for (int iidx = 0 ; iidx < cidxlist.size(); iidx++)
			{
				/* Finds the idx of the tagged point in the tracker list */
				size_t idx  = cidxlist[iidx];   // element iidx of the tagged list in tracker
				size_t fidx = foldidx(idx);     // in case we need it ...
				size_t midx = afield->Folded() ? fidx : idx;

				LogMsg(VERB_PARANOID,"[GA] Seed idx %d midx %d (m2h %f)",idx, midx, m2h[midx]);
				/* Creates a group unless the point is already classified/labelled in a group
				remember that STRING_MASK means point-candidate (Axiton)
				we use STRING_WALL for those points which are already grouped in a closed group
				THAT WOULD BE WEIRD */
				LogMsg(VERB_PARANOID,"[GA] tag %d",tag[midx]);
				if (tag[idx] & STRING_WALL){
					LogMsg(VERB_PARANOID,"[GA] already studied ... continue %d",idx);
					continue;
				} 

				LogMsg(VERB_PARANOID,"[GA] Seed %d (m2h %f) will span GROUP %d",idx,m2h[midx],groupId);
				/* Creates a group for point idx with reserved memory*/
				Group *newgroup = new Group();
				newgroup->setID(groupId);
				/* Adds the tagged point to the Group with unfolded idx */
				newgroup->AddPoint(idx);

				/* Main WHILE loop, checks for neighbours of all points in the temp group until all are tagged
				adds points to the group and iterates */
				int npoints = 1;
				int npointsincrease = 1; // any value >0 to make the while loop start

				LogMsg(VERB_PARANOID,"[GA] while");
				while (npointsincrease)
				{
					LogMsg(VERB_PARANOID,"[GA] Npoints %d increase %d ",newgroup->NPoints(),npointsincrease);
					/* We reset the counter */
					npointsincrease = 0;

					/* We make a copy of the points in newgroup to avoid race conditions
					the while loop keeps a hierarchy in the neighbours added */
					std::vector<size_t> temp_idx;
					for (int ii = 0; ii<newgroup->NPoints(); ii++)
					{
						size_t aidx = newgroup->IdxList()[ii];
						size_t maidx = afield->Folded()? foldidx(aidx) : aidx;
						LogMsg(VERB_PARANOID,"[GA] loop aux %d midx %d (tag %d)",aidx,maidx,tag[maidx] & STRING_WALL);
						/* but we include only those which have not been checked for neighbours */
						if ((tag[maidx] & STRING_WALL) == 0)
						{
							LogMsg(VERB_PARANOID,"[GA] idx %d AUX added %d (tag %d)",idx, aidx,tag[maidx] | STRING_XY_POSITIVE);
							temp_idx.push_back(aidx);         // we add the unfolded idx
							tag[maidx] |= STRING_XY_POSITIVE; // tag as belonging to A group
						}
					}
					LogMsg(VERB_PARANOID,"[GA] Run the idx2 loop");
					// for (const auto& idx2: temp_idx)
					for (size_t ii =0;ii < temp_idx.size(); ii++)
					{
						size_t idx2 = temp_idx.data()[ii];
						LogMsg(VERB_PARANOID,"[GA] idx2 %d ",idx2);
						/* Find idx of neighbours */
						size_t X[3], O[4];
						indexXeon::idx2VecNeigh(idx2,X,O,Lx);

						/* Creates a collection of neighbours to test */
						std::vector<size_t> nebo;
						nebo.push_back(O[0]);
						nebo.push_back(O[1]);
						nebo.push_back(O[2]);
						nebo.push_back(O[3]);
						/* Avoid ghost zones */
						if (idx2 < V-S)
							nebo.push_back(idx2+S);
						if (idx2 > S)
							nebo.push_back(idx2-S);

						/* Loop over neighbours to incorporate them */
						for (const auto& nidx : nebo)
						{
							LogMsg(VERB_PARANOID,"[GA] nebo loop %d ",nidx);
							/* If candidates satisfy the criteriun include them in the group and tag them
							(UNLESS THEY ARE ALREADY! )*/
							size_t mnidx = afield->Folded()? foldidx(nidx) : nidx;
							if ((tag[mnidx] & STRING_XY_POSITIVE) == 0) // this avoids points already in the group
								if (m2h[mnidx] > 0)
								{
									if (newgroup->AddPoint(nidx)){
										npointsincrease++;
										LogMsg(VERB_PARANOID,"[GA] added %d ",nidx);
									}
									tag[mnidx] |= STRING_XY_POSITIVE; // tagged as grouped
									m2h[mnidx] = groupId;           ; // pasted groupId into m2h
								}
							}
						/* idx2 neighbouras should note be checked again */
						size_t midx2 = afield->Folded()? foldidx(idx2) : idx2;
						tag[midx2] |= STRING_WALL;
					}
					/* How many new points have been added ? */
					LogMsg(VERB_PARANOID,"[Group] group %d, %d new points have been added",groupId,npointsincrease);
				}

				/* Now Newgroup contains the unfolded IDs of all points in a group,
				and we can continue with the next point (which is not in a group) */

				/* Publish group into Halolist
				(we can now use the opportunity to fold the IDs, for instance)
				note that it already has the groupid inside */
				LogMsg(VERB_HIGH,"[Group] group %d with %d points",groupId,newgroup->NPoints());
				AddGroup(newgroup);	
				delete newgroup;
				groupId++;
			} //end candidate point list
			
			/* MPI mapping,
			exchange ghosts to associate groups from different ranks */
			const int sliceBytes = afield->Surf()*afield->Precision();
			void *sB = afield->m2hStart();
			void *rF = static_cast<void *> (static_cast<char *> (afield->m2hStart() + V*afield->Precision())); // m2BackGhost(); // slice after m
			void *sF = static_cast<void *> (static_cast<char *> (afield->m2hStart() + (V-S)*afield->Precision())); // static_cast<void *> (static_cast<char *> (m2Start()) +fSize*n3-ghostBytes);
			void *rB = afield->m2half();
			LogMsg(VERB_PARANOID,"[MB] send ghosts") ;LogFlush();
			afield->sendGeneral(COMM_SDRV, sliceBytes, MPI_BYTE, sB, rF, sF, rB);

			/* Go over ghost zones and create a dictionary
			loop over ranks from second and give them a global idx
			and label m2 with this global id */

			int ggid;
			int totalgroupnum;
			//if (commRank() == 0)
			ggid = Halolist.size();
			/* For displaying division of groups between ranks */
			// for (int i = 0; i<commSize()+1;i++)
			// {
			// 	if (commRank() == i)
			// 	{
			// 		printf("Rank %d has %d groups!\n",i,ggid);
			// 	}
			// }
			MPI_Allreduce(&ggid, &totalgroupnum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			LogMsg(VERB_PARANOID,"[GA2] Global number of groups is %d",totalgroupnum);
			
			LogOut("Found %d groups to sort between ranks\n",totalgroupnum);

			


			// for (int i = 1; i<commSize();i++)
			// {
			// 	if (commRank() == i)
			// 	{
			// 		/* scan the first slice and propose changes */

			// 	}
			// }

			/* print stuff */
			createMeas(afield, 17562);
			memcpy(afield->m2Start(),afield->m2hStart(),afield->Size()*afield->Precision());
			writeEDens (afield,MAP_M2S);
			destroyMeas();

		}

		return groupId;
	}


	bool Tracker::PatchGroups ()
	{
		LogOut("\nPatching groups ... ");

		// for (int ig = 0; ig < Halolist.size(); ig++)
		// {
		// 	LogOut("group %d\n",Halolist[ig]->gID());	
		// }

		//LogOut("done!\n");
		return true;
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


	/* Adds a group into the tracker */

	bool	Tracker::AddGroup (Group* newg)
	{
			Halolist.push_back(newg);
			/* Quick check */
			LogOut("group %d pushed %d points to halolist\n",Halolist.back()->gID(),Halolist.back()->NPoints());	
			return true;
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


	// template<typename Float>
	// int Tracker::SearchHalos()
	// {
	// 	/* Assumes density contrast in m2
	// 		TO INTRODUCE:
	// 		- halo_thr
	// 		- halom2 as a varibale of the tracjer Class
	// 	*/
	// 	Float *m2 = static_cast<Float*>(afield->m2Start());
	//
	// 	int halott = 0;
	// 	int halom2 = 0;
	// 	Float halo_thr = (Float) (en_th;); //halo_thr might be element of HaloInfo Struct
	//
	// 	const int nThreads = commThreads();
	//
	//
	// 	#pragma omp parallel for schedule(static) reduction (+:halott)
	// 	for (size_t iidx = 0 ; iidx < V; iidx++)
	// 	{
	// 		int max = false;
	// 		if ( std::abs(m2[iidx]) > halo_thr )
	// 			max = true;
	//
	// 		if (max && addifm)
	// 		{
	// 			size_t esta = iidx;
	// 			if (!afield->Folded())
	// 			{
	// 				/* fidx = iZ*n2 + iiy*shift*Lx +ix*shift +sy
	// 					idx  = iZ*n2 + [iy+sy*(n1/shift)]*Lx +ix
	// 					ix   = rix*ref , etc... */
	// 				size_t X[3];
	// 				indexXeon::idx2Vec (iidx, X, Lx);
	// 				size_t sy  = X[1]/(Lx/shift);
	// 				size_t iiy = X[1] - (sy*Lx/shift);
	// 				esta = X[2]*S + shift*(iiy*Lx+X[0]) +sy;
	// 			}
	//
	// 			halott++;
	//
	// 			if (max)
	// 				halom2++;
	//
	// 			bool bola = false;
	//
	// 			#pragma omp critical (writeaxiton)
	// 				bola = AddPoint(esta);
	//
	// 			if (bola)
	// 				halonew++;
	// 		}
	// 	}
	// 	LogMsg(VERB_PARANOID,"[AT] Search Halos returned %d (%d with m2 criterion) but only %d new",halott,halom2,halonew);
	// 	return axitm;
	//
	// }

	size_t Tracker::foldidx(size_t idx)
	{
	size_t X[3];
	indexXeon::idx2Vec (idx, X, Lx);
	size_t sy  = X[1]/(Lx/shift);
	size_t iiy = X[1] - (sy*Lx/shift);
	return X[2]*S + shift*(iiy*Lx+X[0]) +sy;
	}

	size_t Tracker::unfoldidx(size_t fidx)
	{
	size_t sy  = fidx % shift; // danger, assumes S is divisible/shift
	size_t iz  = fidx / S;
	size_t tem = (fidx - iz*S - sy)/shift;
	size_t iiy = tem/Lx;
	size_t ix  = tem % Lx;
	return iz*S + (iiy + sy*Lx/shift)*Lx + ix;
	}


#endif
