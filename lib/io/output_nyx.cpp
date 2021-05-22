/*

 Based on output_nyx.cc - part of MUSIC -
 a code to generate multi-scale initial conditions
 for cosmological simulations

 Copyright (C) 2010  Oliver Hahn
 Copyright (C) 2012  Jan Frederik Engels

 */

#ifdef USE_NYX_OUTPUT

#include <io/output_nyx.h>

#include <AMReX_VisMF.H>
#include <AMReX_Box.H>
#include <AMReX_RealBox.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_FabArray.H>
#include <AMReX_MultiFab.H>

#define MAX_GRID_SIZE	32
#define BL_SPACEDIM 3


  amrex::nyx_output_plugin::nyx_output_plugin( Scalar *axion, int index ) : faxion(axion)
	{
		int argc=0;
		char **argv;
    LogMsg(VERB_NORMAL, "[ONYXp] Init ARMeX");LogFlush();
		amrex::Initialize(argc,argv);

    LogMsg(VERB_NORMAL, "[ONYXp] Done!");LogFlush();
		bool bhave_hydro = false;


    /* Number and names of scalar fields to print */
	  n_data_items = 4;

		field_name.resize(n_data_items);
		field_name[0] = "cm_re";
		field_name[1] = "cm_im";
		field_name[2] = "cv_re";
		field_name[3] = "cv_im";
		the_sim_header.particle_idx = 0;

    /* There is only one refinement level in jaxions,
    so we require 1 Multifab in the Multifab vector mfs */
		f_lev = 0;
    mfs.resize(f_lev+1);

    /* what is this? */
		Vector<int> pmap(2);
		pmap[0]=0;
		pmap[1]=0;

		gridp = faxion->Length();

		int off[] = {0, 0, 0};


	  for(int lev = 0; lev <= f_lev; lev++)
    {

      //JAVI some stuff deleted
      int mlev = lev+levelmin_;
      int fac  = (1<<lev);

  		BoxArray   domainBoxArray(1);

  		IntVect    pdLo(0,0,0);
  		IntVect    pdHi(faxion->Length()-1,faxion->Length()-1,faxion->Depth()-1);

      /* Creates the problem domain */
  		Box probDomain(pdLo,pdHi);

  	  domainBoxArray.set(0, probDomain);
  		domainBoxArray.maxSize(32);
  		pmap.resize(domainBoxArray.size(),0);

      /* Creates the distribution mapping domain */
  		DistributionMapping domainDistMap(pmap);
  		boxarrays.push_back(domainBoxArray);
  		boxes.push_back(probDomain);

      /* ghost cells to save*/
  		int ngrow = 0;

      mfs[lev] = new MultiFab(domainBoxArray, domainDistMap, n_data_items, ngrow);
    }

		// bool haveblockingfactor		= cf.containsKey( "setup", "blocking_factor");
    // bool haveblockingfactor		= true;


    /* Header stuff */
		the_sim_header.dimensions.push_back( 1<<levelmin_ );
		the_sim_header.dimensions.push_back( 1<<levelmin_ );
		the_sim_header.dimensions.push_back( 1<<levelmin_ );

		the_sim_header.offset.push_back( 0 );
		the_sim_header.offset.push_back( 0 );
		the_sim_header.offset.push_back( 0 );

		the_sim_header.a_start	 = *faxion->RV();
		the_sim_header.dx	       = faxion->Delta(); // not sure?!?
		the_sim_header.boxlength = faxion->BckGnd()->PhysSize();
		the_sim_header.h0	       = 0.7;
		the_sim_header.omega_b	 = 0.0;

		the_sim_header.omega_m	 = 0.31;
		the_sim_header.omega_v	 = 0.69;
		the_sim_header.vfact	   = 1.0;   //.. need to multiply by h, nyx wants this factor for non h-1 units

    /* Here I initialise nonessentials */
    levelmin_ = 0;
    levelmax_ = 0;
    fname_    = "save" + std::to_string(index);

    LogMsg(VERB_NORMAL,"[ONYXp] Constructor done");LogFlush();
	}

	amrex::nyx_output_plugin::~nyx_output_plugin()
	{
    LogMsg(VERB_NORMAL,"[ONYXp] Create dirs");LogFlush();
		std::string FullPath = fname_;
		if (!UtilCreateDirectory(FullPath, 0755))
			CreateDirectoryFailed(FullPath);
		if (!FullPath.empty() && FullPath[FullPath.size()-1] != '/')
			FullPath += '/';
		FullPath += "Header";
		std::ofstream Header(FullPath.c_str());

		for(int lev=0; lev <= f_lev; lev++)
		{
			LogMsg(VERB_NORMAL,"[ONYXp] writeLevelPlotFile %d",lev);LogFlush();
			amrex::nyx_output_plugin::writeLevelPlotFile (	fname_,
						Header,
						VisMF::OneFilePerCPU,
						lev);
		}
		Header.close();

    LogMsg(VERB_NORMAL,"[ONYXp] WriteGridsFile");LogFlush();
		amrex::nyx_output_plugin::writeGridsFile(fname_);

    LogMsg(VERB_NORMAL,"[ONYXp] Finalize");LogFlush();
    amrex::Finalize();
	}

  void amrex::nyx_output_plugin::dump_grid_data(std::string fieldname, double factor, double add)
	{
		// std::cout << fieldname << " is dumped... to mf index " << comp << std::endl;

		//FIXME adapt for multiple levels!
      int mlevel = 0;
      int blevel = mlevel-levelmin_;

			std::vector<int> ng;
			// ng.push_back( gh.get_grid(mlevel)->size(0) );
			// ng.push_back( gh.get_grid(mlevel)->size(1) );
			// ng.push_back( gh.get_grid(mlevel)->size(2) );

			// std::cout << ng[0] << " " << ng[1] << " " << ng[2] << std::endl;

			//write data to mf
			// for(MFIter mfi(mfs); mfi.isValid(); ++mfi) {
      for(MFIter mfi(*(mfs[blevel])); mfi.isValid(); ++mfi) {
			  FArrayBox &myFab = (*(mfs[blevel]))[mfi];
			  const Box& box = mfi.validbox();
			  const int  *fab_lo = box.loVect();
			  const int  *fab_hi = box.hiVect();

        if (faxion->Precision() == FIELD_SINGLE)
        {
          #pragma omp parallel for default(shared)
			    for (int k = fab_lo[2]; k <= fab_hi[2]; k++) {
			      for (int j = fab_lo[1]; j <= fab_hi[1]; j++) {
			    	  for (int i = fab_lo[0]; i <= fab_hi[0]; i++) {

				  IntVect iv(i,j,k);
				  int idx = myFab.box().index(iv);
          size_t fidx = faxion->Surf()*k+faxion->Length()*j + i;

          /* cm_re, cm_im cv_re, cv_im */
          // LogOut("g ~ %d %d %d -> %d %f\n",k,j,i,fidx,static_cast<float*>(faxion->mStart())[fidx*2]);
          myFab.dataPtr(0)[idx] = static_cast<float*>(faxion->mStart())[fidx*2];
          myFab.dataPtr(1)[idx] = static_cast<float*>(faxion->mStart())[fidx*2+1];
          myFab.dataPtr(2)[idx] = static_cast<float*>(faxion->vStart())[fidx*2];
          myFab.dataPtr(3)[idx] = static_cast<float*>(faxion->vStart())[fidx*2+1];
          }}}
        }
        else
        {
          #pragma omp parallel for default(shared)
          for (int k = fab_lo[2]; k <= fab_hi[2]; k++) {
            for (int j = fab_lo[1]; j <= fab_hi[1]; j++) {
              for (int i = fab_lo[0]; i <= fab_hi[0]; i++) {

          IntVect iv(i,j,k);
          int idx = myFab.box().index(iv);
          size_t fidx = faxion->Surf()*k+faxion->Length()*j + i;

          /* cm_re, cm_im cv_re, cv_im */
          myFab.dataPtr(0)[idx] = static_cast<double*>(faxion->mStart())[fidx*2];
          myFab.dataPtr(1)[idx] = static_cast<double*>(faxion->mStart())[fidx*2+1];
          myFab.dataPtr(2)[idx] = static_cast<double*>(faxion->vStart())[fidx*2];
          myFab.dataPtr(3)[idx] = static_cast<double*>(faxion->vStart())[fidx*2+1];
          }}}
        }

			} // MFI


	//	char nyxname[256], filename[256];
	//
	//	for(unsigned ilevel=levelmin_; ilevel<=levelmax_; ++ilevel )
	//	{
	}

	void amrex::nyx_output_plugin::finalize( void )
	{
		//
		//before finalizing we write out an inputs and a probin file for Nyx.
		//
		std::ofstream inputs("inputs");
		std::ofstream probin("probin");

		//at first the fortran stuff...
		probin << "&fortin" << std::endl;
		probin << "  comoving_OmM = " << the_sim_header.omega_m << "d0" << std::endl;
		probin << "  comoving_OmB = " << the_sim_header.omega_b << "d0" << std::endl;
		probin << "  comoving_OmL = " << the_sim_header.omega_v << "d0" << std::endl;
		probin << "  comoving_h   = " << the_sim_header.h0      << "d0" << std::endl;
		probin << "/" << std::endl;
		probin << std::endl;

		//afterwards the cpp stuff...(for which we will need a template, which is read in by the code...)
		inputs << "nyx.final_a = 1.0 " << std::endl;
		inputs << "max_step = 100000 " << std::endl;
		inputs << "nyx.small_dens = 1e-4" << std::endl;
		inputs << "nyx.small_temp = 10" << std::endl;
		inputs << "nyx.cfl            = 0.9     # cfl number for hyperbolic system" << std::endl;
		inputs << "nyx.init_shrink    = 1.0     # scale back initial timestep" << std::endl;
		inputs << "nyx.change_max     = 1.05    # scale back initial timestep" << std::endl;
		inputs << "nyx.dt_cutoff      = 5.e-20  # level 0 timestep below which we halt" << std::endl;
		inputs << "nyx.sum_interval   = 1      # timesteps between computing mass" << std::endl;
		inputs << "nyx.v              = 1       # verbosity in Castro.cpp" << std::endl;
		inputs << "gravity.v             = 1       # verbosity in Gravity.cpp" << std::endl;
		inputs << "amr.v                 = 1       # verbosity in Amr.cpp" << std::endl;
		inputs << "mg.v                  = 0       # verbosity in Amr.cpp" << std::endl;
		inputs << "particles.v           = 1       # verbosity in Particle class" << std::endl;
		inputs << "amr.ref_ratio       = 2 2 2 2 2 2 2 2 " << std::endl;
		inputs << "amr.regrid_int      = 2 2 2 2 2 2 2 2 " << std::endl;
		inputs << "amr.initial_grid_file = init/grids_file" << std::endl;
		inputs << "amr.useFixedCoarseGrids = 1" << std::endl;
		inputs << "amr.check_file      = chk " << std::endl;
		inputs << "amr.check_int       = 10 " << std::endl;
		inputs << "amr.plot_file       = plt " << std::endl;
		inputs << "amr.plot_int        = 10 " << std::endl;
		inputs << "amr.derive_plot_vars = particle_count particle_mass_density pressure" << std::endl;
		inputs << "amr.plot_vars = ALL" << std::endl;
		inputs << "nyx.add_ext_src = 0" << std::endl;
		inputs << "gravity.gravity_type = PoissonGrav    " << std::endl;
		inputs << "gravity.no_sync      = 1              " << std::endl;
		inputs << "gravity.no_composite = 1              " << std::endl;
		inputs << "mg.bottom_solver = 1                  " << std::endl;
		inputs << "geometry.is_periodic =  1     1     1 " << std::endl;
		inputs << "geometry.coord_sys   =  0             " << std::endl;
		inputs << "amr.max_grid_size    = 32             " << std::endl;
		inputs << "nyx.lo_bc       =  0   0   0          " << std::endl;
		inputs << "nyx.hi_bc       =  0   0   0          " << std::endl;
		inputs << "nyx.do_grav  = 1                      " << std::endl;
		inputs << "nyx.do_dm_particles = 1               " << std::endl;
		inputs << "nyx.particle_init_type = Cosmological " << std::endl;
    inputs << "nyx.print_fortran_warnings = 0        " << std::endl;
		inputs << "cosmo.initDirName  = init             " << std::endl;
		inputs << "nyx.particle_move_type = Gravitational" << std::endl;
		inputs << "amr.probin_file = probin              " << std::endl;
		inputs << "cosmo.ic-source = MUSIC               " << std::endl;


		// inputs << "amr.blocking_factor = " << cf_.getValue<double>("setup","blocking_factor") << std::endl;
                inputs << "amr.blocking_factor = " << true;

		inputs << "nyx.do_hydro = "<< (the_sim_header.omega_b>0?1:0) << std::endl;
		inputs << "amr.max_level       = " << levelmax_-levelmin_ << std::endl;
		inputs << "nyx.initial_z = " << 1/the_sim_header.a_start-1 << std::endl;
		// inputs << "amr.n_cell           = " << sizex_[0] << " " << sizey_[0] << " " << sizez_[0] << std::endl;
		// inputs << "nyx.n_particles      = " << sizex_[0] << " " << sizey_[0] << " " << sizez_[0] << std::endl;
    inputs << "amr.n_cell           = 0 0 0" << std::endl;
		inputs << "nyx.n_particles      = 0 0 0" << std::endl;

		inputs << "geometry.prob_lo     = 0 0 0" << std::endl;

		//double dx = the_sim_header.dx/the_sim_header.h0;
		double bl = the_sim_header.boxlength/the_sim_header.h0;
		inputs << "geometry.prob_hi     = " << bl << " " << bl << " " << bl << std::endl;



		probin.close();
		inputs.close();
		// std::cout << "finalizing..." << std::endl;

	}

	void amrex::nyx_output_plugin::writeLevelPlotFile (const std::string& dir, std::ostream& os, VisMF::How	how, int level)
	{
		int i, n;

		const Real cur_time = 0.0;

		// std::cout << "in writeLevelPlotFile" << std::endl;
		double h0 = 1.0;

		//
		// The first thing we write out is the plotfile type.
		//
		os << "ARMeX_Jaxions_output" << '\n';

		os << n_data_items << '\n';

		for (i = 0; i < n_data_items; i++)
			os << field_name[i] << '\n';

		os << 3 << '\n';
		os << 0 << '\n';

		os << f_lev << '\n';

		for (i = 0; i < BL_SPACEDIM; i++)
			os << 0 << ' '; //ProbLo
		os << '\n';
		double boxlength  = faxion->BckGnd()->PhysSize();
		for (i = 0; i < BL_SPACEDIM; i++)
			os << boxlength/h0 << ' '; //ProbHi
		os << '\n';

		for (i = 0; i < f_lev; i++)
			os << 2 << ' '; //refinement factor
		os << '\n';

		IntVect    pdLo(0,0,0);
		IntVect    pdHi(gridp-1,gridp-1,gridp-1);
//			Box        probDomain(pdLo,pdHi);
		for (i = 0; i <= f_lev; i++) //Geom(i).Domain()
		{
//				IntVect    pdLo(offx_[i], offy_[i], offz_[i]);
//				IntVect    pdHi(offx_[i]+sizex_[i], offy_[i]+sizey_[i], offz_[i]+sizez_[i]);
			Box        probDomain(pdLo,pdHi);
			os << probDomain << ' ';
			pdHi *= 2;
			pdHi += 1;
		}
		os << '\n';

		for (i = 0; i <= f_lev; i++) //level steps
			os << 0 << ' ';
		os << '\n';

		double dx = faxion->Delta();
		for (i = 0; i <= f_lev; i++)
		{
			for (int k = 0; k < BL_SPACEDIM; k++)
				os << dx << ' ';
			os << '\n';
			dx = dx/2.;
		}
		os << 0 << '\n';
		os << "0\n"; // Write bndry data.


		//
		// Build the directory to hold the MultiFab at this level.
		// The name is relative to the directory containing the Header file.
		//
		static const std::string BaseName = "/Cell";

		std::string Level = Concatenate("Level_", 0, 1);
		//
		// Now for the full pathname of that directory.
		//
		std::string FullPath = dir;
		if (!FullPath.empty() && FullPath[FullPath.size()-1] != '/')
			FullPath += '/';
		FullPath += Level;
		//
		// Only the I/O processor makes the directory if it doesn't already exist.
		//
		if (!UtilCreateDirectory(FullPath, 0755))
			CreateDirectoryFailed(FullPath);

		os << 0 << ' ' << boxarrays[0].size() << ' ' << 0 << '\n';
		os << 0 << '\n';

		double cellsize[3];
		dx = faxion->BckGnd()->PhysSize()/gridp/h0;
		for (n = 0; n < BL_SPACEDIM; n++)
		{
			cellsize[n] = dx;
		}
		for (i = 0; i < 0; i++)
		{
			for (n = 0; n < BL_SPACEDIM; n++)
			{
				cellsize[n] /= 2.;
			}
		}
		// std::cout << cellsize[0] << std::endl;
		for (i = 0; i < boxarrays[0].size(); ++i)
		{
			double problo[] = {0,0,0};
			// std::cout << boxarrays[0][i] << std::endl;
			RealBox gridloc = RealBox(boxarrays[0][i], cellsize, problo);
			for (n = 0; n < BL_SPACEDIM; n++)
				os << gridloc.lo(n) << ' ' << gridloc.hi(n) << '\n';
		}
		//
		// The full relative pathname of the MultiFabs at this level.
		// The name is relative to the Header file containing this name.
		// It's the name that gets written into the Header.
		//
		std::string PathNameInHeader = Level;
		PathNameInHeader += BaseName;
		os << PathNameInHeader << '\n';

		//
		// Use the Full pathname when naming the MultiFab.
		//
    amrex::nyx_output_plugin::dump_grid_data("dummy", 1.0,1.0);
		std::string TheFullPath = FullPath;
		TheFullPath += BaseName;
		VisMF::Write(*mfs[level],TheFullPath,how,true);
	}

	void amrex::nyx_output_plugin::writeGridsFile (const	std::string&	dir)
	{
		int i, n;

		std::string myFname = dir;
		if (!myFname.empty() && myFname[myFname.size()-1] != '/')
			myFname += '/';
		myFname += "grids_file";

		std::ofstream os(myFname.c_str());

		os << f_lev << '\n';

		for (int lev = 1; lev <= f_lev; lev++)
		{
			os << boxarrays[lev].size() << '\n';
			boxarrays[lev].coarsen(2);
			for (i=0; i < boxarrays[lev].size(); i++)
				os << boxarrays[lev][i] << "\n";
		}
		os.close();
	}

#endif
