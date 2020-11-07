#ifdef USE_NYX_OUTPUT

	#include "scalar/scalarField.h"

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


	namespace amrex {

	class nyx_output_plugin
	{
		protected:

		struct patch_header{
			int component_rank;
			size_t component_size;
			std::vector<int> dimensions;
			int rank;
			std::vector<int> top_grid_dims;
			std::vector<int> top_grid_end;
			std::vector<int> top_grid_start;
		};

		struct sim_header{
			std::vector<int> dimensions;
			std::vector<int> offset;
			float a_start;
			float dx;
			float h0;
			float omega_b;
			float omega_m;
			float omega_v;
			float vfact;
			float boxlength;
			int   particle_idx;
		};

	//	struct grid_on_one_level{
	//		IntVect lo;
	//		IntVect hi;
	//	};

			int n_data_items;
			std::vector<std::string> field_name;
			int f_lev;
			int gridp;

			std::vector<MultiFab*> mfs;

		//	std::vector<grid_on_one_level> grids;

			std::vector<BoxArray> boxarrays;
			std::vector<Box> boxes;

			sim_header the_sim_header;

			void dump_grid_data(std::string fieldname, double factor = 1.0, double add = 0.0 );

			/* JAVI here I write all MUSIC variables which are irrelevant
			for our output but track the structure of the boxlib file format
			and concept */
			int levelmin_;
			int levelmax_;
			std::string fname_ = "save";

			/* JAVI And here my scalar */
			Scalar *faxion;


			public:

			nyx_output_plugin( Scalar *axion ) ;

			~nyx_output_plugin();

			void finalize( void );

			void writeLevelPlotFile (const std::string& dir, std::ostream& os, VisMF::How	how, int level);

			void writeGridsFile (const	std::string&	dir);

	};

	}
#endif
