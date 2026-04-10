#ifndef	_FOLDER_CLASS_
	#define _FOLDER_CLASS_

	#include "scalarField.h"
	#include "utils/tunable.h"

	class	Folder : public Tunable
	{
		private:

		Scalar *field;

		const size_t Nz;
		const size_t Nx;
		const size_t Ny;
		const size_t Nxy;
		const size_t Nxyz;

		size_t shift;
		size_t fSize;

		bool somethingdone;

		template<typename cFloat>
		void	foldField();

		template<typename cFloat>
		void	unfoldField();

		template<typename cFloat>
		void	unfoldField2D(const size_t cZ);

		template<typename cFloat>
		void	unfoldField2DYZ(const size_t sX);

		template<typename cFloat>
		void	unfoldField2DXZ (const size_t sY);

		template<typename cFloat>
		void	foldM2();

		template<typename cFloat>
		void	unfoldM2();

		template<typename cFloat>
		void	unfoldM22D(const size_t cZ);

		public:

			 Folder(Scalar *scalar);
			~Folder() {};

		void	operator()(FoldType fType, size_t cZ=0);
	};
#endif
