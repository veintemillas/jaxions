#ifndef	_FOLDER_CLASS_
	#define _FOLDER_CLASS_

	#include "scalarField.h"
	#include "utils/tunable.h"

	class	Folder : public Tunable
	{
		private:

		Scalar *field;

		const size_t Lz;
		const size_t n1;
		const size_t n2;
		const size_t n3;

		size_t shift;
		size_t fSize;

		template<typename cFloat>
		void	foldField();

		template<typename cFloat>
		void	unfoldField();

		template<typename cFloat>
		void	unfoldField2D(const size_t cZ);

		template<typename cFloat>
		void	unfoldField2DYZ(const size_t sX);

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
