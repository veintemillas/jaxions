#include "scalarField.h"

#ifndef	_FOLDER_CLASS_
	#define _FOLDER_CLASS_

	class	Folder
	{
		private:

		size_t shift;
		size_t fSize;
		const int Lz;
		const int n1;
		const int n2;
		const int n3;

		Scalar *field;

		template<typename cFloat>
		void	foldField();

		template<typename cFloat>
		void	unfoldField();

		template<typename cFloat>
		void	unfoldField2D(size_t cZ);

		public:

			 Folder(Scalar *scalar);
			~Folder() {};

		void	operator()(FoldType fType, size_t cZ=0);
	};
#endif
