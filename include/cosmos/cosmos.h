
#ifndef	_COSMOS_CLASS_
	#define	_COSMOS_CLASS_

	#include"enum-field.h"

	class	Cosmos
	{
		private:

//		size_t	 nSize;

		double	 lSize;

		double	 lambda;

		VqcdType pot;

		double	 indi3;
		double	 nQcd;
		double	 zThRes;
		double	 zRestore;

		double	 biasV;
		double	 biasP;

		double	 gamma;


		double	 frw;
		bool	 mink;

		public:

			 //Cosmos() : nSize(0), lSize(0.0), lambda(-1.e8), msa(-1.e8), indi3(-1.e8), gamma(-1.e8), nQcd(-1.e8), pot(VQCD_NONE), zThRes(-1.e8), zRestore(-1.e8) {}
			 Cosmos() : lSize(0.0), lambda(-1.e8), indi3(-1.e8), gamma(-1.e8), nQcd(-1.e8), biasV(0.0), biasP(0.0), zThRes(-1.e8), zRestore(-1.e8), pot(VQCD_NONE), frw(1.0), mink(false) {}

		double&   PhysSize()	{ return lSize;    }

		double&   Lambda  ()	{ return lambda;   }

		VqcdType& QcdPot  ()	{ return pot;      }

		double&	  Indi3   ()	{ return indi3;    }
		double&   QcdExp  ()	{ return nQcd;     }
		double&   ZThRes  ()	{ return zThRes;   }
		double&   ZRestore()	{ return zRestore; }

		double&   BiasV   ()	{ return biasV;    }
		double&   BiasP   ()	{ return biasP;    }

		double&   Gamma   ()	{ return gamma;    }

		double&   Frw     ()	{ return frw;      }
		bool&     Mink    ()	{ return mink;     }

		void     SetPhysSize(const double mSize){ lSize    = mSize; }
		void     SetLambda  (const double nLmda){ lambda   = nLmda; } //msa     = sqrt(2.*nLmda)*lSize/((double) nSize); }

		void     SetQcdPot  (const VqcdType pt)	{ pot      = pt;    }

		void     SetIndi3   (const double nI3)  { indi3    = nI3;   }
		void     SetQcdExp  (const double qExp)	{ nQcd     = qExp;  }
		void     SetZThRes  (const double newZT){ zThRes   = newZT; }
		void     SetZRestore(const double newZR){ zRestore = newZR; }

		void     SetBiasV   (const double newBiasV){ biasV    = newBiasV; }
		void     SetBiasP   (const double newBiasP){ biasP    = newBiasP; }

		void     SetGamma   (const double nGmma){ gamma    = nGmma; }

		void     SetFrw  (const double fff)	{ frw     = fff;  }
		void     SetMink  (const bool bbb)	{ mink    = bbb;  }

//		double	 AxionMass  (const double z);
//		double	 AxionMassSq(const double z);
//		double	 SaxionShift(const double z);
//		double	 Saskia	    (const double z);

//		double	 dzSize	    (const double z, const FieldType fType);

//		double	 rsvPQ2	    (const double z) { auto z2 = z*z; auto z3 = z2*z; auto z4 = z2*z2;
//						       return (0.125*z + 0.30676113886283973*z2 + 0.20762392505082639*z3 + 0.03303541390146716*z4)/
//						       (1 + 3.0165891109027165*z + 2.857822775289389*z2 + 0.8969613324856603*z3 + 0.05640260585369341*z4); }
	};
#endif
