class	Smoother
{
	private:

	const double alpha;
	const int iter;
	const int Lx, Lz, V, Sf;

	FieldPrecision prec;

	Scalar	*axionField;

	int bytes;

	public:

		 Smoother(Scalar *field, const int iter, const double alpha);
		~Smoother() {};

	void	operator();
}

	Smoother::Smoother(Scalar *field, const int iter, const double alpha) : axionField(field), alpha(alpha), iter(iter), Lx(axionField->Length()), Lz(axionField->Depth()),
									       V(axionField->Size()), Sf(axionField->Surf()), prec(axionField->Precision())
{
	int bytes = 2*axionField->dataSize()*axionField->eSize();
}

void	Smoother::operator()
{
	#define BLSIZE 256
	dim3	gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz,1);
	dim3	blockSize(BLSIZE,1,1);

	if (prec == FIELD_DOUBLE)
	{
		complex<double> *mIn, *mOut, *tmp;

		mIn  = ((complex<double> *) axionField->mGpu()  + Sf;
		mOut =  (complex<double> *) axionField->m2Gpu() + Sf

		axionField->exchangeGhosts(FIELD_M);

		for (int it=0; it<iter; it++)
		{
			smoothGpuKernel_d<<gridSize,blockSize>>(mIn, mOut, alpha, Lx, Sf, V);

			if (it & 1)
				exchangeGhosts(FIELD_M);
			else
				exchangeGhosts(FIELD_M2);

			/*	Swap mIn and mOut	*/

			tmp = mIn;
			mIn = mOut;
			mOut = tmp;
		}
	} else if (prec == FIELD_SINGLE) {
		complex<float> *mIn, *mOut, *tmp;

		mIn  = ((complex<float> *) axionField->mGpu()  + Sf;
		mOut =  (complex<float> *) axionField->m2Gpu() + Sf

		axionField->exchangeGhosts(FIELD_M);

		for (int it=0; it<iter; it++)
		{
			smoothGpuKernel_s<<gridSize,blockSize>>(mIn, mOut, alpha, Lx, Sf, V);

			if (it & 1)
				exchangeGhosts(FIELD_M);
			else
				exchangeGhosts(FIELD_M2);

			/*	Swap mIn and mOut	*/

			tmp = mIn;
			mIn = mOut;
			mOut = tmp;
		}
	}	

	if (iter & 1)		// With an odd number of iterations, the output field is in m2
		cudaMemcpy(m, m2, bytes, cudaMemcpyDeviceToDevice);
}

void	smoothConfGpu	(Scalar *field, const int iter, const double alpha)
{
	Smoother *smooth = new Smoother(field, iter, alpha);

	smooth();

	delete smooth;

	return;
}
