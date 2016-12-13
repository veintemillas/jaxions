#include "scalar/scalarField.h"

void	analyzePow (Scalar *axion, double *powera, int *nmodes, const int K3);

int	spectrumUNFOLDED(Scalar *field, void *spectrumK, void *spectrumG, void *spectrumV);
//int	spectrumUNFOLDED(Scalar *field, void *spectrumK, void *spectrumG, void *spectrumV);
void powerspectrumUNFOLDED(Scalar *axion, void *spectrumK, void *spectrumG, void *spectrumV, FlopCounter *fCount);