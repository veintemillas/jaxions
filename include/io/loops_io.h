#pragma once
#include "enum-field.h"   // IOParms definition
#include "scalar/scalarField.h"
#include <cstddef>

void writeStringLoopObservables(Scalar* axion,
                                StringLoopParms slp,
                                int rango,
                                IOParms& iop);

void	writeStringLabelMap (Scalar *axion, IOParms& iop);
