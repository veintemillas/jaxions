#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "energy/energy.h"
#include "utils/utils.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "strings/strings.h"


#include "meas/measa.h"
#include "WKB/WKB.h"

using namespace std;
using namespace AxionWKB;

int	main (int argc, char *argv[])
{
	Cosmos myCosmos = initAxions(argc, argv);

	commSync();

	Scalar *axion;

	MeasData lm;
	lm.str.strDen = 0 ;

	MeasInfo ninfa;
	ninfa.sliceprint = 0;
	ninfa.idxprint = 0 ;
	ninfa.index = 0;
	ninfa.measdata = MEAS_NOTHING;

	readConf(&myCosmos, &axion, fIndex);

	strings2	(axion);
	ninfa.measdata = MEAS_STRING | MEAS_STRINGCOO;
	lm = Measureme (axion, ninfa);

	delete axion;
	endAxions();

	exit(0);

}
