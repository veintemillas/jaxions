#ifndef	__TABLEINTRINSICS
#define	__TABLEINTRINSICS

#include<cmath>
#include<array>

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>

#ifdef	__AVX512F__
	#define _MData_ __m512d
	#define	_MInt_  __m512i
	#define	_MHnt_  __m256i
#elif   defined(__AVX__)
	#define _MData_ __m256d
	#define	_MInt_  __m256i
	#define	_MHnt_  __m128i
#else
	#define _MData_ __m128d
	#define	_MInt_  __m128i
#endif

#if	defined(__AVX512F__)
	#define	_PREFIX_ _mm512
	#define	_PREFXL_ _mm256
	#define opCodl(x,...) opCode_N(_PREFXL_, x, __VA_ARGS__)
#else
	#if not defined(__AVX__) and not defined(__AVX2__)
		#define	_PREFIX_ _mm
	#else
		#define	_PREFIX_ _mm256
		#define	_PREFXL_ _mm
		#define opCodl(x,...) opCode_N(_PREFXL_, x, __VA_ARGS__)
	#endif
#endif

#define	M_PI2	(M_PI *M_PI)
#define	M_PI4	(M_PI2*M_PI2)
#define	M_PI6	(M_PI4*M_PI2)
/* Only single precision for the moment */

constexpr size_t iSgAd = 0b1000000000000000000000000000000000000000000000000000000000000000;
constexpr size_t iSgAb = 0b1000000000000000000000000000000010000000000000000000000000000000;
#ifdef	__AVX512F__
constexpr _MInt_  iSgnAbsd  = {       iSgAd,       iSgAd,       iSgAd,       iSgAd,       iSgAd,       iSgAd,       iSgAd,       iSgAd };
constexpr _MInt_  iSgnAbsf  = {       iSgAb,       iSgAb,       iSgAb,       iSgAb,       iSgAb,       iSgAb,       iSgAb,       iSgAb };
#elif   defined(__AVX__)
constexpr _MInt_  iSgnAbsd  = {       iSgAd,       iSgAd,       iSgAd,       iSgAd };
constexpr _MInt_  iSgnAbsf  = {       iSgAb,       iSgAb,       iSgAb,       iSgAb };
#else
constexpr _MInt_  iSgnAbsd  = {       iSgAd,       iSgAd };
constexpr _MInt_  iSgnAbsf  = {       iSgAb,       iSgAb };
#endif

#ifndef	__INTEL_COMPILER

constexpr double Inf_d = __builtin_inf();
constexpr double Nan_d = __builtin_nan("");//0xFFFFF");

constexpr size_t m32Mk = 0b0000000000000000000000000001111100000000000000000000000000011111;
constexpr size_t iSgMk = 0b1100000000000000000000000000000011000000000000000000000000000000;
constexpr size_t iFlMk = 0b1111100000000000000000000000000011111000000000000000000000000000;

/*	For the exponential	*/
constexpr double th1_d  = +1.80911414126145723458e+03;
constexpr double th2_d  = +5.55111512312578270212e-17;
constexpr double ivL_d  = +4.61662413084468283841e+01;
constexpr double L1_d   = +2.16608493901730980724e-02;
constexpr double L2_d   = +2.32519284687887401481e-12;
constexpr double A1_d   = +5.00000000000000000000e-01;
constexpr double A2_d   = +1.66666666665260865265e-01;
constexpr double A3_d   = +4.16666666662260792853e-02;
constexpr double A4_d   = +8.33336798434219580556e-03;
constexpr double A5_d   = +1.38889490863777190395e-03;
constexpr double B1_d   = -5.00000000000000111022e-01;
constexpr double B2_d   = +3.33333333316142621516e-01;
constexpr double B3_d   = -2.49999999962519925401e-01;
constexpr double B4_d   = +2.00003251399606729599e-01;
constexpr double B5_d   = -1.66670608463718505909e-01;

constexpr double Sl00_d = 1.00000000000000000000e+00;
constexpr double Sl01_d = 1.02189714865410508082e+00;
constexpr double Sl02_d = 1.04427378242741042413e+00;
constexpr double Sl03_d = 1.06714040067681992241e+00;
constexpr double Sl04_d = 1.09050773266524458904e+00;
constexpr double Sl05_d = 1.11438674259588310633e+00;
constexpr double Sl06_d = 1.13878863475667913008e+00;
constexpr double Sl07_d = 1.16372485877757014805e+00;
constexpr double Sl08_d = 1.18920711500271636396e+00;
constexpr double Sl09_d = 1.21524735998046651275e+00;
constexpr double Sl10_d = 1.24185781207347645250e+00;
constexpr double Sl11_d = 1.26905095719172322788e+00;
constexpr double Sl12_d = 1.29683955465100098081e+00;
constexpr double Sl13_d = 1.32523664315974087913e+00;
constexpr double Sl14_d = 1.35425554693688354746e+00;
constexpr double Sl15_d = 1.38390988196383091235e+00;
constexpr double Sl16_d = 1.41421356237309225889e+00;
constexpr double Sl17_d = 1.44518080697703510396e+00;
constexpr double Sl18_d = 1.47682614593949779191e+00;
constexpr double Sl19_d = 1.50916442759341862256e+00;
constexpr double Sl20_d = 1.54221082540793474891e+00;
constexpr double Sl21_d = 1.57598084510787828094e+00;
constexpr double Sl22_d = 1.61049033194925073076e+00;
constexpr double Sl23_d = 1.64575547815395850648e+00;
constexpr double Sl24_d = 1.68179283050741901206e+00;
constexpr double Sl25_d = 1.71861929812247637983e+00;
constexpr double Sl26_d = 1.75625216037329323626e+00;
constexpr double Sl27_d = 1.79470907500309806437e+00;
constexpr double Sl28_d = 1.83400808640934087634e+00;
constexpr double Sl29_d = 1.87416763411029307917e+00;
constexpr double Sl30_d = 1.91520656139714162691e+00;
constexpr double Sl31_d = 1.95714412417540017941e+00;

constexpr double St00_d = 0.00000000000000000000e+00;
constexpr double St01_d = 1.15974117063913618369e-14;
constexpr double St02_d = 3.41618797093084913461e-15;
constexpr double St03_d = 3.69575974405711634226e-15;
constexpr double St04_d = 1.30701638697787231928e-14;
constexpr double St05_d = 9.42997619141976990955e-15;
constexpr double St06_d = 1.25236260025620074908e-14;
constexpr double St07_d = 7.36576401089527357793e-15;
constexpr double St08_d = 4.70275685574031410345e-15;
constexpr double St09_d = 2.36536434724852953227e-15;
constexpr double St10_d = 7.59609684336943426188e-15;
constexpr double St11_d = 9.99467515375775096979e-15;
constexpr double St12_d = 8.68512209487110863385e-15;
constexpr double St13_d = 4.15501897749673983948e-16;
constexpr double St14_d = 9.18083828572431297142e-15;
constexpr double St15_d = 1.04251790803720876383e-15;
constexpr double St16_d = 2.78990693089087774733e-15;
constexpr double St17_d = 1.15160818747516875124e-14;
constexpr double St18_d = 1.51947228890629129108e-15;
constexpr double St19_d = 4.11720196080016564552e-15;
constexpr double St20_d = 6.07470268107282183500e-15;
constexpr double St21_d = 8.20551346575487959595e-15;
constexpr double St22_d = 3.57742087137029902059e-15;
constexpr double St23_d = 6.33803674368915982631e-15;
constexpr double St24_d = 1.00739973218322238167e-14;
constexpr double St25_d = 1.53579843029258803130e-15;
constexpr double St26_d = 6.24685034485536557515e-15;
constexpr double St27_d = 9.12205626035419583226e-15;
constexpr double St28_d = 1.58714330671767538549e-15;
constexpr double St29_d = 6.82215511854592947014e-15;
constexpr double St30_d = 5.66696026748885461802e-15;
constexpr double St31_d = 8.96076779103666776760e-17;

constexpr double Cl000_d = 0.00000000000000000000e+00;	constexpr double Ct000_d = +0.00000000000000000000e+00;
constexpr double Cl001_d = 7.78214044294145423919e-03;	constexpr double Ct001_d = -8.86505291613039810711e-13;
constexpr double Cl002_d = 1.55041865364182740450e-02;	constexpr double Ct002_d = -4.53019894106141806506e-13;
constexpr double Cl003_d = 2.31670592820591991767e-02;	constexpr double Ct003_d = -5.24820948350867948207e-13;
constexpr double Cl004_d = 3.07716586667083902285e-02;	constexpr double Ct004_d = +4.52981429492974041473e-14;
constexpr double Cl005_d = 3.83188643027096986771e-02;	constexpr double Ct005_d = -5.73099483451230962139e-13;
constexpr double Cl006_d = 4.58095360318111488596e-02;	constexpr double Ct006_d = -5.16945691711023086468e-13;
constexpr double Cl007_d = 5.32445145181554835290e-02;	constexpr double Ct007_d = +6.56799335513498505623e-13;
constexpr double Cl008_d = 6.06246218158048577607e-02;	constexpr double Ct008_d = +6.29984820278532697824e-13;
constexpr double Cl009_d = 6.79506619089806918055e-02;	constexpr double Ct009_d = -4.72942411416325120266e-13;
constexpr double Cl010_d = 7.52234212377516087145e-02;	constexpr double Ct010_d = -1.64083014482421102320e-13;
constexpr double Cl011_d = 8.24436692109884461388e-02;	constexpr double Ct011_d = +8.61451277607734600128e-14;
constexpr double Cl012_d = 8.96121586902154376730e-02;	constexpr double Ct012_d = -5.28305054055006745317e-13;
constexpr double Cl013_d = 9.67296264589094789699e-02;	constexpr double Ct013_d = -3.58366676991069693647e-13;
constexpr double Cl014_d = 1.03796793680885457434e-01;	constexpr double Ct014_d = +7.58107396574714959003e-13;
constexpr double Cl015_d = 1.10814366340491687879e-01;	constexpr double Ct015_d = -2.01573682062378828350e-13;
constexpr double Cl016_d = 1.17783035655520507134e-01;	constexpr double Ct016_d = +8.62947402066865237868e-13;
constexpr double Cl017_d = 1.24703478501032805070e-01;	constexpr double Ct017_d = -7.55692031303642486328e-14;
constexpr double Cl018_d = 1.31576357789526809938e-01;	constexpr double Ct018_d = -8.07537345911928106101e-13;
constexpr double Cl019_d = 1.38402322858382831328e-01;	constexpr double Ct019_d = +7.36304355160311008177e-13;
constexpr double Cl020_d = 1.45182009844575077295e-01;	constexpr double Ct020_d = -7.71800158505531186393e-14;
constexpr double Cl021_d = 1.51916042026641662233e-01;	constexpr double Ct021_d = -7.99687152976992499109e-13;
constexpr double Cl022_d = 1.58605030175749561749e-01;	constexpr double Ct022_d = +8.89022349118660792922e-13;
constexpr double Cl023_d = 1.65249572895845631137e-01;	constexpr double Ct023_d = -5.38468263575932193366e-13;
constexpr double Cl024_d = 1.71850256927427835763e-01;	constexpr double Ct024_d = -7.68613417270053167485e-13;
constexpr double Cl025_d = 1.78407657473144354299e-01;	constexpr double Ct025_d = -3.26057174424194962103e-13;
constexpr double Cl026_d = 1.84922338494288851507e-01;	constexpr double Ct026_d = -2.76858844552342608925e-13;
constexpr double Cl027_d = 1.91394853000019793399e-01;	constexpr double Ct027_d = -3.90338789394394092280e-13;
constexpr double Cl028_d = 1.97825743329303804785e-01;	constexpr double Ct028_d = +6.16075577055189005371e-13;
constexpr double Cl029_d = 2.04215541429221048020e-01;	constexpr double Ct029_d = -5.30156512223642351067e-13;
constexpr double Cl030_d = 2.10564769107804750092e-01;	constexpr double Ct030_d = -4.55112415310467144369e-13;
constexpr double Cl031_d = 2.16873938301432644948e-01;	constexpr double Ct031_d = -8.18285326650847189001e-13;
constexpr double Cl032_d = 2.23143551314933574758e-01;	constexpr double Ct032_d = -7.23818994650092528698e-13;
constexpr double Cl033_d = 2.29374101065332069993e-01;	constexpr double Ct033_d = -4.86239995207797537358e-13;
constexpr double Cl034_d = 2.35566071312860003673e-01;	constexpr double Ct034_d = -9.30945858595988440243e-14;
constexpr double Cl035_d = 2.41719936886511277407e-01;	constexpr double Ct035_d = +6.33890739920074663960e-13;
constexpr double Cl036_d = 2.47836163904139539227e-01;	constexpr double Ct036_d = +4.41717558255332043426e-13;
constexpr double Cl037_d = 2.53915209981641964987e-01;	constexpr double Ct037_d = -6.78520851700795124550e-13;
constexpr double Cl038_d = 2.59957524436686071567e-01;	constexpr double Ct038_d = +2.39995415034222059347e-13;
constexpr double Cl039_d = 2.65963548497893498279e-01;	constexpr double Ct039_d = -7.55556939712950792476e-13;
constexpr double Cl040_d = 2.71933715483100968413e-01;	constexpr double Ct040_d = +5.40790421341488802831e-13;
constexpr double Cl041_d = 2.77868451003087102436e-01;	constexpr double Ct041_d = +3.69203745267249194839e-13;
constexpr double Cl042_d = 2.83768173130738432519e-01;	constexpr double Ct042_d = -9.38341743715598308739e-14;
constexpr double Cl043_d = 2.89633292582948342897e-01;	constexpr double Ct043_d = +9.43339915730756484180e-14;
constexpr double Cl044_d = 2.95464212893421063200e-01;	constexpr double Ct044_d = +4.14813203317848655516e-13;
constexpr double Cl045_d = 3.01261330578199704178e-01;	constexpr double Ct045_d = -3.79231693790893498175e-14;
constexpr double Cl046_d = 3.07025035295737325214e-01;	constexpr double Ct046_d = -8.25463124503461598280e-13;
constexpr double Cl047_d = 3.12755710003330023028e-01;	constexpr double Ct047_d = +5.66865375169547935919e-13;
constexpr double Cl048_d = 3.18453731119006988592e-01;	constexpr double Ct048_d = -4.72372791923691970339e-13;
constexpr double Cl049_d = 3.24119468654316733591e-01;	constexpr double Ct049_d = -1.04757484154296975909e-13;
constexpr double Cl050_d = 3.29753286372579168528e-01;	constexpr double Ct050_d = -1.11186721721973080790e-13;
constexpr double Cl051_d = 3.35355541921671829186e-01;	constexpr double Ct051_d = -5.33998924722930978781e-13;
constexpr double Cl052_d = 3.40926586970454081893e-01;	constexpr double Ct052_d = +1.39128427745563443274e-13;
constexpr double Cl053_d = 3.46466767347010318190e-01;	constexpr double Ct053_d = -8.01737257312418183375e-13;
constexpr double Cl054_d = 3.51976423156884266064e-01;	constexpr double Ct054_d = +2.93918589553548992299e-13;
constexpr double Cl055_d = 3.57455888921322184615e-01;	constexpr double Ct055_d = +4.81589621697045555671e-13;
constexpr double Cl056_d = 3.62905493690050207078e-01;	constexpr double Ct056_d = -6.81753942579146898773e-13;
constexpr double Cl057_d = 3.68325561159508652054e-01;	constexpr double Ct057_d = -8.00998996948118491268e-13;
constexpr double Cl058_d = 3.73716409792905324139e-01;	constexpr double Ct058_d = +6.78756692778365033902e-13;
constexpr double Cl059_d = 3.79078352934811846353e-01;	constexpr double Ct059_d = +1.57612041907367883553e-13;
constexpr double Cl060_d = 3.84411698911208077334e-01;	constexpr double Ct060_d = -8.76037605087795490100e-13;
constexpr double Cl061_d = 3.89716751140440464951e-01;	constexpr double Ct061_d = -4.15251573361238857096e-13;
constexpr double Cl062_d = 3.94993808240542421117e-01;	constexpr double Ct062_d = +3.26557005178237935716e-13;
constexpr double Cl063_d = 4.00243164127459749579e-01;	constexpr double Ct063_d = -4.47042630778021754523e-13;
constexpr double Cl064_d = 4.05465108107819105498e-01;	constexpr double Ct064_d = +3.45276487522666597485e-13;
constexpr double Cl065_d = 4.10659924984429380856e-01;	constexpr double Ct065_d = +8.39005080688348670037e-13;
constexpr double Cl066_d = 4.15827895143593195826e-01;	constexpr double Ct066_d = +1.17769780472870566612e-13;
constexpr double Cl067_d = 4.20969294644237379543e-01;	constexpr double Ct067_d = -1.07743404042376367258e-13;
constexpr double Cl068_d = 4.26084395310681429692e-01;	constexpr double Ct068_d = +2.18633433839848789759e-13;
constexpr double Cl069_d = 4.31173464818130014464e-01;	constexpr double Ct069_d = +2.41326408726219576728e-13;
constexpr double Cl070_d = 4.36236766774527495727e-01;	constexpr double Ct070_d = +3.90574616919436845563e-13;
constexpr double Cl071_d = 4.41274560804231441580e-01;	constexpr double Ct071_d = +6.43787920109190059392e-13;
constexpr double Cl072_d = 4.46287102628048160113e-01;	constexpr double Ct072_d = +3.71351414245671418435e-13;
constexpr double Cl073_d = 4.51274644139630254358e-01;	constexpr double Ct073_d = -1.71669210964620067017e-13;
constexpr double Cl074_d = 4.56237433481874177232e-01;	constexpr double Ct074_d = -2.86582850549457757161e-13;
constexpr double Cl075_d = 4.61175715121498797089e-01;	constexpr double Ct075_d = +6.71369291540754864478e-13;
constexpr double Cl076_d = 4.66089729925442952663e-01;	constexpr double Ct076_d = -8.43728109297184336590e-13;
constexpr double Cl077_d = 4.70979715219073113985e-01;	constexpr double Ct077_d = -2.82101436394760485271e-13;
constexpr double Cl078_d = 4.75845904869856894948e-01;	constexpr double Ct078_d = +1.07019319621481923122e-13;
constexpr double Cl079_d = 4.80688529345570714213e-01;	constexpr double Ct079_d = +1.81193466263981917130e-13;
constexpr double Cl080_d = 4.85507815781602403149e-01;	constexpr double Ct080_d = +9.84046557347267869531e-14;
constexpr double Cl081_d = 4.90303988044615834951e-01;	constexpr double Ct081_d = +5.78003194352111338006e-13;
constexpr double Cl082_d = 4.95077266798034543172e-01;	constexpr double Ct082_d = -1.83028559756076569798e-13;
constexpr double Cl083_d = 4.99827869556611403823e-01;	constexpr double Ct083_d = -1.62073994633069040638e-13;
constexpr double Cl084_d = 5.04556010751912253909e-01;	constexpr double Ct084_d = +4.83033155574547068412e-13;
constexpr double Cl085_d = 5.09261901790523552336e-01;	constexpr double Ct085_d = -7.15605526224738497376e-13;
constexpr double Cl086_d = 5.13945751101346104406e-01;	constexpr double Ct086_d = +8.88212409438232652903e-13;
constexpr double Cl087_d = 5.18607764208354637958e-01;	constexpr double Ct087_d = -3.09005804884771029251e-13;
constexpr double Cl088_d = 5.23248143765158602037e-01;	constexpr double Ct088_d = -6.10765507180061062442e-13;
constexpr double Cl089_d = 5.27867089620485785417e-01;	constexpr double Ct089_d = +3.56599678171670708693e-13;
constexpr double Cl090_d = 5.32464798869114019908e-01;	constexpr double Ct090_d = +3.57823959264841340300e-13;
constexpr double Cl091_d = 5.37041465897345915437e-01;	constexpr double Ct091_d = -4.62260871941788664685e-13;
constexpr double Cl092_d = 5.41597282432121573947e-01;	constexpr double Ct092_d = +6.22797657629653444999e-13;
constexpr double Cl093_d = 5.46132437597407260910e-01;	constexpr double Ct093_d = +7.28389462460732328708e-13;
constexpr double Cl094_d = 5.50647117952394182794e-01;	constexpr double Ct094_d = +2.68096471672113478846e-13;
constexpr double Cl095_d = 5.55141507540611200966e-01;	constexpr double Ct095_d = -1.09608231779051434884e-13;
constexpr double Cl096_d = 5.59615787935399566777e-01;	constexpr double Ct096_d = +2.31195271260808965508e-14;
constexpr double Cl097_d = 5.64070138285387656651e-01;	constexpr double Ct097_d = -5.84690553352001929355e-13;
constexpr double Cl098_d = 5.68504735352689749561e-01;	constexpr double Ct098_d = -2.10374794841483581109e-14;
constexpr double Cl099_d = 5.72919753562018740922e-01;	constexpr double Ct099_d = -2.33231836621605737037e-13;
constexpr double Cl100_d = 5.77315365035246941261e-01;	constexpr double Ct100_d = -4.23336929747386570000e-13;
constexpr double Cl101_d = 5.81691739635061821900e-01;	constexpr double Ct101_d = -4.39339374342512245519e-13;
constexpr double Cl102_d = 5.86049045003164792433e-01;	constexpr double Ct102_d = +4.13416479869144204429e-13;
constexpr double Cl103_d = 5.90387446602107957006e-01;	constexpr double Ct103_d = +6.84176565228511446293e-14;
constexpr double Cl104_d = 5.94707107746216934174e-01;	constexpr double Ct104_d = +4.75855357721932659132e-13;
constexpr double Cl105_d = 5.99008189645246602595e-01;	constexpr double Ct105_d = +8.36796777703430194606e-13;
constexpr double Cl106_d = 6.03290851438941899687e-01;	constexpr double Ct106_d = -8.57637366073055185112e-13;
constexpr double Cl107_d = 6.07555250224322662689e-01;	constexpr double Ct107_d = +2.19132817360495613102e-13;
constexpr double Cl108_d = 6.11801541106615331955e-01;	constexpr double Ct108_d = -6.22428432579813506464e-13;
constexpr double Cl109_d = 6.16029877215623855591e-01;	constexpr double Ct109_d = -1.09835914235273390815e-13;
constexpr double Cl110_d = 6.20240409751204424538e-01;	constexpr double Ct110_d = +6.53104306747032126168e-13;
constexpr double Cl111_d = 6.24433288012369303033e-01;	constexpr double Ct111_d = -4.75801960764937748039e-13;
constexpr double Cl112_d = 6.28608659422752680257e-01;	constexpr double Ct112_d = -3.78542520679953087281e-13;
constexpr double Cl113_d = 6.32766669570628437214e-01;	constexpr double Ct113_d = +4.09392355085747006171e-13;
constexpr double Cl114_d = 6.36907462236194987781e-01;	constexpr double Ct114_d = +8.74243819698472535151e-13;
constexpr double Cl115_d = 6.41031179420679109171e-01;	constexpr double Ct115_d = +2.52181901663067753461e-13;
constexpr double Cl116_d = 6.45137961373620782979e-01;	constexpr double Ct116_d = -3.60813267284709748850e-14;
constexpr double Cl117_d = 6.49227946625615004450e-01;	constexpr double Ct117_d = -5.05185574362770895007e-13;
constexpr double Cl118_d = 6.53301272011958644725e-01;	constexpr double Ct118_d = +7.86994059330525796980e-13;
constexpr double Cl119_d = 6.57358072709030238912e-01;	constexpr double Ct119_d = -6.70208734430272068394e-13;
constexpr double Cl120_d = 6.61398482245203922503e-01;	constexpr double Ct120_d = +1.61085771457902815484e-13;
constexpr double Cl121_d = 6.65422632544505177066e-01;	constexpr double Ct121_d = +5.85271902556888656832e-13;
constexpr double Cl122_d = 6.69430653942981734872e-01;	constexpr double Ct122_d = -3.52467566851893954194e-13;
constexpr double Cl123_d = 6.73422675212350441143e-01;	constexpr double Ct123_d = -1.83720822843208564379e-13;
constexpr double Cl124_d = 6.77398823590920073912e-01;	constexpr double Ct124_d = +8.86066881759101088178e-13;
constexpr double Cl125_d = 6.81359224807238206267e-01;	constexpr double Ct125_d = +6.64862696148071918856e-13;
constexpr double Cl126_d = 6.85304003098281100392e-01;	constexpr double Ct126_d = +6.38316168585090215615e-13;
constexpr double Cl127_d = 6.89233281238557538018e-01;	constexpr double Ct127_d = +2.51442313151106766611e-13;
constexpr double Cl128_d = 6.93147180560117703862e-01;	constexpr double Ct128_d = -1.72394433797795620933e-13;

constexpr double dlInt   = 4503599627370496.0;

constexpr double lgLed_d = 3.60436533891171560912e+01;
constexpr double lgTrl_d = 1.30437327605648079043e-13;

/*	For the trigonometric functions	*/
constexpr double PiA_d = -3.1415926218032836914;
constexpr double PiB_d = -3.1786509424591713469e-08;
constexpr double PiC_d = -1.2246467864107188502e-16;
constexpr double PiD_d = -1.2736634327021899816e-24;

constexpr double s0_d  = -7.97255955009037868891952e-18;
constexpr double s1_d  =  2.81009972710863200091251e-15;
constexpr double s2_d  = -7.64712219118158833288484e-13;
constexpr double s3_d  =  1.60590430605664501629054e-10;
constexpr double s4_d  = -2.50521083763502045810755e-08;
constexpr double s5_d  =  2.75573192239198747630416e-06;
constexpr double s6_d  = -0.000198412698412696162806809;
constexpr double s7_d  =  0.00833333333333332974823815;
constexpr double s8_d  = -0.166666666666666657414808;
#ifdef	__AVX512F__
constexpr _MData_ rPid	    = { M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24) };
constexpr _MData_ dPid	    = { M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23) };
constexpr _MData_ rCte      = {   16777216.,   16777216.,   16777216.,   16777216.,   16777216.,   16777216.,   16777216.,   16777216. };
constexpr _MData_ oPid      = {     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI };
constexpr _MData_ zeroNegd  = {        -0.0,        -0.0,        -0.0,        -0.0,        -0.0,        -0.0,        -0.0,        -0.0 };
constexpr _MData_ dHlf      = {         0.5,         0.5,         0.5,         0.5,         0.5,         0.5,         0.5,         0.5 };
constexpr _MData_ dOne      = {         1.0,         1.0,         1.0,         1.0,         1.0,         1.0,         1.0,         1.0 };
constexpr _MData_ TriMaxd   = {        1e15,        1e15,        1e15,        1e15,        1e15,        1e15,        1e15,        1e15 };
constexpr _MData_ dInf      = {       Inf_d,       Inf_d,       Inf_d,       Inf_d,       Inf_d,       Inf_d,       Inf_d,       Inf_d };
constexpr _MData_ dNan      = {       Nan_d,       Nan_d,       Nan_d,       Nan_d,       Nan_d,       Nan_d,       Nan_d,       Nan_d };
constexpr _MData_ PiAd      = {       PiA_d,       PiA_d,       PiA_d,       PiA_d,       PiA_d,       PiA_d,       PiA_d,       PiA_d };
constexpr _MData_ PiBd      = {       PiB_d,       PiB_d,       PiB_d,       PiB_d,       PiB_d,       PiB_d,       PiB_d,       PiB_d };
constexpr _MData_ PiCd      = {       PiC_d,       PiC_d,       PiC_d,       PiC_d,       PiC_d,       PiC_d,       PiC_d,       PiC_d };
constexpr _MData_ PiDd      = {       PiD_d,       PiD_d,       PiD_d,       PiD_d,       PiD_d,       PiD_d,       PiD_d,       PiD_d };
constexpr _MData_ hPiAd     = {   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d };
constexpr _MData_ hPiBd     = {   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d };
constexpr _MData_ hPiCd     = {   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d };
constexpr _MData_ hPiDd     = {   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d };
constexpr _MData_ s0d       = {        s0_d,        s0_d,        s0_d,        s0_d,        s0_d,        s0_d,        s0_d,        s0_d };
constexpr _MData_ s1d       = {        s1_d,        s1_d,        s1_d,        s1_d,        s1_d,        s1_d,        s1_d,        s1_d };
constexpr _MData_ s2d       = {        s2_d,        s2_d,        s2_d,        s2_d,        s2_d,        s2_d,        s2_d,        s2_d };
constexpr _MData_ s3d       = {        s3_d,        s3_d,        s3_d,        s3_d,        s3_d,        s3_d,        s3_d,        s3_d };
constexpr _MData_ s4d       = {        s4_d,        s4_d,        s4_d,        s4_d,        s4_d,        s4_d,        s4_d,        s4_d };
constexpr _MData_ s5d       = {        s5_d,        s5_d,        s5_d,        s5_d,        s5_d,        s5_d,        s5_d,        s5_d };
constexpr _MData_ s6d       = {        s6_d,        s6_d,        s6_d,        s6_d,        s6_d,        s6_d,        s6_d,        s6_d };
constexpr _MData_ s7d       = {        s7_d,        s7_d,        s7_d,        s7_d,        s7_d,        s7_d,        s7_d,        s7_d };
constexpr _MData_ s8d       = {        s8_d,        s8_d,        s8_d,        s8_d,        s8_d,        s8_d,        s8_d,        s8_d };

constexpr _MData_ vTh1d     = {       th1_d,       th1_d,       th1_d,       th1_d,       th1_d,       th1_d,       th1_d,       th1_d };
constexpr _MData_ vTh2d     = {       th2_d,       th2_d,       th2_d,       th2_d,       th2_d,       th2_d,       th2_d,       th2_d };
constexpr _MData_ vL1d      = {        L1_d,        L1_d,        L1_d,        L1_d,        L1_d,        L1_d,        L1_d,        L1_d };
constexpr _MData_ vL2d      = {       -L2_d,       -L2_d,       -L2_d,       -L2_d,       -L2_d,       -L2_d,       -L2_d,       -L2_d };
constexpr _MData_ vA1d      = {        A1_d,        A1_d,        A1_d,        A1_d,        A1_d,        A1_d,        A1_d,        A1_d };
constexpr _MData_ vA2d      = {        A2_d,        A2_d,        A2_d,        A2_d,        A2_d,        A2_d,        A2_d,        A2_d };
constexpr _MData_ vA3d      = {        A3_d,        A3_d,        A3_d,        A3_d,        A3_d,        A3_d,        A3_d,        A3_d };
constexpr _MData_ vA4d      = {        A4_d,        A4_d,        A4_d,        A4_d,        A4_d,        A4_d,        A4_d,        A4_d };
constexpr _MData_ vA5d      = {        A5_d,        A5_d,        A5_d,        A5_d,        A5_d,        A5_d,        A5_d,        A5_d };
constexpr _MData_ vIvLd     = {       ivL_d,       ivL_d,       ivL_d,       ivL_d,       ivL_d,       ivL_d,       ivL_d,       ivL_d };
constexpr _MData_ vi2t7d    = {     1./128.,     1./128.,     1./128.,     1./128.,     1./128.,     1./128.,     1./128.,     1./128. };
constexpr _MData_ vB1d      = {        B1_d,        B1_d,        B1_d,        B1_d,        B1_d,        B1_d,        B1_d,        B1_d };
constexpr _MData_ vB2d      = {        B2_d,        B2_d,        B2_d,        B2_d,        B2_d,        B2_d,        B2_d,        B2_d };
constexpr _MData_ vB3d      = {        B3_d,        B3_d,        B3_d,        B3_d,        B3_d,        B3_d,        B3_d,        B3_d };
constexpr _MData_ vB4d      = {        B4_d,        B4_d,        B4_d,        B4_d,        B4_d,        B4_d,        B4_d,        B4_d };
constexpr _MData_ vB5d      = {        B5_d,        B5_d,        B5_d,        B5_d,        B5_d,        B5_d,        B5_d,        B5_d };
constexpr _MData_ vLg2ld    = {     Cl128_d,     Cl128_d,     Cl128_d,     Cl128_d,     Cl128_d,     Cl128_d,     Cl128_d,     Cl128_d };
constexpr _MData_ vLg2td    = {     Ct128_d,     Ct128_d,     Ct128_d,     Ct128_d,     Ct128_d,     Ct128_d,     Ct128_d,     Ct128_d };
constexpr _MData_ nmDnd     = {       dlInt,       dlInt,       dlInt,       dlInt,       dlInt,       dlInt,       dlInt,       dlInt };
constexpr _MData_ lgDld     = {     lgLed_d,     lgLed_d,     lgLed_d,     lgLed_d,     lgLed_d,     lgLed_d,     lgLed_d,     lgLed_d };
constexpr _MData_ lgDtd     = {     lgTrl_d,     lgTrl_d,     lgTrl_d,     lgTrl_d,     lgTrl_d,     lgTrl_d,     lgTrl_d,     lgTrl_d };

constexpr _MInt_  iZero     = {           0,           0,           0,           0,           0,           0,           0,           0 };
constexpr _MInt_  one       = {  4294967297,  4294967297,  4294967297,  4294967297,  4294967297,  4294967297,  4294967297,  4294967297 };
constexpr _MInt_  two       = {  8589934594,  8589934594,  8589934594,  8589934594,  8589934594,  8589934594,  8589934594,  8589934594 };
constexpr _MHnt_  hOne      = {  4294967297,  4294967297,  4294967297,  4294967297 };
constexpr _MHnt_  hTwo      = {  8589934594,  8589934594,  8589934594,  8589934594 };
constexpr _MHnt_  iZerh     = {           0,           0,           0,           0 };
constexpr _MInt_  m32Mask   = {       m32Mk,       m32Mk,       m32Mk,       m32Mk,       m32Mk,       m32Mk,       m32Mk,       m32Mk };
constexpr _MInt_  iSignMsk  = {       iSgMk,       iSgMk,       iSgMk,       iSgMk,       iSgMk,       iSgMk,       iSgMk,       iSgMk };
constexpr _MInt_  iFillMsk  = {       iFlMk,       iFlMk,       iFlMk,       iFlMk,       iFlMk,       iFlMk,       iFlMk,       iFlMk };
constexpr _MHnt_  h32Mask   = {       m32Mk,       m32Mk,       m32Mk,       m32Mk };
constexpr _MHnt_  hSignMsk  = {       iSgMk,       iSgMk,       iSgMk,       iSgMk };
constexpr _MHnt_  hFillMsk  = {       iFlMk,       iFlMk,       iFlMk,       iFlMk };
#elif   defined(__AVX__)
constexpr _MData_ rPid	    = { M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24) };
constexpr _MData_ dPid	    = { M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23) };
constexpr _MData_ rCte      = {     (1<<24),     (1<<24),     (1<<24),     (1<<24) };
constexpr _MData_ dCte      = {     (1<<23),     (1<<23),     (1<<23),     (1<<23) };
constexpr _MData_ oPid      = {      M_1_PI,      M_1_PI,      M_1_PI,      M_1_PI };
constexpr _MData_ zeroNegd  = {        -0.0,        -0.0,        -0.0,        -0.0 };
constexpr _MData_ dHlf      = {         0.5,         0.5,         0.5,         0.5 };
constexpr _MData_ dOne      = {         1.0,         1.0,         1.0,         1.0 };
#ifdef	__FMA__
constexpr _MData_ TriMaxd   = {        1e15,        1e15,        1e15,        1e15 };
#else
constexpr _MData_ TriMaxd   = {        1e12,        1e12,        1e12,        1e12 };
#endif
constexpr _MData_ dInf      = {       Inf_d,       Inf_d,       Inf_d,       Inf_d };
constexpr _MData_ dNan      = {       Nan_d,       Nan_d,       Nan_d,       Nan_d };
constexpr _MData_ PiAd      = {       PiA_d,       PiA_d,       PiA_d,       PiA_d };
constexpr _MData_ PiBd      = {       PiB_d,       PiB_d,       PiB_d,       PiB_d };
constexpr _MData_ PiCd      = {       PiC_d,       PiC_d,       PiC_d,       PiC_d };
constexpr _MData_ PiDd      = {       PiD_d,       PiD_d,       PiD_d,       PiD_d };
constexpr _MData_ hPiAd     = {   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d };
constexpr _MData_ hPiBd     = {   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d };
constexpr _MData_ hPiCd     = {   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d };
constexpr _MData_ hPiDd     = {   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d };
constexpr _MData_ s0d       = {        s0_d,        s0_d,        s0_d,        s0_d };
constexpr _MData_ s1d       = {        s1_d,        s1_d,        s1_d,        s1_d };
constexpr _MData_ s2d       = {        s2_d,        s2_d,        s2_d,        s2_d };
constexpr _MData_ s3d       = {        s3_d,        s3_d,        s3_d,        s3_d };
constexpr _MData_ s4d       = {        s4_d,        s4_d,        s4_d,        s4_d };
constexpr _MData_ s5d       = {        s5_d,        s5_d,        s5_d,        s5_d };
constexpr _MData_ s6d       = {        s6_d,        s6_d,        s6_d,        s6_d };
constexpr _MData_ s7d       = {        s7_d,        s7_d,        s7_d,        s7_d };
constexpr _MData_ s8d       = {        s8_d,        s8_d,        s8_d,        s8_d };

constexpr _MData_ vTh1d     = {       th1_d,       th1_d,       th1_d,       th1_d };
constexpr _MData_ vTh2d     = {       th2_d,       th2_d,       th2_d,       th2_d };
constexpr _MData_ vL1d      = {        L1_d,        L1_d,        L1_d,        L1_d };
constexpr _MData_ vL2d      = {       -L2_d,       -L2_d,       -L2_d,       -L2_d };
constexpr _MData_ vA1d      = {        A1_d,        A1_d,        A1_d,        A1_d };
constexpr _MData_ vA2d      = {        A2_d,        A2_d,        A2_d,        A2_d };
constexpr _MData_ vA3d      = {        A3_d,        A3_d,        A3_d,        A3_d };
constexpr _MData_ vA4d      = {        A4_d,        A4_d,        A4_d,        A4_d };
constexpr _MData_ vA5d      = {        A5_d,        A5_d,        A5_d,        A5_d };
constexpr _MData_ vIvLd     = {       ivL_d,       ivL_d,       ivL_d,       ivL_d };
constexpr _MData_ vi2t7d    = {     1./128.,     1./128.,     1./128.,     1./128. };
constexpr _MData_ vB1d      = {        B1_d,        B1_d,        B1_d,        B1_d };
constexpr _MData_ vB2d      = {        B2_d,        B2_d,        B2_d,        B2_d };
constexpr _MData_ vB3d      = {        B3_d,        B3_d,        B3_d,        B3_d };
constexpr _MData_ vB4d      = {        B4_d,        B4_d,        B4_d,        B4_d };
constexpr _MData_ vB5d      = {        B5_d,        B5_d,        B5_d,        B5_d };
constexpr _MData_ vLg2ld    = {     Cl128_d,     Cl128_d,     Cl128_d,     Cl128_d };
constexpr _MData_ vLg2td    = {     Ct128_d,     Ct128_d,     Ct128_d,     Ct128_d };
constexpr _MData_ nmDnd     = {       dlInt,       dlInt,       dlInt,       dlInt };
constexpr _MData_ lgDld     = {     lgLed_d,     lgLed_d,     lgLed_d,     lgLed_d };
constexpr _MData_ lgDtd     = {     lgTrl_d,     lgTrl_d,     lgTrl_d,     lgTrl_d };

constexpr _MInt_  iZero     = {           0,           0,           0,           0 };
constexpr _MInt_  one       = {  4294967297,  4294967297,  4294967297,  4294967297 };
constexpr _MInt_  two       = {  8589934594,  8589934594,  8589934594,  8589934594 };
constexpr _MHnt_  iZerh     = {           0,           0 };
constexpr _MHnt_  hOne      = {  4294967297,  4294967297 };
constexpr _MHnt_  hTwo      = {  8589934594,  8589934594 };
constexpr _MInt_  m32Mask   = {       m32Mk,       m32Mk,       m32Mk,       m32Mk };
constexpr _MInt_  iSignMsk  = {       iSgMk,       iSgMk,       iSgMk,       iSgMk };
constexpr _MInt_  iFillMsk  = {       iFlMk,       iFlMk,       iFlMk,       iFlMk };
constexpr _MHnt_  h32Mask   = {       m32Mk,       m32Mk };
constexpr _MHnt_  hSignMsk  = {       iSgMk,       iSgMk };
constexpr _MHnt_  hFillMsk  = {       iFlMk,       iFlMk };
#else
constexpr _MData_ rPid	    = { M_1_PI/(1<<24), M_1_PI/(1<<24) };
constexpr _MData_ dPid	    = { M_1_PI/(1<<23), M_1_PI/(1<<23) };
constexpr _MData_ rCte      = {   (1 << 24),   (1 << 24) };
constexpr _MData_ dCte      = {   (1 << 23),   (1 << 23) };
constexpr _MData_ oPid      = {      M_1_PI,      M_1_PI };
constexpr _MData_ zeroNegd  = {        -0.0,        -0.0 };
constexpr _MData_ dHlf      = {         0.5,         0.5 };
constexpr _MData_ dOne      = {         1.0,         1.0 };
constexpr _MData_ TriMaxd   = {        1e15,        1e15 };
constexpr _MData_ dInf      = {       Inf_d,       Inf_d };
constexpr _MData_ dNan      = {       Nan_d,       Nan_d };
constexpr _MData_ PiAd      = {       PiA_d,       PiA_d };
constexpr _MData_ PiBd      = {       PiB_d,       PiB_d };
constexpr _MData_ PiCd      = {       PiC_d,       PiC_d };
constexpr _MData_ PiDd      = {       PiD_d,       PiD_d };
constexpr _MData_ hPiAd     = {   0.5*PiA_d,   0.5*PiA_d };
constexpr _MData_ hPiBd     = {   0.5*PiB_d,   0.5*PiB_d };
constexpr _MData_ hPiCd     = {   0.5*PiC_d,   0.5*PiC_d };
constexpr _MData_ hPiDd     = {   0.5*PiD_d,   0.5*PiD_d };
constexpr _MData_ s0d       = {        s0_d,        s0_d };
constexpr _MData_ s1d       = {        s1_d,        s1_d };
constexpr _MData_ s2d       = {        s2_d,        s2_d };
constexpr _MData_ s3d       = {        s3_d,        s3_d };
constexpr _MData_ s4d       = {        s4_d,        s4_d };
constexpr _MData_ s5d       = {        s5_d,        s5_d };
constexpr _MData_ s6d       = {        s6_d,        s6_d };
constexpr _MData_ s7d       = {        s7_d,        s7_d };
constexpr _MData_ s8d       = {        s8_d,        s8_d };

constexpr _MData_ vTh1d     = {       th1_d,       th1_d };
constexpr _MData_ vTh2d     = {       th2_d,       th2_d };
constexpr _MData_ vL1d      = {        L1_d,        L1_d };
constexpr _MData_ vL2d      = {       -L2_d,       -L2_d };
constexpr _MData_ vA1d      = {        A1_d,        A1_d };
constexpr _MData_ vA2d      = {        A2_d,        A2_d };
constexpr _MData_ vIvLd     = {       ivL_d,       ivL_d };
constexpr _MData_ vi2t7d    = {     1./128.,     1./128. };
constexpr _MData_ vB1d      = {        B1_d,        B1_d };
constexpr _MData_ vB2d      = {        B2_d,        B2_d };
constexpr _MData_ vB3d      = {        B3_d,        B3_d };
constexpr _MData_ vB4d      = {        B4_d,        B4_d };
constexpr _MData_ vB5d      = {        B5_d,        B5_d };
constexpr _MData_ vLg2ld    = {     Cl128_d,     Cl128_d };
constexpr _MData_ vLg2td    = {     Ct128_d,     Ct128_d };
constexpr _MData_ nmDnd     = {       dlInt,       dlInt };
constexpr _MData_ lgDld     = {     lgLed_d,     lgLed_d };
constexpr _MData_ lgDtd     = {     lgTrl_d,     lgTrl_d };

constexpr _MInt_  iZero     = {           0,           0 };
constexpr _MInt_  one       = {  4294967297,  4294967297 };
constexpr _MInt_  two       = {  8589934594,  8589934594 };
constexpr _MInt_  m32Mask   = {       m32Mk,       m32Mk };
constexpr _MInt_  iSignMsk  = {       iSgMk,       iSgMk };
constexpr _MInt_  iFillMsk  = {       iFlMk,       iFlMk };
#endif

#endif

inline void printlVar(_MInt_ d, const char *name)
{
	printf ("%s", name);
#if	defined(__AVX512F__)
	long long int r[8] __attribute((aligned(64)));
	opCode(store_si512, ((_MInt_ *)r), d);
	for (int i=0; i<8; i++)
#elif	defined(__AVX__)
	long long int r[4] __attribute((aligned(32)));
	opCode(store_si256, ((_MInt_ *)r), d);
	for (int i=0; i<4; i++)
#else
	long long int r[2] __attribute((aligned(16)));
	opCode(store_si128, ((_MInt_ *)r), d);
	for (int i=0; i<2; i++)
#endif
		printf(" %zu", r[i]);
	printf("\n");
}

#ifdef	__AVX__
inline void printhVar(_MHnt_ d, const char *name)
#else
inline void printhVar(_MInt_ d, const char *name)
#endif
{
	printf ("%s", name);
#if	defined(__AVX512F__)
	int r[8] __attribute((aligned(32)));
	opCodl(store_si256, ((_MHnt_ *)r), d);
	for (int i=0; i<8; i++)
#elif	defined(__AVX__)
	int r[4] __attribute((aligned(16)));
	opCodl(store_si128, ((_MHnt_ *)r), d);
	for (int i=0; i<4; i++)
#else
	int r[4] __attribute((aligned(16)));
	opCode(store_si128, ((_MInt_ *)r), d);
	for (int i=0; i<4; i++)
#endif
		printf(" %d", r[i]);
	printf("\n");
}

inline void printdVar(_MData_ d, const char *name) {

	printf ("%s", name);
#if	defined(__AVX512F__)
	for (int i=0; i<8; i++)
#elif	defined(__AVX__)
	for (int i=0; i<4; i++)
#else
	for (int i=0; i<2; i++)
#endif
		printf(" %+.17le", d[i]);
	printf("\n");
}

#undef	_MData_

#if	defined(__AVX512F__)
	#define	_MData_ __m512
#elif	defined(__AVX__)
	#define	_MData_ __m256
#else
	#define	_MData_ __m128
#endif 

#ifndef	__INTEL_COMPILER

constexpr float Inf_f = __builtin_inff();
constexpr float Nan_f = __builtin_nanf("0x3FFFFF");

/*	For the exponential	*/
constexpr float th1_f = 220.4208f;
constexpr float th2_f = 2.9802322E-8f;
constexpr float ivL_f = 46.16624f;
constexpr float L1_f  = 0.021660805f;
constexpr float L2_f  = 4.464396e-8f;
constexpr float A1_f  = 0.50000405f;
constexpr float A2_f  = 0.16666764f;
constexpr float mEx_f = 1.17549435e-38f;

constexpr float Sl00_f = 1.0000000f;	 constexpr float Sl01_f = 1.0218964f;	  constexpr float Sl02_f = 1.0442734f;	   constexpr float Sl03_f = 1.0671387f;
constexpr float Sl04_f = 1.0905075f;	 constexpr float Sl05_f = 1.1143799f;	  constexpr float Sl06_f = 1.1387863f;	   constexpr float Sl07_f = 1.1637192f;
constexpr float Sl08_f = 1.1892014f;	 constexpr float Sl09_f = 1.2152405f;	  constexpr float Sl10_f = 1.2418518f;	   constexpr float Sl11_f = 1.2690506f;
constexpr float Sl12_f = 1.2968369f;	 constexpr float Sl13_f = 1.3252335f;	  constexpr float Sl14_f = 1.3542480f;	   constexpr float Sl15_f = 1.3839035f;
constexpr float Sl16_f = 1.4142075f;	 constexpr float Sl17_f = 1.4451752f;	  constexpr float Sl18_f = 1.4768219f;	   constexpr float Sl19_f = 1.5091629f;
constexpr float Sl20_f = 1.5422058f;	 constexpr float Sl21_f = 1.5759735f;	  constexpr float Sl22_f = 1.6104889f;	   constexpr float Sl23_f = 1.6457520f;
constexpr float Sl24_f = 1.6817856f;	 constexpr float Sl25_f = 1.7186127f;	  constexpr float Sl26_f = 1.7562485f;	   constexpr float Sl27_f = 1.7947083f;
constexpr float Sl28_f = 1.8340073f;	 constexpr float Sl29_f = 1.8741608f;	  constexpr float Sl30_f = 1.9151993f;	   constexpr float Sl31_f = 1.9571381f;

constexpr float St00_f = 0.0000000e+00f; constexpr float St01_f = 7.8634940e-07f; constexpr float St02_f = 4.0596257e-07f; constexpr float St03_f = 1.7288019e-06f;
constexpr float St04_f = 2.2534104e-07f; constexpr float St05_f = 6.8597833e-06f; constexpr float St06_f = 2.3188388e-06f; constexpr float St07_f = 5.6815315e-06f;
constexpr float St08_f = 5.7600223e-06f; constexpr float St09_f = 6.8814647e-06f; constexpr float St10_f = 6.0054331e-06f; constexpr float St11_f = 3.5904719e-07f;
constexpr float St12_f = 2.7016238e-06f; constexpr float St13_f = 3.1836871e-06f; constexpr float St14_f = 7.5000621e-06f; constexpr float St15_f = 6.3785460e-06f;
constexpr float St16_f = 6.1038768e-06f; constexpr float St17_f = 5.6360786e-06f; constexpr float St18_f = 4.2465254e-06f; constexpr float St19_f = 1.5247614e-06f;
constexpr float St20_f = 5.0148610e-06f; constexpr float St21_f = 7.3343658e-06f; constexpr float St22_f = 1.4403477e-06f; constexpr float St23_f = 3.5250289e-06f;
constexpr float St24_f = 7.2470111e-06f; constexpr float St25_f = 6.6272241e-06f; constexpr float St26_f = 3.6862523e-06f; constexpr float St27_f = 8.2304996e-07f;
constexpr float St28_f = 8.2322578e-07f; constexpr float St29_f = 6.8675085e-06f; constexpr float St30_f = 7.2816119e-06f; constexpr float St31_f = 6.0626521e-06f;

/*	For the logarithm		*/

constexpr float B1_f   = -5.00002861e-01;
constexpr float B2_f   = +3.33329856e-01;

constexpr float Cl000_f = 0.0000000e+00;	constexpr float Ct000_f = +0.0000000e+00;
constexpr float Cl001_f = 7.7819824e-03;	constexpr float Ct001_f = +1.5802018e-07;
constexpr float Cl002_f = 1.5502930e-02;	constexpr float Ct002_f = +1.2568485e-06;
constexpr float Cl003_f = 2.3162842e-02;	constexpr float Ct003_f = +4.2174847e-06;
constexpr float Cl004_f = 3.0776978e-02;	constexpr float Ct004_f = -5.3188723e-06;
constexpr float Cl005_f = 3.8314819e-02;	constexpr float Ct005_f = +4.0449662e-06;
constexpr float Cl006_f = 4.5806885e-02;	constexpr float Ct006_f = +2.6512657e-06;
constexpr float Cl007_f = 5.3237915e-02;	constexpr float Ct007_f = +6.5994797e-06;
constexpr float Cl008_f = 6.0623169e-02;	constexpr float Ct008_f = +1.4528711e-06;
constexpr float Cl009_f = 6.7947388e-02;	constexpr float Ct009_f = +3.2742132e-06;
constexpr float Cl010_f = 7.5225830e-02;	constexpr float Ct010_f = -2.4088405e-06;
constexpr float Cl011_f = 8.2443237e-02;	constexpr float Ct011_f = +4.3190639e-07;
constexpr float Cl012_f = 8.9614868e-02;	constexpr float Ct012_f = -2.7094744e-06;
constexpr float Cl013_f = 9.6725464e-02;	constexpr float Ct013_f = +4.1625914e-06;
constexpr float Cl014_f = 1.0379028e-01;	constexpr float Ct014_f = +6.5104785e-06;
constexpr float Cl015_f = 1.1080933e-01;	constexpr float Ct015_f = +5.0401684e-06;
constexpr float Cl016_f = 1.1778259e-01;	constexpr float Ct016_f = +4.4288295e-07;
constexpr float Cl017_f = 1.2471008e-01;	constexpr float Ct017_f = -6.6045069e-06;
constexpr float Cl018_f = 1.3157654e-01;	constexpr float Ct018_f = -1.8029722e-07;
constexpr float Cl019_f = 1.3839722e-01;	constexpr float Ct019_f = +5.1060622e-06;
constexpr float Cl020_f = 1.4518738e-01;	constexpr float Ct020_f = -5.3680852e-06;
constexpr float Cl021_f = 1.5191650e-01;	constexpr float Ct021_f = -4.6188041e-07;
constexpr float Cl022_f = 1.5859985e-01;	constexpr float Ct022_f = +5.1766610e-06;
constexpr float Cl023_f = 1.6525269e-01;	constexpr float Ct023_f = -3.1126516e-06;
constexpr float Cl024_f = 1.7184448e-01;	constexpr float Ct024_f = +5.7745048e-06;
constexpr float Cl025_f = 1.7840576e-01;	constexpr float Ct025_f = +1.8957541e-06;
constexpr float Cl026_f = 1.8492126e-01;	constexpr float Ct026_f = +1.0738456e-06;
constexpr float Cl027_f = 1.9139099e-01;	constexpr float Ct027_f = +3.8617887e-06;
constexpr float Cl028_f = 1.9783020e-01;	constexpr float Ct028_f = -4.4568654e-06;
constexpr float Cl029_f = 2.0420837e-01;	constexpr float Ct029_f = +7.1674053e-06;
constexpr float Cl030_f = 2.1057129e-01;	constexpr float Ct030_f = -6.5199552e-06;
constexpr float Cl031_f = 2.1687317e-01;	constexpr float Ct031_f = +7.6935530e-07;
constexpr float Cl032_f = 2.2314453e-01;	constexpr float Ct032_f = -9.7993579e-07;
constexpr float Cl033_f = 2.2937012e-01;	constexpr float Ct033_f = +3.9838773e-06;
constexpr float Cl034_f = 2.3556519e-01;	constexpr float Ct034_f = +8.8576589e-07;
constexpr float Cl035_f = 2.4171448e-01;	constexpr float Ct035_f = +5.4593481e-06;
constexpr float Cl036_f = 2.4783325e-01;	constexpr float Ct036_f = +2.9119515e-06;
constexpr float Cl037_f = 2.5392151e-01;	constexpr float Ct037_f = -6.2988081e-06;
constexpr float Cl038_f = 2.5996399e-01;	constexpr float Ct038_f = -6.4648209e-06;
constexpr float Cl039_f = 2.6596069e-01;	constexpr float Ct039_f = +2.8551378e-06;
constexpr float Cl040_f = 2.7192688e-01;	constexpr float Ct040_f = +6.8356008e-06;
constexpr float Cl041_f = 2.7786255e-01;	constexpr float Ct041_f = +5.9021753e-06;
constexpr float Cl042_f = 2.8376770e-01;	constexpr float Ct042_f = +4.7293533e-07;
constexpr float Cl043_f = 2.8962708e-01;	constexpr float Ct043_f = +6.2173877e-06;
constexpr float Cl044_f = 2.9547119e-01;	constexpr float Ct044_f = -6.9785124e-06;
constexpr float Cl045_f = 3.0125427e-01;	constexpr float Ct045_f = +7.0581172e-06;
constexpr float Cl046_f = 3.0702209e-01;	constexpr float Ct046_f = +2.9405683e-06;
constexpr float Cl047_f = 3.1275940e-01;	constexpr float Ct047_f = -3.6894102e-06;
constexpr float Cl048_f = 3.1845093e-01;	constexpr float Ct048_f = +2.8033842e-06;
constexpr float Cl049_f = 3.2411194e-01;	constexpr float Ct049_f = +7.5301776e-06;
constexpr float Cl050_f = 3.2975769e-01;	constexpr float Ct050_f = -4.4040572e-06;
constexpr float Cl051_f = 3.3535767e-01;	constexpr float Ct051_f = -2.1240945e-06;
constexpr float Cl052_f = 3.4092712e-01;	constexpr float Ct052_f = -5.3705284e-07;
constexpr float Cl053_f = 3.4646606e-01;	constexpr float Ct053_f = +7.0289308e-07;
constexpr float Cl054_f = 3.5197449e-01;	constexpr float Ct054_f = +1.9358525e-06;
constexpr float Cl055_f = 3.5745239e-01;	constexpr float Ct055_f = +3.4963437e-06;
constexpr float Cl056_f = 3.6289978e-01;	constexpr float Ct056_f = +5.7134159e-06;
constexpr float Cl057_f = 3.6833191e-01;	constexpr float Ct057_f = -6.3480210e-06;
constexpr float Cl058_f = 3.7371826e-01;	constexpr float Ct058_f = -1.8519252e-06;
constexpr float Cl059_f = 3.7907410e-01;	constexpr float Ct059_f = +4.2562553e-06;
constexpr float Cl060_f = 3.8441467e-01;	constexpr float Ct060_f = -2.9739412e-06;
constexpr float Cl061_f = 3.8970947e-01;	constexpr float Ct061_f = +7.2784838e-06;
constexpr float Cl062_f = 3.9498901e-01;	constexpr float Ct062_f = +4.7945690e-06;
constexpr float Cl063_f = 4.0023804e-01;	constexpr float Ct063_f = +5.1270176e-06;
constexpr float Cl064_f = 4.0547180e-01;	constexpr float Ct064_f = -6.6936496e-06;
constexpr float Cl065_f = 4.1065979e-01;	constexpr float Ct065_f = +1.3494621e-07;
constexpr float Cl066_f = 4.1583252e-01;	constexpr float Ct066_f = -4.6243875e-06;
constexpr float Cl067_f = 4.2097473e-01;	constexpr float Ct067_f = -5.4368012e-06;
constexpr float Cl068_f = 4.2608643e-01;	constexpr float Ct068_f = -2.0304703e-06;
constexpr float Cl069_f = 4.3116760e-01;	constexpr float Ct069_f = +5.8622793e-06;
constexpr float Cl070_f = 4.3623352e-01;	constexpr float Ct070_f = +3.2462671e-06;
constexpr float Cl071_f = 4.4126892e-01;	constexpr float Ct071_f = +5.6399064e-06;
constexpr float Cl072_f = 4.4628906e-01;	constexpr float Ct072_f = -1.9598716e-06;
constexpr float Cl073_f = 4.5127869e-01;	constexpr float Ct073_f = -4.0423840e-06;
constexpr float Cl074_f = 4.5623779e-01;	constexpr float Ct074_f = -3.5948716e-07;
constexpr float Cl075_f = 4.6118164e-01;	constexpr float Ct075_f = -5.9255028e-06;
constexpr float Cl076_f = 4.6609497e-01;	constexpr float Ct076_f = -5.2407785e-06;
constexpr float Cl077_f = 4.7097778e-01;	constexpr float Ct077_f = +1.9320157e-06;
constexpr float Cl078_f = 4.7584534e-01;	constexpr float Ct078_f = +5.6795590e-07;
constexpr float Cl079_f = 4.8068237e-01;	constexpr float Ct079_f = +6.1562989e-06;
constexpr float Cl080_f = 4.8550415e-01;	constexpr float Ct080_f = +3.6653911e-06;
constexpr float Cl081_f = 4.9031067e-01;	constexpr float Ct081_f = -6.6809001e-06;
constexpr float Cl082_f = 4.9507141e-01;	constexpr float Ct082_f = +5.8556650e-06;
constexpr float Cl083_f = 4.9983215e-01;	constexpr float Ct083_f = -4.2837639e-06;
constexpr float Cl084_f = 5.0456238e-01;	constexpr float Ct084_f = -6.3671773e-06;
constexpr float Cl085_f = 5.0926208e-01;	constexpr float Ct085_f = -1.8317113e-07;
constexpr float Cl086_f = 5.1394653e-01;	constexpr float Ct086_f = -7.8210089e-07;
constexpr float Cl087_f = 5.1860046e-01;	constexpr float Ct087_f = +7.3003409e-06;
constexpr float Cl088_f = 5.2325439e-01;	constexpr float Ct088_f = -6.2507667e-06;
constexpr float Cl089_f = 5.2786255e-01;	constexpr float Ct089_f = +4.5407927e-06;
constexpr float Cl090_f = 5.3247070e-01;	constexpr float Ct090_f = -5.9042555e-06;
constexpr float Cl091_f = 5.3704834e-01;	constexpr float Ct091_f = -6.8739469e-06;
constexpr float Cl092_f = 5.4159546e-01;	constexpr float Ct092_f = +1.8234484e-06;
constexpr float Cl093_f = 5.4612732e-01;	constexpr float Ct093_f = +5.1182622e-06;
constexpr float Cl094_f = 5.5064392e-01;	constexpr float Ct094_f = +3.1970542e-06;
constexpr float Cl095_f = 5.5514526e-01;	constexpr float Ct095_f = -3.7561314e-06;
constexpr float Cl096_f = 5.5961609e-01;	constexpr float Ct096_f = -3.0093176e-07;
constexpr float Cl097_f = 5.6407166e-01;	constexpr float Ct097_f = -1.5169886e-06;
constexpr float Cl098_f = 5.6851196e-01;	constexpr float Ct098_f = -7.2275380e-06;
constexpr float Cl099_f = 5.7292175e-01;	constexpr float Ct099_f = -1.9993679e-06;
constexpr float Cl100_f = 5.7731628e-01;	constexpr float Ct100_f = -9.1914486e-07;
constexpr float Cl101_f = 5.8169556e-01;	constexpr float Ct101_f = -3.8170060e-06;
constexpr float Cl102_f = 5.8604431e-01;	constexpr float Ct102_f = +4.7334801e-06;
constexpr float Cl103_f = 5.9039307e-01;	constexpr float Ct103_f = -5.6198041e-06;
constexpr float Cl104_f = 5.9471130e-01;	constexpr float Ct104_f = -4.1959642e-06;
constexpr float Cl105_f = 5.9901428e-01;	constexpr float Ct105_f = -6.0925805e-06;
constexpr float Cl106_f = 6.0328674e-01;	constexpr float Ct106_f = +4.1082740e-06;
constexpr float Cl107_f = 6.0755920e-01;	constexpr float Ct107_f = -3.9538770e-06;
constexpr float Cl108_f = 6.1180115e-01;	constexpr float Ct108_f = +3.9364506e-07;
constexpr float Cl109_f = 6.1602783e-01;	constexpr float Ct109_f = +2.0451843e-06;
constexpr float Cl110_f = 6.2023926e-01;	constexpr float Ct110_f = +1.1519394e-06;
constexpr float Cl111_f = 6.2443542e-01;	constexpr float Ct111_f = -2.1367928e-06;
constexpr float Cl112_f = 6.2860107e-01;	constexpr float Ct112_f = +7.5852036e-06;
constexpr float Cl113_f = 6.3276672e-01;	constexpr float Ct113_f = -5.4061775e-08;
constexpr float Cl114_f = 6.3690186e-01;	constexpr float Ct114_f = +5.6067683e-06;
constexpr float Cl115_f = 6.4103699e-01;	constexpr float Ct115_f = -5.8078838e-06;
constexpr float Cl116_f = 6.4514160e-01;	constexpr float Ct116_f = -3.6401889e-06;
constexpr float Cl117_f = 6.4923096e-01;	constexpr float Ct117_f = -3.0104061e-06;
constexpr float Cl118_f = 6.5330505e-01;	constexpr float Ct118_f = -3.7816982e-06;
constexpr float Cl119_f = 6.5736389e-01;	constexpr float Ct119_f = -5.8188932e-06;
constexpr float Cl120_f = 6.6139221e-01;	constexpr float Ct120_f = +6.2703313e-06;
constexpr float Cl121_f = 6.6542053e-01;	constexpr float Ct121_f = +2.1003185e-06;
constexpr float Cl122_f = 6.6943359e-01;	constexpr float Ct122_f = -2.9398074e-06;
constexpr float Cl123_f = 6.7341614e-01;	constexpr float Ct123_f = +6.5375169e-06;
constexpr float Cl124_f = 6.7739868e-01;	constexpr float Ct124_f = +1.4195118e-07;
constexpr float Cl125_f = 6.8136597e-01;	constexpr float Ct125_f = -6.7419890e-06;
constexpr float Cl126_f = 6.8530273e-01;	constexpr float Ct126_f = +1.2687239e-06;
constexpr float Cl127_f = 6.8923950e-01;	constexpr float Ct127_f = -6.2207143e-06;
constexpr float Cl128_f = 6.9314575e-01;	constexpr float Ct128_f = +1.4286068e-06;

constexpr float lgTrl_f = 2.34037874236037e-06;

/*	For the trigonometric functions	*/

constexpr float PiA_f = -3.140625f;
constexpr float PiB_f = -0.0009670257568359375f;
constexpr float PiC_f = -6.2771141529083251953e-07f;
constexpr float PiD_f = -1.2154201256553420762e-10f;

constexpr float s0_f  =  2.6083159809786593541503e-06f;
constexpr float s1_f  = -0.0001981069071916863322258f;
constexpr float s2_f  =  0.00833307858556509017944336f;
constexpr float s3_f  = -0.166666597127914428710938f;

#ifdef	__AVX512F__
constexpr _MData_ oPif      = {     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,
				    1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI };
constexpr _MData_ zeroNegf  = {       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,
				      -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f };
constexpr _MData_ fHlf      = {        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,
				       0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f };
constexpr _MData_ TriMaxf   = {         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,
					1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7 };
constexpr _MData_ fInf      = {       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,
				      Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f };
constexpr _MData_ fNan      = {       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,
				      Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f };
constexpr _MData_ PiAf      = {       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,
				      PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f };
constexpr _MData_ PiBf      = {       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,
				      PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f };
constexpr _MData_ PiCf      = {       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,
				      PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f };
constexpr _MData_ PiDf      = {       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,
				      PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f };
constexpr _MData_ hPiAf     = {  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,
				 0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f };
constexpr _MData_ hPiBf     = {  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,
				 0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f };
constexpr _MData_ hPiCf     = {  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,
				 0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f };
constexpr _MData_ hPiDf     = {  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,
				 0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f };
constexpr _MData_ s0f       = {        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,
				       s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f };
constexpr _MData_ s1f       = {        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,
				       s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f };
constexpr _MData_ s2f       = {        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,
				       s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f };
constexpr _MData_ s3f       = {        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,
				       s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f };

constexpr _MData_ vTh1f     = {       th1_f,       th1_f,       th1_f,       th1_f,       th1_f,       th1_f,       th1_f,       th1_f,
				      th1_f,       th1_f,       th1_f,       th1_f,       th1_f,       th1_f,       th1_f,       th1_f };
constexpr _MData_ vTh2f     = {       th2_f,       th2_f,       th2_f,       th2_f,       th2_f,       th2_f,       th2_f,       th2_f,
				      th2_f,       th2_f,       th2_f,       th2_f,       th2_f,       th2_f,       th2_f,       th2_f };
constexpr _MData_ vL1f      = {        L1_f,        L1_f,        L1_f,        L1_f,        L1_f,        L1_f,        L1_f,        L1_f,
			 	       L1_f,        L1_f,        L1_f,        L1_f,        L1_f,        L1_f,        L1_f,        L1_f };
constexpr _MData_ vL2f      = {       -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f,
				      -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f };
constexpr _MData_ vA1f      = {        A1_f,        A1_f,        A1_f,        A1_f,        A1_f,        A1_f,        A1_f,        A1_f,
			 	       A1_f,        A1_f,        A1_f,        A1_f,        A1_f,        A1_f,        A1_f,        A1_f };
constexpr _MData_ vA2f      = {        A2_f,        A2_f,        A2_f,        A2_f,        A2_f,        A2_f,        A2_f,        A2_f,
				       A2_f,        A2_f,        A2_f,        A2_f,        A2_f,        A2_f,        A2_f,        A2_f };
constexpr _MData_ vExpf     = {       mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f,
				      mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f };
constexpr _MData_ vIvLf     = {       ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f,
				      ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f };
constexpr _MData_ vi2t7f    = {     1./128.,     1./128.,     1./128.,     1./128.,     1./128.,     1./128.,     1./128.,     1./128.,
				    1./128.,     1./128.,     1./128.,     1./128.,     1./128.,     1./128.,     1./128.,     1./128. };
constexpr _MData_ vB1f      = {        B1_f,        B1_f,        B1_f,        B1_f,        B1_f,        B1_f,        B1_f,        B1_f,
			 	       B1_f,        B1_f,        B1_f,        B1_f,        B1_f,        B1_f,        B1_f,        B1_f };
constexpr _MData_ vB2f      = {        B2_f,        B2_f,        B2_f,        B2_f,        B2_f,        B2_f,        B2_f,        B2_f,
				       B2_f,        B2_f,        B2_f,        B2_f,        B2_f,        B2_f,        B2_f,        B2_f };
constexpr _MData_ vLg2lf    = {     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,
				    Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f };
constexpr _MData_ vLg2tf    = {     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,
				    Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f };
constexpr _MData_ nmDnf     = {   8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f,
				  8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f };
constexpr _MData_ lgDlf     = { 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f,
				15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f };
constexpr _MData_ lgDtf     = {     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,
				    lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f };
#elif   defined(__AVX__)
constexpr _MData_ oPif      = {     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI };
constexpr _MData_ zeroNegf  = {       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f };
constexpr _MData_ fHlf       = {        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f };
#ifdef	__FMA__
constexpr _MData_ TriMaxf   = {         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7 };
#else
constexpr _MData_ TriMaxf   = {         1e5,         1e5,         1e5,         1e5,         1e5,         1e5,         1e5,         1e5 };
#endif
constexpr _MData_ fInf      = {       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f };
constexpr _MData_ fNan      = {       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f };
constexpr _MData_ PiAf      = {       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f };
constexpr _MData_ PiBf      = {       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f };
constexpr _MData_ PiCf      = {       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f };
constexpr _MData_ PiDf      = {       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f };
constexpr _MData_ hPiAf     = {  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f };
constexpr _MData_ hPiBf     = {  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f };
constexpr _MData_ hPiCf     = {  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f };
constexpr _MData_ hPiDf     = {  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f };
constexpr _MData_ s0f       = {        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f };
constexpr _MData_ s1f       = {        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f };
constexpr _MData_ s2f       = {        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f };
constexpr _MData_ s3f       = {        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f };

constexpr _MData_ vTh1f     = {       th1_f,       th1_f,       th1_f,       th1_f,       th1_f,       th1_f,       th1_f,       th1_f };
constexpr _MData_ vTh2f     = {       th2_f,       th2_f,       th2_f,       th2_f,       th2_f,       th2_f,       th2_f,       th2_f };
constexpr _MData_ vL1f      = {        L1_f,        L1_f,        L1_f,        L1_f,        L1_f,        L1_f,        L1_f,        L1_f };
constexpr _MData_ vL2f      = {       -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f,       -L2_f };
constexpr _MData_ vA1f      = {        A1_f,        A1_f,        A1_f,        A1_f,        A1_f,        A1_f,        A1_f,        A1_f };
constexpr _MData_ vA2f      = {        A2_f,        A2_f,        A2_f,        A2_f,        A2_f,        A2_f,        A2_f,        A2_f };
constexpr _MData_ vExpf     = {       mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f,       mEx_f };
constexpr _MData_ vIvLf     = {       ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f,       ivL_f };
constexpr _MData_ vi2t7f    = {   1.f/128.f,   1.f/128.f,   1.f/128.f,   1.f/128.f,   1.f/128.f,   1.f/128.f,   1.f/128.f,   1.f/128.f };
constexpr _MData_ vB1f      = {        B1_f,        B1_f,        B1_f,        B1_f,        B1_f,        B1_f,        B1_f,        B1_f };
constexpr _MData_ vB2f      = {        B2_f,        B2_f,        B2_f,        B2_f,        B2_f,        B2_f,        B2_f,        B2_f };
constexpr _MData_ vLg2lf    = {     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f };
constexpr _MData_ vLg2tf    = {     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f };
constexpr _MData_ nmDnf     = {   8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f,   8388608.f };
constexpr _MData_ lgDlf     = { 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f };
constexpr _MData_ lgDtf     = {     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f,     lgTrl_f };
#else
constexpr _MData_ oPif      = {     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI };
constexpr _MData_ zeroNegf  = {       -0.0f,       -0.0f,       -0.0f,       -0.0f };
constexpr _MData_ fHlf      = {        0.5f,        0.5f,        0.5f,        0.5f };
constexpr _MData_ TriMaxf   = {         1e5,         1e5,         1e5,         1e5 };
constexpr _MData_ fInf      = {       Inf_f,       Inf_f,       Inf_f,       Inf_f };
constexpr _MData_ PiAf      = {       PiA_f,       PiA_f,       PiA_f,       PiA_f };
constexpr _MData_ PiBf      = {       PiB_f,       PiB_f,       PiB_f,       PiB_f };
constexpr _MData_ PiCf      = {       PiC_f,       PiC_f,       PiC_f,       PiC_f };
constexpr _MData_ PiDf      = {       PiD_f,       PiD_f,       PiD_f,       PiD_f };
constexpr _MData_ hPiAf     = {  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f };
constexpr _MData_ hPiBf     = {  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f };
constexpr _MData_ hPiCf     = {  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f };
constexpr _MData_ hPiDf     = {  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f };
constexpr _MData_ s0f       = {        s0_f,        s0_f,        s0_f,        s0_f };
constexpr _MData_ s1f       = {        s1_f,        s1_f,        s1_f,        s1_f };
constexpr _MData_ s2f       = {        s2_f,        s2_f,        s2_f,        s2_f };
constexpr _MData_ s3f       = {        s3_f,        s3_f,        s3_f,        s3_f };

constexpr _MData_ vTh1f     = {       th1_f,       th1_f,       th1_f,       th1_f };
constexpr _MData_ vTh2f     = {       th2_f,       th2_f,       th2_f,       th2_f };
constexpr _MData_ vL1f      = {        L1_f,        L1_f,        L1_f,        L1_f };
constexpr _MData_ vL2f      = {       -L2_f,       -L2_f,       -L2_f,       -L2_f };
constexpr _MData_ vA1f      = {        A1_f,        A1_f,        A1_f,        A1_f };
constexpr _MData_ vA2f      = {        A2_f,        A2_f,        A2_f,        A2_f };
constexpr _MData_ vExpf     = {       mEx_f,       mEx_f,       mEx_f,       mEx_f };
constexpr _MData_ vIvLf     = {       ivL_f,       ivL_f,       ivL_f,       ivL_f };
constexpr _MData_ vi2t7f    = {     1./128.,     1./128.,     1./128.,     1./128. };
constexpr _MData_ vB1f      = {        B1_f,        B1_f,        B1_f,        B1_f };
constexpr _MData_ vB2f      = {        B2_f,        B2_f,        B2_f,        B2_f };
constexpr _MData_ vLg2lf    = {     Cl128_f,     Cl128_f,     Cl128_f,     Cl128_f };
constexpr _MData_ vLg2tf    = {     Ct128_f,     Ct128_f,     Ct128_f,     Ct128_f };
constexpr _MData_ nmDnf     = {   8388608.f,   8388608.f,   8388608.f,   8388608.f };
constexpr _MData_ lgDlf     = { 15.9423828f, 15.9423828f, 15.9423828f, 15.9423828f };
constexpr _MData_ lgDtf     = {     lgTrl_f,      lgTrlf,      lgTrlf,      lgTrlf };
#endif

constexpr std::array<float,   32> sTrail_f = { St00_f, St01_f, St02_f, St03_f, St04_f, St05_f, St06_f, St07_f,
					       St08_f, St09_f, St10_f, St11_f, St12_f, St13_f, St14_f, St15_f,
					       St16_f, St17_f, St18_f, St19_f, St20_f, St21_f, St22_f, St23_f,
					       St24_f, St25_f, St26_f, St27_f, St28_f, St29_f, St30_f, St31_f };

constexpr std::array<float,   32> sLead_f  = { Sl00_f, Sl01_f, Sl02_f, Sl03_f, Sl04_f, Sl05_f, Sl06_f, Sl07_f,
					       Sl08_f, Sl09_f, Sl10_f, Sl11_f, Sl12_f, Sl13_f, Sl14_f, Sl15_f,
					       Sl16_f, Sl17_f, Sl18_f, Sl19_f, Sl20_f, Sl21_f, Sl22_f, Sl23_f,
					       Sl24_f, Sl25_f, Sl26_f, Sl27_f, Sl28_f, Sl29_f, Sl30_f, Sl31_f };

constexpr std::array<float,  129> cTrail_f = { Ct000_f, Ct001_f, Ct002_f, Ct003_f, Ct004_f, Ct005_f, Ct006_f, Ct007_f, Ct008_f, Ct009_f, Ct010_f, Ct011_f, Ct012_f, Ct013_f, Ct014_f,
					       Ct015_f, Ct016_f, Ct017_f, Ct018_f, Ct019_f, Ct020_f, Ct021_f, Ct022_f, Ct023_f, Ct024_f, Ct025_f, Ct026_f, Ct027_f, Ct028_f, Ct029_f,
					       Ct030_f, Ct031_f, Ct032_f, Ct033_f, Ct034_f, Ct035_f, Ct036_f, Ct037_f, Ct038_f, Ct039_f, Ct040_f, Ct041_f, Ct042_f, Ct043_f, Ct044_f,
					       Ct045_f, Ct046_f, Ct047_f, Ct048_f, Ct049_f, Ct050_f, Ct051_f, Ct052_f, Ct053_f, Ct054_f, Ct055_f, Ct056_f, Ct057_f, Ct058_f, Ct059_f,
					       Ct060_f, Ct061_f, Ct062_f, Ct063_f, Ct064_f, Ct065_f, Ct066_f, Ct067_f, Ct068_f, Ct069_f, Ct070_f, Ct071_f, Ct072_f, Ct073_f, Ct074_f,
					       Ct075_f, Ct076_f, Ct077_f, Ct078_f, Ct079_f, Ct080_f, Ct081_f, Ct082_f, Ct083_f, Ct084_f, Ct085_f, Ct086_f, Ct087_f, Ct088_f, Ct089_f,
					       Ct090_f, Ct091_f, Ct092_f, Ct093_f, Ct094_f, Ct095_f, Ct096_f, Ct097_f, Ct098_f, Ct099_f, Ct100_f, Ct101_f, Ct102_f, Ct103_f, Ct104_f,
					       Ct105_f, Ct106_f, Ct107_f, Ct108_f, Ct109_f, Ct110_f, Ct111_f, Ct112_f, Ct113_f, Ct114_f, Ct115_f, Ct116_f, Ct117_f, Ct118_f, Ct119_f,
					       Ct120_f, Ct121_f, Ct122_f, Ct123_f, Ct124_f, Ct125_f, Ct126_f, Ct127_f, Ct128_f };

constexpr std::array<float,  129> cLead_f  = { Cl000_f, Cl001_f, Cl002_f, Cl003_f, Cl004_f, Cl005_f, Cl006_f, Cl007_f, Cl008_f, Cl009_f, Cl010_f, Cl011_f, Cl012_f, Cl013_f, Cl014_f,
					       Cl015_f, Cl016_f, Cl017_f, Cl018_f, Cl019_f, Cl020_f, Cl021_f, Cl022_f, Cl023_f, Cl024_f, Cl025_f, Cl026_f, Cl027_f, Cl028_f, Cl029_f,
					       Cl030_f, Cl031_f, Cl032_f, Cl033_f, Cl034_f, Cl035_f, Cl036_f, Cl037_f, Cl038_f, Cl039_f, Cl040_f, Cl041_f, Cl042_f, Cl043_f, Cl044_f,
					       Cl045_f, Cl046_f, Cl047_f, Cl048_f, Cl049_f, Cl050_f, Cl051_f, Cl052_f, Cl053_f, Cl054_f, Cl055_f, Cl056_f, Cl057_f, Cl058_f, Cl059_f,
					       Cl060_f, Cl061_f, Cl062_f, Cl063_f, Cl064_f, Cl065_f, Cl066_f, Cl067_f, Cl068_f, Cl069_f, Cl070_f, Cl071_f, Cl072_f, Cl073_f, Cl074_f,
					       Cl075_f, Cl076_f, Cl077_f, Cl078_f, Cl079_f, Cl080_f, Cl081_f, Cl082_f, Cl083_f, Cl084_f, Cl085_f, Cl086_f, Cl087_f, Cl088_f, Cl089_f,
					       Cl090_f, Cl091_f, Cl092_f, Cl093_f, Cl094_f, Cl095_f, Cl096_f, Cl097_f, Cl098_f, Cl099_f, Cl100_f, Cl101_f, Cl102_f, Cl103_f, Cl104_f,
					       Cl105_f, Cl106_f, Cl107_f, Cl108_f, Cl109_f, Cl110_f, Cl111_f, Cl112_f, Cl113_f, Cl114_f, Cl115_f, Cl116_f, Cl117_f, Cl118_f, Cl119_f,
					       Cl120_f, Cl121_f, Cl122_f, Cl123_f, Cl124_f, Cl125_f, Cl126_f, Cl127_f, Cl128_f };

constexpr std::array<double,  32> sTrail_d = { St00_d, St01_d, St02_d, St03_d, St04_d, St05_d, St06_d, St07_d,
					       St08_d, St09_d, St10_d, St11_d, St12_d, St13_d, St14_d, St15_d,
					       St16_d, St17_d, St18_d, St19_d, St20_d, St21_d, St22_d, St23_d,
					       St24_d, St25_d, St26_d, St27_d, St28_d, St29_d, St30_d, St31_d };

constexpr std::array<double,  32> sLead_d  = { Sl00_d, Sl01_d, Sl02_d, Sl03_d, Sl04_d, Sl05_d, Sl06_d, Sl07_d,
					       Sl08_d, Sl09_d, Sl10_d, Sl11_d, Sl12_d, Sl13_d, Sl14_d, Sl15_d,
					       Sl16_d, Sl17_d, Sl18_d, Sl19_d, Sl20_d, Sl21_d, Sl22_d, Sl23_d,
					       Sl24_d, Sl25_d, Sl26_d, Sl27_d, Sl28_d, Sl29_d, Sl30_d, Sl31_d };

constexpr std::array<double, 129> cTrail_d = { Ct000_d, Ct001_d, Ct002_d, Ct003_d, Ct004_d, Ct005_d, Ct006_d, Ct007_d, Ct008_d, Ct009_d, Ct010_d, Ct011_d, Ct012_d, Ct013_d, Ct014_d,
					       Ct015_d, Ct016_d, Ct017_d, Ct018_d, Ct019_d, Ct020_d, Ct021_d, Ct022_d, Ct023_d, Ct024_d, Ct025_d, Ct026_d, Ct027_d, Ct028_d, Ct029_d,
					       Ct030_d, Ct031_d, Ct032_d, Ct033_d, Ct034_d, Ct035_d, Ct036_d, Ct037_d, Ct038_d, Ct039_d, Ct040_d, Ct041_d, Ct042_d, Ct043_d, Ct044_d,
					       Ct045_d, Ct046_d, Ct047_d, Ct048_d, Ct049_d, Ct050_d, Ct051_d, Ct052_d, Ct053_d, Ct054_d, Ct055_d, Ct056_d, Ct057_d, Ct058_d, Ct059_d,
					       Ct060_d, Ct061_d, Ct062_d, Ct063_d, Ct064_d, Ct065_d, Ct066_d, Ct067_d, Ct068_d, Ct069_d, Ct070_d, Ct071_d, Ct072_d, Ct073_d, Ct074_d,
					       Ct075_d, Ct076_d, Ct077_d, Ct078_d, Ct079_d, Ct080_d, Ct081_d, Ct082_d, Ct083_d, Ct084_d, Ct085_d, Ct086_d, Ct087_d, Ct088_d, Ct089_d,
					       Ct090_d, Ct091_d, Ct092_d, Ct093_d, Ct094_d, Ct095_d, Ct096_d, Ct097_d, Ct098_d, Ct099_d, Ct100_d, Ct101_d, Ct102_d, Ct103_d, Ct104_d,
					       Ct105_d, Ct106_d, Ct107_d, Ct108_d, Ct109_d, Ct110_d, Ct111_d, Ct112_d, Ct113_d, Ct114_d, Ct115_d, Ct116_d, Ct117_d, Ct118_d, Ct119_d,
					       Ct120_d, Ct121_d, Ct122_d, Ct123_d, Ct124_d, Ct125_d, Ct126_d, Ct127_d, Ct128_d };

constexpr std::array<double, 129> cLead_d  = { Cl000_d, Cl001_d, Cl002_d, Cl003_d, Cl004_d, Cl005_d, Cl006_d, Cl007_d, Cl008_d, Cl009_d, Cl010_d, Cl011_d, Cl012_d, Cl013_d, Cl014_d,
					       Cl015_d, Cl016_d, Cl017_d, Cl018_d, Cl019_d, Cl020_d, Cl021_d, Cl022_d, Cl023_d, Cl024_d, Cl025_d, Cl026_d, Cl027_d, Cl028_d, Cl029_d,
					       Cl030_d, Cl031_d, Cl032_d, Cl033_d, Cl034_d, Cl035_d, Cl036_d, Cl037_d, Cl038_d, Cl039_d, Cl040_d, Cl041_d, Cl042_d, Cl043_d, Cl044_d,
					       Cl045_d, Cl046_d, Cl047_d, Cl048_d, Cl049_d, Cl050_d, Cl051_d, Cl052_d, Cl053_d, Cl054_d, Cl055_d, Cl056_d, Cl057_d, Cl058_d, Cl059_d,
					       Cl060_d, Cl061_d, Cl062_d, Cl063_d, Cl064_d, Cl065_d, Cl066_d, Cl067_d, Cl068_d, Cl069_d, Cl070_d, Cl071_d, Cl072_d, Cl073_d, Cl074_d,
					       Cl075_d, Cl076_d, Cl077_d, Cl078_d, Cl079_d, Cl080_d, Cl081_d, Cl082_d, Cl083_d, Cl084_d, Cl085_d, Cl086_d, Cl087_d, Cl088_d, Cl089_d,
					       Cl090_d, Cl091_d, Cl092_d, Cl093_d, Cl094_d, Cl095_d, Cl096_d, Cl097_d, Cl098_d, Cl099_d, Cl100_d, Cl101_d, Cl102_d, Cl103_d, Cl104_d,
					       Cl105_d, Cl106_d, Cl107_d, Cl108_d, Cl109_d, Cl110_d, Cl111_d, Cl112_d, Cl113_d, Cl114_d, Cl115_d, Cl116_d, Cl117_d, Cl118_d, Cl119_d,
					       Cl120_d, Cl121_d, Cl122_d, Cl123_d, Cl124_d, Cl125_d, Cl126_d, Cl127_d, Cl128_d };

#endif

/*	Sleef	*/
inline void printiVar(_MInt_ d, const char *name) {

	printf ("%s", name);
#if	defined(__AVX512F__)
	int r[16] __attribute((aligned(64)));
	opCode(store_si512, r, d);
	for (int i=0; i<16; i++)
#elif	defined(__AVX__)
	int r[8] __attribute((aligned(32)));
	opCode(store_si256, ((_MInt_ *)r), d);
	for (int i=0; i<8; i++)
#else
	int r[4] __attribute((aligned(16)));
	opCode(store_si128, ((_MInt_ *)r), d);
	for (int i=0; i<4; i++)
#endif
		printf(" %d", r[i]);
	printf("\n");
}

inline void printuVar(_MInt_ d, const char *name) {

	printf ("%s", name);
#if	defined(__AVX512F__)
	unsigned int r[16] __attribute((aligned(64)));
	opCode(store_si512, r, d);
	for (int i=0; i<16; i++)
#elif	defined(__AVX__)
	unsigned int r[8] __attribute((aligned(32)));
	opCode(store_si256, ((_MInt_ *)r), d);
	for (int i=0; i<8; i++)
#else
	unsigned int r[4] __attribute((aligned(16)));
	opCode(store_si128, ((_MInt_ *)r), d);
	for (int i=0; i<4; i++)
#endif
		printf(" %u", r[i]);
	printf("\n");
}

inline void printsVar(_MData_ d, const char *name) {

	printf ("%s", name);
#if	defined(__AVX512F__)
	for (int i=0; i<16; i++)
#elif	defined(__AVX__)
	for (int i=0; i<8; i++)
#else
	for (int i=0; i<4; i++)
#endif
		printf(" %f", d[i]);
	printf("\n");
}

#undef	_MData_
#undef	_PREFIX
#undef opCode_P
#undef opCode_N
#undef opCode

#endif
