#ifndef _J0_TABLER_
#define _J0_TABLER_

#include <vector>
#include <cstdio>
#include <cstdint>
#include <cmath>

struct J0ProductHeader {
    uint64_t magic;
    uint64_t version;
    uint64_t NrGlobal;
    uint64_t cacheSize;
    uint64_t sizeofFloat;
};

template<typename Float>
bool loadJ0ProductCache(const char *fname, Float *cache,
                        size_t NrGlobal, size_t cacheSize)
{
    FILE *fp = fopen(fname, "rb");
    if (!fp) return false;

    J0ProductHeader h;
    const uint64_t magic = 0x4A3050524F445543ULL; // "J0PRODUC"
    const bool headerRead = fread(&h, sizeof(h), 1, fp) == 1;
    const bool valid = headerRead && h.magic == magic && h.version == 1 &&
        h.NrGlobal == NrGlobal && h.cacheSize == cacheSize &&
        h.sizeofFloat == sizeof(Float);
    const bool dataRead = valid &&
        fread(cache, sizeof(Float), cacheSize, fp) == cacheSize;
    fclose(fp);
    return dataRead;
}

template<typename Float>
bool saveJ0ProductCache(const char *fname, const Float *cache,
                        size_t NrGlobal, size_t cacheSize)
{
    FILE *fp = fopen(fname, "wb");
    if (!fp) return false;

    const J0ProductHeader h = {
        0x4A3050524F445543ULL, 1, uint64_t(NrGlobal), uint64_t(cacheSize),
        uint64_t(sizeof(Float))
    };
    const bool written = fwrite(&h, sizeof(h), 1, fp) == 1 &&
        fwrite(cache, sizeof(Float), cacheSize, fp) == cacheSize;
    fclose(fp);
    return written;
}

template<typename Float>
struct J0Header {
    uint64_t magic;
    uint64_t version;
    uint64_t NrGlobal;
    uint64_t NkpLocal;
    uint64_t kpOffset;
    double dr;
    double dkp;
    uint64_t sizeofFloat;
};

template<typename Float>
bool loadJ0Table(const char *fname,
                 std::vector<Float> &B,
                 size_t NrGlobal,
                 size_t NkpLocal,
                 size_t kpOffset,
                 double dr,
                 double dkp)
{
    FILE *fp = fopen(fname, "rb");
    if (!fp) return false;

    J0Header<Float> h;
    if (fread(&h, sizeof(h), 1, fp) != 1) {
        fclose(fp);
        return false;
    }

    const uint64_t magic = 0xB355E10B355E10ULL;

    bool ok =
        h.magic       == magic &&
        h.version     == 1 &&
        h.NrGlobal    == NrGlobal &&
        h.NkpLocal    == NkpLocal &&
        h.kpOffset    == kpOffset &&
        h.sizeofFloat == sizeof(Float) &&
        h.dr          == dr &&
        h.dkp         == dkp;

    if (!ok) {
        fclose(fp);
        return false;
    }

    B.resize(NkpLocal * NrGlobal);

    size_t nread = fread(B.data(), sizeof(Float), B.size(), fp);
    fclose(fp);

    return nread == B.size();
}

template<typename Float>
void saveJ0Table(const char *fname,
                 const std::vector<Float> &B,
                 size_t NrGlobal,
                 size_t NkpLocal,
                 size_t kpOffset,
                 double dr,
                 double dkp)
{
    FILE *fp = fopen(fname, "wb");
    if (!fp) return;

    J0Header<Float> h;
    h.magic       = 0xB355E10B355E10ULL;
    h.version     = 1;
    h.NrGlobal    = NrGlobal;
    h.NkpLocal    = NkpLocal;
    h.kpOffset    = kpOffset;
    h.dr          = dr;
    h.dkp         = dkp;
    h.sizeofFloat = sizeof(Float);

    fwrite(&h, sizeof(h), 1, fp);
    fwrite(B.data(), sizeof(Float), B.size(), fp);

    fclose(fp);
}

template<typename Float>
void buildJ0Table(std::vector<Float> &B,
                  size_t NrGlobal,
                  size_t NkpLocal,
                  size_t kpOffset,
                  double dr,
                  double dkp)
{
    B.resize(NkpLocal * NrGlobal);

    for (size_t ikpLoc = 0; ikpLoc < NkpLocal; ++ikpLoc) {

        const size_t ikpGlob = ikpLoc + kpOffset;
        const double kp = double(ikpGlob) * dkp;

        for (size_t ir = 0; ir < NrGlobal; ++ir) {

            const double rho = double(ir) * dr;
            B[ikpLoc * NrGlobal + ir] =
                Float(rho * ::j0(kp * rho));
        }
    }
}

template<typename Float>
std::vector<Float> getJ0Table(size_t NrGlobal,
                              size_t NkpLocal,
                              size_t kpOffset,
                              double dr,
                              double dkp,
                              int rank)
{
    char fname[256];
    snprintf(fname, sizeof(fname), "out/J0.rank%d.tab", rank);

    std::vector<Float> B;

    if (!loadJ0Table(fname, B, NrGlobal, NkpLocal, kpOffset, dr, dkp)) {
        buildJ0Table(B, NrGlobal, NkpLocal, kpOffset, dr, dkp);
        saveJ0Table(fname, B, NrGlobal, NkpLocal, kpOffset, dr, dkp);
    }

    return B;
}

#endif
