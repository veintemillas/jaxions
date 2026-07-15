#include "J0tabler.h"


template<typename Float, SpectrumMaskType mask>
void SpecBin::nRun_2D(nRunType nrt)
{
    switch (fType) {

        case FIELD_SAXION:
        {

            auto &myPlan = AxionFFT::fetchPlan("spec1Dm2");
            auto &myTran = AxionFFT::fetchPlan("transpose");

            if (!(nrt & NRUN_K))
                return;

            constexpr double twopi = 2.0 * M_PI;

            auto *m  = static_cast<Float *>(field->mStart());
            auto *v  = static_cast<Float *>(field->vCpu());
            auto *m2 = static_cast<Float *>(field->m2Cpu());

            const size_t Nz       = field->NX();   // z transform length
            const size_t NrLocal  = field->NZ();   // local rho
            const size_t NrGlobal = field->TZ();   // global rho

            const size_t NkzLocal = NrLocal;
            const size_t kzOffset = NrLocal*commRank();
            const double delta = field->Delta();

            const double dkz = M_PI / (double(Nz) * delta);
            const double dkp = M_PI / (double(NrGlobal) * delta); // or your chosen rhoMax convention

            auto bessel0_table = getJ0Table<Float>(NrGlobal, NkpLocal, kpOffset, 1, dkp, commRank());

            /*
             * 1. Build thetadot(z,rho), sine-transform along z.
             *
             * Before transpose:
             *
             *     m2[irho_local*Nz + ikz]
             *
             * After transpose:
             *
             *     m2[ikz_local*NrGlobal + irho_global]
             */

            for (size_t irho = 0; irho < NrLocal; ++irho) {

                /*
                 * Use temporary m2[0 ... Nz-1].
                 *
                 * RODFT10 corresponds roughly to half-cell odd extension.
                 * If you include z=0, make sure theta_dot(0)=0.
                 */

                for (size_t iz = 0; iz < Nz; ++iz) {

                    if (iz == 0)
                    {
                        m2[iz] = (Float) 0.0;
                        continue;
                    }
                    const size_t id = irho*Nz+iz;

                    const Float phir = m[2*id    ];
                    const Float phii = m[2*id + 1];

                    const Float vr = v[2*id    ];
                    const Float vi = v[2*id + 1];

                    const Float mod2 = phir*phir + phii*phii;

                    Float thetaDot = Float(0);

                    if (mod2 > Float(0)) {
                        /*
                         * Im(dotphi * phi^*) / |phi|^2
                         *
                         * dotphi * phi^* = (vr+i vi)(phir - i phii)
                         * Im = vi*phir - vr*phii
                         */
                        thetaDot = (vi*phir - vr*phii) / mod2;
                    } else {
                        thetaDot = (vi*phir - vr*phii) ;
                    }

                    m2[iz] = thetaDot;
                }

                myPlan.run(FFT_FWD);

                /*
                 * Store transformed line into its rho slot.
                 * Since m2[0:Nz] was the temporary line, copy to m2[irho*Nz:Nz].
                 */

                memmove(&m2[irho*Nz],&m2[0],sizeof(Float)*Nz);
            }

            /*
             * 2. Transpose:
             *
             *   m2[irho_local][ikz_all]
             *        ->
             *   m2[ikz_local][irho_global]
             */

            myTran.run(FFT_FWD);



            std::fill(bin_rho.begin(), bin_rho.end(), 0.0);

            /*
             * 3. Bessel transform and binning.
             *
             * Since we only use thetadot and later time-average:
             *
             *   n_k   = fA^2 / omega * |thetadot_k|^2
             *   rho_k = fA^2         * |thetadot_k|^2
             */

            for (size_t ikzLoc = 0; ikzLoc < NkzLocal; ++ikzLoc) {

                const size_t ikzGlob = ikzLoc + kzOffset;
                const double kz = dkz * double(ikzGlob + 1);

                for (size_t ikpLoc = 0; ikpLoc < NkpLocal; ++ikpLoc) {

                    const size_t ikpGlob = ikpLoc + kpOffset;
                    const double kp = dkp * double(ikpGlob);
                    const double k  = std::sqrt(kz*kz + kp*kp);

                    double td_k = 0.0;

                    for (size_t ir = 0; ir < NrGlobal; ++ir) {
                        const Float B = bessel0_table[ikpLoc*NrGlobal + ir];
                        td_k += double(B) * double(m2[ikzLoc*NrGlobal + ir]);
                    }

                    td_k *= twopi * delta * delta * delta;

                    const double td2 = td_k * td_k;

                    const int ib = kbin(k);

                    if (ib >= 0 && ib < nBins) {
                        bin_rho[ib] += kp * td2;
                    }
                }
            }

            MPI_Allreduce(MPI_IN_PLACE, bin_n.data(),
                          nBins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            MPI_Allreduce(MPI_IN_PLACE, bin_rho.data(),
                          nBins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            MPI_Allreduce(MPI_IN_PLACE, bin_pth.data(),
                          nBins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            const double norm = dkp * dkz / (4.0 * M_PI * M_PI);

            for (int ib = 0; ib < nBins; ++ib) {
                drho_dlogk[ib] = norm * bin_rho[ib] / dlogk;
            }

        
            field->setM2(M2_DIRTY);
        }
        break;

        case FIELD_WKB:
        default:
            LogError("Error: Field not supported");
            return;
    }
}