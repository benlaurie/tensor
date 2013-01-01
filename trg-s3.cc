/* -*- indent-tabs-mode: nil -*- */

#include <stdint.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

#include "tensor.h"

void Make9j(DTensor9 *K) {
  DTensor3 PP;
  PP.Set(0, 0, 0, 1);
  PP.Set(0, 1, 1, 1);
  PP.Set(0, 2, 2, 1/sqrt(2));
  PP.Set(0, 3, 3, 1/sqrt(2));
  PP.Set(1, 0, 1, 1);
  PP.Set(1, 1, 0, 1);
  PP.Set(1, 2, 3, -1/sqrt(2));
  PP.Set(1, 3, 2, 1/sqrt(2));
  PP.Set(2, 0, 2, 1/sqrt(2));
  PP.Set(2, 1, 3, -1/sqrt(2));
  PP.Set(2, 2, 0, 1/sqrt(2));
  PP.Set(2, 2, 2, -1./2);
  PP.Set(2, 3, 1, 1/sqrt(2));
  PP.Set(2, 3, 3, 1./2);
  PP.Set(3, 0, 3, 1/sqrt(2));
  PP.Set(3, 1, 2, 1/sqrt(2));
  PP.Set(3, 2, 1, -1/sqrt(2));
  PP.Set(3, 2, 3, 1./2);
  PP.Set(3, 3, 0, 1/sqrt(2));
  PP.Set(3, 3, 2, 1./2);

  uint8_t ind1[] = {0, 1, 2};
  uint8_t ind2[] = {0, 1, 3};

#define F(x) \
  for (uint8_t x = 0; x < 3; ++x) \
    for (uint8_t a##x = ind1[x]; a##x <= ind2[x]; ++a##x)

  F(i) F(j) F(k) F(l) F(m) F(n) F(o) F(p) F(q)
    K->Set(i, j, k, l, m, n, o, p, q,
        K->Get(i, j, k, l, m, n, o, p, q) +
        PP.Get(ai, aj, aq) * PP.Get(ak, al, aq) * PP.Get(an, am, ai) *
        PP.Get(ao, an, aj) * PP.Get(ap, ao, ak) * PP.Get(am, ap, al));
}

void MakeFirstBlocks(DTensor4 *B, double a, double b, double c) {
  double expr1 = 1 + a + 2 * b + 2 * c;
  double expr2 = pow(a, 2) + 4 * a * b + 4 * pow(b, 2) - pow(1 + 2 * c, 2);
  double expr3 = pow(a, 2) - 2 * a * b + pow(b, 2) + pow(-1 + c, 2);
  double expr4 = -1 + a + 2 * b - 2 * c;
  double expr5 = pow(a, 4) + 2 * b * pow(a, 3) + 4 * pow(b, 4) +
      pow(b, 2) * (-5 + 4 * c - 8 * pow(c, 2)) -
      2 * a * b * (1 + 2 * pow(b, 2) - 8 * c- 2 * pow(c, 2)) +
      pow(1 + c - 2 * pow(c, 2), 2) -
      pow(a, 2) * (2 + 3 * pow(b, 2) + 2 * c + 5 * pow(c, 2));
  double expr6 = pow(a, 4) - 2 * pow(b, 4) - 2 * b * pow((-1 + c), 3) -
      pow(b, 3) * (1 + 2 * c) - pow((-1 + c), 3) * (1 + 2 * c) +
      pow(a, 3) * (1 - b + 2 * c) - 3 * pow(a, 2) * b * (1 + b + 2 * c) +
      a * (5 * pow(b, 3) - pow(-1 + c, 3) + pow(b, 2) * (3 + 6 * c));
  double expr7 = pow(a, 4) - 4 * b * pow(a, 3) + 6 * pow(a, 2) * pow(b, 2) -
      4 * a * pow(b, 3) + pow(b, 4) + pow(-1 + c, 4);
  double expr8 = pow(a, 4) - 2 * pow(b, 4) -
      3 * b * pow(a, 2) * (-1 + b - 2 * c) + 2 * b * pow(-1 + c, 3) +
      pow(b, 3) * (1 + 2 * c) - pow(-1 + c, 3) * (1 + 2 * c) -
      pow(a, 3) * (1 + b + 2 * c) + a * (5 * pow(b, 3) + pow(-1 + c, 3) -
      3 * pow(b, 2) * (1 + 2 * c));
  double expr9 = pow(a, 2) - 2 * a * b + pow(b, 2) - pow(-1 + c, 2);

  B->Set(0, 0, 0, 0, pow(expr1, 4));
  B->Set(0, 0, 0, 1, pow(expr2, 2));
  B->Set(0, 0, 0, 2, 2 * expr3 * pow(expr1, 2));
  B->Set(0, 0, 1, 0, pow(expr2, 2));
  B->Set(0, 0, 1, 1, pow(expr4, 4));
  B->Set(0, 0, 1, 2, 2 * expr3 * pow(expr4, 2));
  B->Set(0, 0, 2, 0, 2 * expr3 * pow(expr1, 2));
  B->Set(0, 0, 2, 1, 2 * expr3 * pow(expr4, 2));
  B->Set(0, 0, 2, 2, 4 * pow(expr3, 2));

  B->Set(0, 2, 0, 0, 8 * pow(a - b, 2) * pow(-1 + c, 2));

  B->Set(1, 1, 0, 0, pow(expr2, 2));
  B->Set(1, 1, 0, 1, pow(expr2, 2));
  B->Set(1, 1, 0, 2, 2 * expr5);
  B->Set(1, 1, 1, 0, pow(expr2, 2));
  B->Set(1, 1, 1, 1, pow(expr2, 2));
  B->Set(1, 1, 1, 2, 2 * expr5);
  B->Set(1, 1, 2, 0, 2 * expr5);
  B->Set(1, 1, 2, 1, 2 * expr5);
  B->Set(1, 1, 2, 2, 4 * pow(expr9, 2));

  B->Set(2, 0, 0, 0, 8 * pow(a - b, 2) * pow(-1 + c, 2));

  B->Set(2, 2, 0, 0, expr3 * pow(expr1, 2));
  B->Set(2, 2, 0, 1, expr3 * pow(expr1, 2));
  B->Set(2, 2, 0, 2, sqrt(2) * expr6);
  B->Set(2, 2, 0, 3, expr5);
  B->Set(2, 2, 0, 4, expr5);
  B->Set(2, 2, 1, 0, expr3 * pow(expr1, 2));
  B->Set(2, 2, 1, 1, expr3 * pow(expr1, 2));
  B->Set(2, 2, 1, 2, sqrt(2) * expr6);
  B->Set(2, 2, 1, 3, expr5);
  B->Set(2, 2, 1, 4, expr5);
  B->Set(2, 2, 2, 0, sqrt(2) * expr6);
  B->Set(2, 2, 2, 1, sqrt(2) * expr6);
  B->Set(2, 2, 2, 2, 2 * expr7);
  B->Set(2, 2, 2, 3, sqrt(2) * expr8);
  B->Set(2, 2, 2, 4, sqrt(2) * expr8);
  B->Set(2, 2, 3, 0, expr5);
  B->Set(2, 2, 3, 1, expr5);
  B->Set(2, 2, 3, 2, sqrt(2) * expr8);
  B->Set(2, 2, 3, 3, expr3 * pow(expr4, 2));
  B->Set(2, 2, 3, 4, expr3 * pow(expr4, 2));
  B->Set(2, 2, 4, 0, expr5);
  B->Set(2, 2, 4, 1, expr5);
  B->Set(2, 2, 4, 2, sqrt(2) * expr8);
  B->Set(2, 2, 4, 3, expr3 * pow(expr4, 2));
  B->Set(2, 2, 4, 4, expr3 * pow(expr4, 2));
}

//FIXME: get rid of the ugly globals!
static gsl_vector *S[3][3];
static uint8_t dim[] = {1, 1, 2};

int CompareSVs(const void *a_, const void *b_) {
  const uint8_t *a = (const uint8_t *) a_;
  const uint8_t *b = (const uint8_t *) b_;
  double d = gsl_vector_get(S[b[0]][b[1]], b[2]) * dim[b[0]] * dim[b[1]];
  double c = gsl_vector_get(S[a[0]][a[1]], a[2]) * dim[a[0]] * dim[a[1]];
  if (c > d)
    return -1;
  else if (c == d)
    return 0;
  else
    return 1;
}

void DoFirstSVD(DTensor5 result[2], uint8_t sv_len[3][3], DTensor4 *B,
    const uint8_t dc, const double condi) {
  uint8_t sv_list[9*dc][3];
  uint8_t sv_num = 0;
  gsl_matrix *U[3][3];
  gsl_matrix *V[3][3];
  //FIXME: make GetGSLMatrix find size from tensor in general
  //(using msize works here)
  uint8_t msize[3][3] = {{3, 1, 1}, {1, 3, 1}, {1, 1, 5}};
  for (uint8_t rho_M = 0; rho_M < 3; ++rho_M)
    for (uint8_t rho_N = 0; rho_N < 3; ++rho_N) {
      sv_len[rho_M][rho_N] = 0;
      U[rho_M][rho_N] = B->GetGSLMatrix(rho_M, rho_N, 2, 3, msize[rho_M][rho_N],
          msize[rho_M][rho_N]);
      S[rho_M][rho_N] = gsl_vector_alloc(msize[rho_N][rho_M]);
      V[rho_M][rho_N] = gsl_matrix_alloc(msize[rho_N][rho_M],
          msize[rho_N][rho_M]);
      gsl_linalg_SV_decomp_jacobi(U[rho_M][rho_N], V[rho_M][rho_N],
          S[rho_M][rho_N]);
      for (uint8_t i = 0; i < std::min(msize[rho_M][rho_N], dc); ++i)
        if (fabs(gsl_vector_get(S[rho_M][rho_N], i)) > condi) {
          sv_list[sv_num][0] = rho_M;
          sv_list[sv_num][1] = rho_N;
          sv_list[sv_num][2] = i;
          ++sv_num;
        }
    }
  qsort(sv_list, sv_num, sizeof(sv_list[0]), CompareSVs);
  uint8_t rho_A[3][3][5] = {{{0, 1, 2}, {2}, {2}},
      {{2}, {0, 1, 2}, {2}}, {{2}, {2}, {0, 2, 2, 1, 2}}};
  uint8_t rho_B[3][3][5] = {{{0, 1, 2}, {2}, {2}},
      {{2}, {1, 0, 2}, {2}}, {{2}, {2}, {2, 0, 2, 2, 1}}};
  uint8_t rho_M;
  uint8_t rho_N;
  uint8_t i;
  double sv;
  double sv_max = gsl_vector_get(S[sv_list[0][0]][sv_list[0][1]],
      sv_list[0][2]);
  for (uint8_t n = 0; n < std::min(sv_num, dc); ++n) {
    rho_M = sv_list[n][0];
    rho_N = sv_list[n][1];
    i = sv_list[n][2];
    for (uint8_t m = 0; m < msize[rho_M][rho_N]; ++m) {
      sv = sqrt(gsl_vector_get(S[rho_M][rho_N], i) * dim[rho_M] * dim[rho_N] /
          sv_max);
      result[0].Set(i, rho_M, rho_N, rho_A[rho_M][rho_N][m],
          rho_B[rho_M][rho_N][m], sv * gsl_matrix_get(U[rho_M][rho_N], m, i));
      result[1].Set(i, rho_M, rho_N, rho_A[rho_M][rho_N][m],
          rho_B[rho_M][rho_N][m], sv * gsl_matrix_get(V[rho_M][rho_N], m, i));
    }
    ++sv_len[rho_M][rho_N];
  }
  for (uint8_t rho_M = 0; rho_M < 3; ++rho_M)
    for (uint8_t rho_N = 0; rho_N < 3; ++rho_N) {
      gsl_matrix_free(U[rho_M][rho_N]);
      gsl_vector_free(S[rho_M][rho_N]);
      gsl_matrix_free(V[rho_M][rho_N]);
    }
}

void DoFirstContraction(DTensor14 *C, const DTensor9 &K, const DTensor5 &SU,
    const DTensor5 &SV) {
  DTensor9 SUSU;
  // U00, U01, U02, U03+U14, U04, U10, U11, U12, U13
  Contract(&SUSU, SU, 3, SU, 4);
  DTensor17 KSUSU;
  // K00, K01+U01, K02, K03, K04, K05, K06, K07, K08, U00, U02, U03+U14, U04,
  // U10, U11, U12, U13
  Contract(&KSUSU, K, 1, SUSU, 1);
  DTensor16 KSUSU1;
  // K00, K01+U01, K02+U11, K03, K04, K05, K06, K07, K08, U00, U02, U03+U14,
  // U04, U10, U12, U13
  ContractSelf(&KSUSU1, KSUSU, 2, 14);
  DTensor15 KSUSU2;
  // K00, K01+U01, K02+U11, K03, K04, K05+U04, K06, K07, K08, U00, U02,
  // U03+U14, U10, U12, U13
  ContractSelf(&KSUSU2, KSUSU1, 5, 12);
  DTensor14 KSUSU3;
  // K00, K01+U01, K02+U11, K03, K04, K05+U04, K06+U03+U14, K07, K08, U00, U02,
  // U10, U12, U13
  ContractSelf(&KSUSU3, KSUSU2, 6, 11);
  DTensor13 KSUSU4;
  // K00, K01+U01, K02+U11, K03, K04, K05+U04, K06+U03+U14, K07+U13, K08, U00,
  // U02, U10, U12
  ContractSelf(&KSUSU4, KSUSU3, 7, 13);
  DTensor9 SVSV;
  // V00, V01, V02, V03, V04+V13, V10, V11, V12, V14
  Contract(&SVSV, SV, 4, SV, 3);
  DTensor17 KSVSV;
  // K10+V02, K11, K12, K13, K14, K15, K16, K17, K18, V00, V01, V02, V04+V13,
  // V10, V11, V12, V14
  Contract(&KSVSV, K, 0, SVSV, 2);
  DTensor16 KSVSV1;
  // K10+V02, K11, K12, K13+V12, K14, K15, K16, K17, K18, V00, V01, V02,
  // V04+V13, V10, V11, V14
  ContractSelf(&KSVSV1, KSVSV, 3, 15);
  DTensor15 KSVSV2;
  // K10+V02, K11, K12, K13+V12, K14+V04+V13, K15, K16, K17, K18, V00, V01,
  // V02, V10, V11, V14
  ContractSelf(&KSVSV2, KSVSV1, 4, 12);
  DTensor14 KSVSV3;
  // K10+V02, K11, K12, K13+V12, K14+V04+V13, K15+V02, K16, K17, K18, V00, V01,
  // V10, V11, V14
  ContractSelf(&KSVSV3, KSVSV2, 5, 11);
  DTensor13 KSVSV4;
  // K10+V02, K11, K12, K13+V12, K14+V04+V13, K15+V02, K16, K17+V14, K18, V00,
  // V01, V10, V11
  ContractSelf(&KSVSV4, KSVSV3, 7, 13);
  DTensor25 KKSUSUSVSV;
  // K00+V01, K01+U01, K02+U11, K03, K04, K05+U04, K06+U03+U14, K07+U13, K08,
  // U00, U02, U10, U12, K10+V02, K11, K12, K13+V12, K14+V04+V13, K15+V02, K16,
  // K17+V14, K18, V00, V10, V11
  Contract(&KKSUSUSVSV, KSUSU4, 0, KSVSV4, 10);
  DTensor24 KKSUSUSVSV1;
  // K00+V01, K01+U01, K02+U11, K03+V11, K04, K05+U04, K06+U03+U14, K07+U13,
  // K08, U00, U02, U10, U12, K10+V02, K11, K12, K13+V12, K14+V04+V13, K15+V02,
  // K16, K17+V14, K18, V00, V10
  ContractSelf(&KKSUSUSVSV1, KKSUSUSVSV, 3, 24);
  DTensor22 KKSUSUSVSV2;
  // K00+V01, K01+U01, K02+U11, K03+V11, K05+U04, K06+U03+U14, K07+U13,
  // K08, U00, U02, U10, U12, K10+V02, K11, K12, K13+V12, K15+V02,
  // K16, K17+V14, K18, V00, V10 (K04+K14+V04+V13)
  ContractSelf2(&KKSUSUSVSV2, KKSUSUSVSV1, 4, 17);
  DTensor20 KKSUSUSVSV3;
  // K00+V01, K01+U01, K02+U11, K03+V11, K06+U03+U14, K07+U13,
  // K08, U00, U02, U10, U12, K10+V02, K11, K12, K13+V12,
  // K16, K17+V14, K18, V00, V10 (K05+U04+K15+V02)
  ContractSelf2(&KKSUSUSVSV3, KKSUSUSVSV2, 4, 16);
  DTensor18 KKSUSUSVSV4;
  // K00+V01, K01+U01, K02+U11, K03+V11, K07+U13,
  // K08, U00, U02, U10, U12, K10+V02, K11, K12, K13+V12,
  // K17+V14, K18, V00, V10 (K06+U03+U14+K16)
  ContractSelf2(&KKSUSUSVSV4, KKSUSUSVSV3, 4, 15);
  DTensor16 KKSUSUSVSV5;
  // K00+V01, K01+U01, K02+U11, K03+V11,
  // K08, U00, U02, U10, U12, K10+V02, K11, K12, K13+V12,
  // K18, V00, V10 (K07+U13+K17+V14)
  ContractSelf2(&KKSUSUSVSV5, KKSUSUSVSV4, 4, 14);
  DTensor15 KKSUSUSVSV6;
  // K00+V01, K01+U01, K02+U11, K03+V11,
  // K08, U00, U02+K11, U10, U12, K10+V02, K12, K13+V12,
  // K18, V00, V10
  ContractSelf(&KKSUSUSVSV6, KKSUSUSVSV5, 6, 10);
  DTensor14 KKSUSUSVSV7;
  // K00+V01, K01+U01, K02+U11, K03+V11,
  // K08, U00, U02+K11, U10, U12+K12, K10+V02, K13+V12,
  // K18, V00, V10
  ContractSelf(&KKSUSUSVSV7, KKSUSUSVSV6, 8, 10);
//  rank_t mapping[14] = {3, 6, 9, 12, 0, 5, 7, 8, 10, 4, 13, 1, 2, 11};
  rank_t mapping[14] = {4, 11, 12, 0, 9, 5, 1, 6, 7, 2, 8, 13, 3, 10};
  Rearrange(C, KKSUSUSVSV7, mapping);
}

void MakeSecondBlocks(DTensor4 *B, DTensor3 *m_A, DTensor3 *m_B,
    const DTensor14 *C, const uint8_t ind[3], const uint8_t sv_len[3][3],
    const uint8_t rho_A[3][5], const uint8_t rho_B[3][5], const double condi) {
  uint8_t k;
  uint8_t l;
  double C_val;

  for (uint8_t rho_M = 0; rho_M < 3; ++rho_M)
    for (uint8_t rho_N = 0; rho_N < 3; ++rho_N) {
      k = 0;
      for (uint8_t m = 0; m < ind[rho_M]; ++m)
        for (uint8_t n = 0; n < ind[rho_N]; ++n)
          for (uint8_t m1 = 0; m1 < sv_len[rho_A[rho_M][m]][rho_A[rho_N][n]];
              ++m1)
            for (uint8_t m2 = 0; m2 < sv_len[rho_B[rho_M][m]][rho_B[rho_N][n]];
                ++m2) {
              m_A->Set(rho_M, rho_N, k, m1);
              m_B->Set(rho_M, rho_N, k, m2);
              l = 0;
              for (uint8_t i = 0; i < ind[rho_M]; ++i)
                for (uint8_t j = 0; j < ind[rho_N]; ++j)
                  for (uint8_t m3 = 0;
                      m3 < sv_len[rho_A[rho_M][i]][rho_A[rho_N][j]]; ++m3)
                    for (uint8_t m4 = 0;
                        m4 < sv_len[rho_B[rho_M][i]][rho_B[rho_N][j]]; ++m4) {
                      C_val = C->Get(rho_M, rho_N, m1, rho_A[rho_M][m],
                          rho_A[rho_N][n], m2, rho_B[rho_M][m],
                          rho_B[rho_N][n], m3, rho_A[rho_M][i],
                          rho_A[rho_N][j], m4, rho_B[rho_M][i],
                          rho_B[rho_N][j]);
                      if (fabs(C_val) > condi)
                        B->Set(rho_M, rho_N, k, l, C_val);
                      ++l;
                    }
              ++k;
            }
    }
}

void DoLoopSVD(DTensor9 result[2], uint8_t sv_len[3][3], DTensor4 *B,
    const uint8_t dc, const double condi, const uint8_t rho_A[3][5],
    const uint8_t rho_B[3][5], const DTensor3 *m_A, const DTensor3 *m_B) {
  uint8_t sv_list[9*dc][3];
  uint8_t sv_num = 0;
  gsl_matrix *U[3][3];
  gsl_matrix *V[3][3];
  //FIXME: make GetGSLMatrix find size from tensor or keep track of tensor sizes
  uint8_t msize[3][3] = {{3, 1, 1}, {1, 3, 1}, {1, 1, 5}};
  for (uint8_t rho_M = 0; rho_M < 3; ++rho_M)
    for (uint8_t rho_N = 0; rho_N < 3; ++rho_N) {
      sv_len[rho_M][rho_N] = 0;
      U[rho_M][rho_N] = B->GetGSLMatrix(rho_M, rho_N, 2, 3, msize[rho_M][rho_N],
          msize[rho_M][rho_N]);
      S[rho_M][rho_N] = gsl_vector_alloc(msize[rho_N][rho_M]);
      V[rho_M][rho_N] = gsl_matrix_alloc(msize[rho_N][rho_M],
          msize[rho_N][rho_M]);
      gsl_linalg_SV_decomp_jacobi(U[rho_M][rho_N], V[rho_M][rho_N],
          S[rho_M][rho_N]);
      for (uint8_t i = 0; i < std::min(msize[rho_M][rho_N], dc); ++i)
        if (fabs(gsl_vector_get(S[rho_M][rho_N], i)) > condi) {
          std::cout << int(rho_M) << std::endl;
          sv_list[sv_num][0] = rho_M;
          sv_list[sv_num][1] = rho_N;
          sv_list[sv_num][2] = i;
          ++sv_num;
        }
    }
  qsort(sv_list, sv_num, sizeof(sv_list[0]), CompareSVs);
  uint8_t rho_M;
  uint8_t rho_N;
  uint8_t i;
  uint8_t j;
  uint8_t m_A_val;
  uint8_t m_B_val;
  double sv;
  double sv_max = gsl_vector_get(S[sv_list[0][0]][sv_list[0][1]],
      sv_list[0][2]);
  for (uint8_t n = 0; n < std::min(sv_num, dc); ++n) {
    rho_M = sv_list[n][0];
    rho_N = sv_list[n][1];
    i = sv_list[n][2];
    j = sv_len[rho_M][rho_N];
    m_A_val = m_A->Get(rho_M, rho_N, j);
    m_B_val = m_B->Get(rho_M, rho_N, j);
    for (uint8_t m = 0; m < msize[rho_M][rho_N]; ++m) {
      sv = sqrt(gsl_vector_get(S[rho_M][rho_N], i) * dim[rho_M] * dim[rho_N] /
          sv_max);
      result[0].Set(i, rho_M, rho_N, m_A_val, rho_A[rho_M][m], rho_A[rho_N][n],
          m_B_val, rho_B[rho_M][m], rho_B[rho_N][n],
          sv * gsl_matrix_get(U[rho_M][rho_N], m, i));
      result[1].Set(i, rho_M, rho_N, m_A_val, rho_A[rho_M][m], rho_A[rho_N][n],
          m_B_val, rho_B[rho_M][m], rho_B[rho_N][n],
          sv * gsl_matrix_get(V[rho_M][rho_N], m, i));
    }
    ++sv_len[rho_M][rho_N];
  }
  for (uint8_t rho_M = 0; rho_M < 3; ++rho_M)
    for (uint8_t rho_N = 0; rho_N < 3; ++rho_N) {
      gsl_matrix_free(U[rho_M][rho_N]);
      gsl_vector_free(S[rho_M][rho_N]);
      gsl_matrix_free(V[rho_M][rho_N]);
    }
}

void TRGS3(const double a, const double b, const double c,
    const uint8_t dc, const double condi, const uint8_t iter) {
  //FIXME: move K construction into main() to save repeated construction
  DTensor9 K;
  Make9j(&K);
  std::cout << K << std::endl;
  DTensor4 B1;
  MakeFirstBlocks(&B1, a, b, c);
  std::cout << B1 << std::endl;
  DTensor5 SVD1[2];
  uint8_t sv_len[3][3];
  DoFirstSVD(SVD1, sv_len, &B1, dc, condi);
  DTensor5 &SU1 = SVD1[0];
  DTensor5 &SV1 = SVD1[1];
  std::cout << SU1 << std::endl;
  std::cout << SV1 << std::endl;
  DTensor14 C1;
  DoFirstContraction(&C1, K, SU1, SV1);
  std::cout << C1 << std::endl;
  DTensor4 B2;
  uint8_t ind[] = {3, 3, 5};
  uint8_t rho_A[3][5] = {{0, 1, 2}, {0, 1, 2}, {0, 2, 2, 1, 2}};
  uint8_t rho_B[3][5] = {{0, 1, 2}, {1, 0, 2}, {2, 0, 2, 2, 1}};
  //FIXME: don't use DTensors to store uint8_ts
  DTensor3 m_A;
  DTensor3 m_B;
  MakeSecondBlocks(&B2, &m_A, &m_B, &C1, ind, sv_len, rho_A, rho_B, condi);
  std::cout << B2 << std::endl;
  for (uint8_t i = 0; i < iter; ++i) {
    DTensor4 B3;
    DTensor9 SVD2[2];
    DoLoopSVD(SVD2, sv_len, &B3, dc, condi, rho_A, rho_B, &m_A, &m_B);
    DTensor9 &SU2 = SVD2[0];
    DTensor9 &SV2 = SVD2[1];
    std::cout << SU2 << std::endl;
    std::cout << SV2 << std::endl;
  }
}

int main(int argc, char **argv) {
  TRGS3(0, 0, 0, 9, 1e-8, 1);
}
