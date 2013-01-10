/* -*- indent-tabs-mode: nil -*- */

#include <stdint.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sort_vector.h>

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

void MakeFirstBlocks(DTensor4 *B, const double a, const double b,
    const double c) {
  double expr1 = 1 + a + 2 * b + 2 * c;
  double expr2 = a * a + 4 * a * b + 4 * b * b - (1 + 2 * c) * (1 + 2 * c);
  double expr3 = a * a - 2 * a * b + b * b + (-1 + c) * (-1 + c);
  double expr4 = -1 + a + 2 * b - 2 * c;
  double expr5 = pow(a, 4) + 2 * b * pow(a, 3) + 4 * pow(b, 4) +
      b * b * (-5 + 4 * c - 8 * c * c) -
      2 * a * b * (1 + 2 * b * b - 8 * c- 2 * c * c) +
      (1 + c - 2 * c * c) * (1 + c - 2 * c * c) -
      a * a * (2 + 3 * b * b + 2 * c + 5 * c * c);
  double expr6 = pow(a, 4) - 2 * pow(b, 4) - 2 * b * pow((-1 + c), 3) -
      pow(b, 3) * (1 + 2 * c) - pow((-1 + c), 3) * (1 + 2 * c) +
      pow(a, 3) * (1 - b + 2 * c) - 3 * a * a * b * (1 + b + 2 * c) +
      a * (5 * pow(b, 3) - pow(-1 + c, 3) + b * b * (3 + 6 * c));
  double expr7 = pow(a, 4) - 4 * b * pow(a, 3) + 6 * a * a * b * b -
      4 * a * pow(b, 3) + pow(b, 4) + pow(-1 + c, 4);
  double expr8 = pow(a, 4) - 2 * pow(b, 4) -
      3 * b * a * a * (-1 + b - 2 * c) + 2 * b * pow(-1 + c, 3) +
      pow(b, 3) * (1 + 2 * c) - pow(-1 + c, 3) * (1 + 2 * c) -
      pow(a, 3) * (1 + b + 2 * c) + a * (5 * pow(b, 3) + pow(-1 + c, 3) -
      3 * b * b * (1 + 2 * c));
  double expr9 = a * a - 2 * a * b + b * b - (-1 + c) * (-1 + c);

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

void MakeSecondBlocks(DTensor4 *B, uint8_t m_A[][3][3], uint8_t m_B[][3][3],
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
              m_A[k][rho_M][rho_N] = m1;
              m_B[k][rho_M][rho_N] = m2;
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

void MakeLoopBlocks(DTensor4 *B, uint8_t m_A[][3][3], uint8_t m_B[][3][3],
    const DTensor14 *C, const uint8_t ind[3], const uint8_t sv_len[3][3],
    const uint8_t rho_A[3][5], const uint8_t rho_B[3][5], const double condi) {
  MakeSecondBlocks(B, m_A, m_B, C, ind, sv_len, rho_A, rho_B, condi);
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
      S[rho_M][rho_N] = gsl_vector_alloc(msize[rho_M][rho_N]);
      V[rho_M][rho_N] = gsl_matrix_alloc(msize[rho_M][rho_N],
          msize[rho_M][rho_N]);
      gsl_linalg_SV_decomp_jacobi(U[rho_M][rho_N], V[rho_M][rho_N],
          S[rho_M][rho_N]);
      for (uint8_t i = 0; i < std::min(msize[rho_M][rho_N], dc); ++i)
        if (fabs(gsl_vector_get(S[rho_M][rho_N], i)) > condi) {
          sv_list[sv_num][0] = rho_M;
          sv_list[sv_num][1] = rho_N;
          sv_list[sv_num][2] = i;
          ++sv_num;
          ++sv_len[rho_M][rho_N];
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
      sv_list[0][2]) * dim[sv_list[0][0]] * dim[sv_list[0][1]];

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
  }

  for (uint8_t rho_M = 0; rho_M < 3; ++rho_M)
    for (uint8_t rho_N = 0; rho_N < 3; ++rho_N) {
      gsl_matrix_free(U[rho_M][rho_N]);
      gsl_vector_free(S[rho_M][rho_N]);
      gsl_matrix_free(V[rho_M][rho_N]);
    }
}

void DoLoopSVD(DTensor9 result[2], uint8_t sv_len[3][3], DTensor4 *B,
    const uint8_t dc, const double condi, const uint8_t rho_A[3][5],
    const uint8_t rho_B[3][5], const uint8_t m_A[][3][3],
    const uint8_t m_B[][3][3]) {
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
      S[rho_M][rho_N] = gsl_vector_alloc(msize[rho_M][rho_N]);
      V[rho_M][rho_N] = gsl_matrix_alloc(msize[rho_M][rho_N],
          msize[rho_M][rho_N]);
      gsl_linalg_SV_decomp_jacobi(U[rho_M][rho_N], V[rho_M][rho_N],
          S[rho_M][rho_N]);
      for (uint8_t i = 0; i < std::min(msize[rho_M][rho_N], dc); ++i) {
        if (fabs(gsl_vector_get(S[rho_M][rho_N], i)) > condi) {
          sv_list[sv_num][0] = rho_M;
          sv_list[sv_num][1] = rho_N;
          sv_list[sv_num][2] = i;
          ++sv_num;
          ++sv_len[rho_M][rho_N];
        }
      }
    }

  qsort(sv_list, sv_num, sizeof(sv_list[0]), CompareSVs);

  uint8_t rho_M;
  uint8_t rho_N;
  uint8_t i;
  uint8_t m_A_val;
  uint8_t m_B_val;
  double sv;
  double sv_max = gsl_vector_get(S[sv_list[0][0]][sv_list[0][1]],
        sv_list[0][2]) * dim[sv_list[0][0]] * dim[sv_list[0][1]];
  for (uint8_t n = 0; n < std::min(sv_num, dc); ++n) {
    rho_M = sv_list[n][0];
    rho_N = sv_list[n][1];
    i = sv_list[n][2];
    for (uint8_t m = 0; m < msize[rho_M][rho_M]; ++m)
      for (uint8_t p = 0; p < msize[rho_N][rho_N]; ++p)
        for (uint8_t j = 0; j < sv_len[rho_A[rho_M][m]][rho_A[rho_N][p]] *
            sv_len[rho_B[rho_M][m]][rho_B[rho_N][p]]; ++j) {
          m_A_val = m_A[j][rho_M][rho_N];
          m_B_val = m_B[j][rho_M][rho_N];
          sv = sqrt(gsl_vector_get(S[rho_M][rho_N], i) *
              dim[rho_M] * dim[rho_N] / sv_max);
          result[0].Set(i, rho_M, rho_N, m_A_val, rho_A[rho_M][m],
              rho_A[rho_N][p], m_B_val, rho_B[rho_M][m], rho_B[rho_N][p],
              sv * gsl_matrix_get(U[rho_M][rho_N], m, i));
          result[1].Set(i, rho_M, rho_N, m_A_val, rho_A[rho_M][m],
              rho_A[rho_N][p], m_B_val, rho_B[rho_M][m], rho_B[rho_N][p],
              sv * gsl_matrix_get(V[rho_M][rho_N], m, i));;
        }
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

  rank_t mapping[14] = {4, 11, 12, 0, 9, 5, 1, 6, 7, 2, 8, 13, 3, 10};
  Rearrange(C, KKSUSUSVSV7, mapping);
}

void DoLoopContraction(DTensor14 *C, const DTensor9 &K, const DTensor9 &SU,
    const DTensor9 &SV) {
  DTensor16 SUSU;
  // U00, U01, U02, U04, U05, U06, U07, U08, U10, U11, U12, U13, U14,
  // U15, U17, U18 (U03+U16)
  Contract2(&SUSU, SU, 3, SU, 6);
  DTensor15 SUSU1;
  // U00, U01, U02, U04+U17, U05, U06, U07, U08, U10, U11, U12, U13, U14,
  // U15, U18
  ContractSelf(&SUSU1, SUSU, 3, 14);
  DTensor14 SUSU2;
  // U00, U01, U02, U04+U17, U05+U18, U06, U07, U08, U10, U11, U12, U13, U14,
  // U15
  ContractSelf(&SUSU2, SUSU1, 4, 14);
  DTensor22 KSUSU;
  // K00, K01+U01, K02, K03, K04, K05, K06, K07, K08, U00, U02, U04+U17,
  // U05+U18, U06, U07, U08, U10, U11, U12, U13, U14, U15
  Contract(&KSUSU, K, 1, SUSU2, 1);
  DTensor21 KSUSU1;
  // K00, K01+U01, K02+U12, K03, K04, K05, K06, K07, K08, U00, U02, U04+U17,
  // U05+U18, U06, U07, U08, U10, U11, U13, U14, U15
  ContractSelf(&KSUSU1, KSUSU, 2, 17);
  DTensor20 KSUSU2;
  // K00, K01+U01, K02+U12, K03, K04, K05+U07, K06, K07, K08, U00, U02, U04+U17,
  // U05+U18, U06, U08, U10, U11, U13, U14, U15
  ContractSelf(&KSUSU2, KSUSU1, 5, 14);
  DTensor18 KSUSU3;
  // K00, K01+U01, K02+U12, K03, K04, K05+U07, K07, K08, U00, U02,
  // U05+U18, U06, U08, U10, U11, U13, U14, U15 (K06+U04+U17)
  ContractSelf2(&KSUSU3, KSUSU2, 6, 11);
  DTensor17 KSUSU4;
  // K00, K01+U01, K02+U12, K03, K04, K05+U07, K07+U14, K08, U00, U02,
  // U05+U18, U06, U08, U10, U11, U13, U15
  ContractSelf(&KSUSU4, KSUSU3, 6, 16);
  DTensor16 SVSV;
  // V00, V01, V02, V03, V04, V05, V07, V08, V10, V11, V12, V14,
  // V15, V16, V17, V18 (V06+V13)
  Contract2(&SVSV, SV, 6, SV, 3);
  DTensor15 SVSV1;
  // V00, V01, V02, V03, V04, V05, V07+V14, V08, V10, V11, V12,
  // V15, V16, V17, V18
  ContractSelf(&SVSV1, SVSV, 6, 11);
  DTensor14 SVSV2;
  // V00, V01, V02, V03, V04, V05, V07+V14, V08+V15, V10, V11, V12,
  // V16, V17, V18
  ContractSelf(&SVSV2, SVSV1, 7, 11);
  DTensor22 KSVSV;
  // K10+V02, K11, K12, K13, K14, K15, K16, K17, K18, V00, V01, V03, V04, V05,
  // V07+V14, V08+V15, V10, V11, V12, V16, V17, V18
  Contract(&KSVSV, K, 0, SVSV2, 2);
  DTensor21 KSVSV1;
  // K10+V02, K11, K12, K13+V12, K14, K15, K16, K17, K18, V00, V01, V03,
  // V04, V05, V07+V14, V08+V15, V10, V11, V16, V17, V18
  ContractSelf(&KSVSV1, KSVSV, 3, 18);
  DTensor19 KSVSV2;
  // K10+V02, K11, K12, K13+V12, K15, K16, K17, K18, V00, V01, V03,
  // V04, V05, V07+V14, V10, V11, V16, V17, V18 (K14+V08+V15)
  ContractSelf2(&KSVSV2, KSVSV1, 4, 15);
  DTensor18 KSVSV3;
  // K10+V02, K11, K12, K13+V12, K15+V05, K16, K17, K18, V00, V01, V03,
  // V04, V07+V14, V10, V11, V16, V17, V18
  ContractSelf(&KSVSV3, KSVSV2, 4, 12);
  DTensor17 KSVSV4;
  // K10+V02, K11, K12, K13+V12, K15+V05, K16, K17+V18, K18, V00, V01, V03,
  // V04, V07+V14, V10, V11, V16, V17
  ContractSelf(&KSVSV4, KSVSV3, 6, 17);
  DTensor33 KKSUSUSVSV;
  // K00+V01, K01+U01, K02+U12, K03, K04, K05+U07, K07+U14, K08, U00, U02,
  // U05+U18, U06, U08, U10, U11, U13, U15,
  // K10+V02, K11, K12, K13+V12, K15+V05, K16, K17+V18, K18, V00, V03,
  // V04, V07+V14, V10, V11, V16, V17
  Contract(&KKSUSUSVSV, KSUSU4, 0, KSVSV4, 9);
  DTensor32 KKSUSUSVSV1;
  // K00+V01, K01+U01, K02+U12, K03+V11, K04, K05+U07, K07+U14, K08, U00, U02,
  // U05+U18, U06, U08, U10, U11, U13, U15,
  // K10+V02, K11, K12, K13+V12, K15+V05, K16, K17+V18, K18, V00, V03,
  // V04, V07+V14, V10, V16, V17
  ContractSelf(&KKSUSUSVSV1, KKSUSUSVSV, 3, 30);
  DTensor30 KKSUSUSVSV2;
  // K00+V01, K01+U01, K02+U12, K03+V11, K05+U07, K07+U14, K08, U00, U02,
  // U05+U18, U06, U08, U10, U11, U13, U15,
  // K10+V02, K11, K12, K13+V12, K15+V05, K16, K17+V18, K18, V00, V03,
  // V04, V10, V16, V17 (K04+V07+V14)
  ContractSelf2(&KKSUSUSVSV2, KKSUSUSVSV1, 4, 28);
  DTensor28 KKSUSUSVSV3;
  // K00+V01, K01+U01, K02+U12, K03+V11, K07+U14, K08, U00, U02,
  // U05+U18, U06, U08, U10, U11, U13, U15,
  // K10+V02, K11, K12, K13+V12, K15+V05, K16, K17+V18, K18, V00, V03,
  // V10, V16, V17 (K05+U07+V04)
  ContractSelf2(&KKSUSUSVSV3, KKSUSUSVSV2, 4, 26);
  DTensor26 KKSUSUSVSV4;
  // K00+V01, K01+U01, K02+U12, K03+V11, K08, U00, U02,
  // U05+U18, U06, U08, U10, U11, U13, U15,
  // K10+V02, K11, K12, K13+V12, K15+V05, K16, K17+V18, K18, V00, V03,
  // V10, V16 (K07+U14+V17)
  ContractSelf2(&KKSUSUSVSV4, KKSUSUSVSV3, 4, 27);
  DTensor25 KKSUSUSVSV5;
  // K00+V01, K01+U01, K02+U12, K03+V11, K08, U00, U02+K11,
  // U05+U18, U06, U08, U10, U11, U13, U15,
  // K10+V02, K12, K13+V12, K15+V05, K16, K17+V18, K18, V00, V03,
  // V10, V16
  ContractSelf(&KKSUSUSVSV5, KKSUSUSVSV4, 6, 15);
  DTensor23 KKSUSUSVSV6;
  // K00+V01, K01+U01, K02+U12, K03+V11, K08, U00, U02+K11,
  // U06, U08, U10, U11, U13, U15,
  // K10+V02, K12, K13+V12, K15+V05, K17+V18, K18, V00, V03,
  // V10, V16 (U05+U18+K16)
  ContractSelf2(&KKSUSUSVSV6, KKSUSUSVSV5, 7, 18);
  DTensor21 KKSUSUSVSV7;
  // K00+V01, K01+U01, K02+U12, K03+V11, K08, U00, U02+K11,
  // U08, U10, U11, U13, U15, K10+V02, K12, K13+V12, K15+V05, K17+V18, K18, V00,
  // V10, V16 (U06+V03)
  ContractSelf2(&KKSUSUSVSV7, KKSUSUSVSV6, 7, 20);
  DTensor19 KKSUSUSVSV8;
  // K00+V01, K01+U01, K02+U12, K03+V11, K08, U00, U02+K11,
  // U10, U11, U13, U15, K10+V02, K12, K13+V12, K17+V18, K18, V00,
  // V10, V16 (U08+K15+V05)
  ContractSelf2(&KKSUSUSVSV8, KKSUSUSVSV7, 7, 15);
  DTensor18 KKSUSUSVSV9;
  // K00+V01, K01+U01, K02+U12, K03+V11, K08, U00, U02+K11,
  // U10, U11+K12, U13, U15, K10+V02, K13+V12, K17+V18, K18, V00, V10, V16
  ContractSelf(&KKSUSUSVSV9, KKSUSUSVSV8, 8, 12);
  DTensor16 KKSUSUSVSV10;
  // K00+V01, K01+U01, K02+U12, K03+V11, K08, U00, U02+K11,
  // U10, U11+K12, U15, K10+V02, K13+V12, K17+V18, K18, V00, V10 (U13+V16)
  ContractSelf2(&KKSUSUSVSV10, KKSUSUSVSV9, 9, 17);
  DTensor14 KKSUSUSVSV11;
  // K00+V01, K01+U01, K02+U12, K03+V11, K08, U00, U02+K11,
  // U10, U11+K12, K10+V02, K13+V12, K18, V00, V10 (U15+K17+V18)
  ContractSelf2(&KKSUSUSVSV11, KKSUSUSVSV10, 9, 12);
  rank_t mapping[14] = {4, 11, 12, 0, 9, 5, 1, 6, 7, 2, 8, 13, 3, 10};
  Rearrange(C, KKSUSUSVSV11, mapping);
}

void TRGS3(const double a, const double b, const double c,
    const uint8_t dc, const double condi, const unsigned iter,
    const DTensor9 &K) {
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
  size_t m_max = 25 * dc * dc;
  uint8_t m_A[m_max][3][3];
  uint8_t m_B[m_max][3][3];
  MakeSecondBlocks(&B2, m_A, m_B, &C1, ind, sv_len, rho_A, rho_B, condi);
  std::cout << B2 << std::endl;

  for (unsigned i = 0; i < iter; ++i) {
    DTensor9 SVD2[2];
    DoLoopSVD(SVD2, sv_len, &B2, dc, condi, rho_A, rho_B, m_A, m_B);
    B2.Clear();
    DTensor9 &SU2 = SVD2[0];
    DTensor9 &SV2 = SVD2[1];
    std::cout << SU2 << std::endl;
    std::cout << SV2 << std::endl;
    DTensor14 C2;
    DoLoopContraction(&C2, K, SU2, SV2);
    std::cout << C2 << std::endl;
    MakeLoopBlocks(&B2, m_A, m_B, &C2, ind, sv_len, rho_A, rho_B, condi);
    std::cout << B2 << std::endl;
  }
}

int main(int argc, char **argv) {
  DTensor9 K;
  Make9j(&K);

  TRGS3(0, 0, 0, 9, 1e-8, 10, K);
}
