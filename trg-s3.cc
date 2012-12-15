/* -*- indent-tabs-mode: nil -*- */

#include <stdint.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

#include "tensor.h"

void Make9j(DTensor9 *K) {
  DTensor3 PP;
  PP.Set(0,0,0,1);
  PP.Set(0,1,1,1);
  PP.Set(0,2,2,1/sqrt(2));
  PP.Set(0,3,3,1/sqrt(2));
  PP.Set(1,0,1,1);
  PP.Set(1,1,0,1);
  PP.Set(1,2,3,-1/sqrt(2));
  PP.Set(1,3,2,1/sqrt(2));
  PP.Set(2,0,2,1/sqrt(2));
  PP.Set(2,1,3,-1/sqrt(2));
  PP.Set(2,2,0,1/sqrt(2));
  PP.Set(2,2,2,-1/2);
  PP.Set(2,3,1,1/sqrt(2));
  PP.Set(2,3,3,1/2);
  PP.Set(3,0,3,1/sqrt(2));
  PP.Set(3,1,2,1/sqrt(2));
  PP.Set(3,2,1,-1/sqrt(2));
  PP.Set(3,2,3,1/2);
  PP.Set(3,3,0,1/sqrt(2));
  PP.Set(3,3,2,1/2);
  uint8_t ind1[] = {0,1,2};
  uint8_t ind2[] = {0,1,3};

#define F(x) \
  for (uint8_t x = 0; x < 3; ++x) \
    for (uint8_t a##x = ind1[x]; a##x <= ind2[x]; ++a##x)

  F(i) F(j) F(k) F(l) F(m) F(n) F(o) F(p) F(q)
    K->Set(i,j,k,l,m,n,o,p,q,
        K->Get(i,j,k,l,m,n,o,p,q)+
        PP.Get(ai,aj,aq)*PP.Get(ak,al,aq)*PP.Get(an,am,ai)*PP.Get(ao,an,aj)*
        PP.Get(ap,ao,ak)*PP.Get(am,ap,al));
}

void MakeFirstBlocks(DTensor4 *B, double a, double b, double c) {
  double expr1 = 1 + a + 2*b + 2*c;
  double expr2 = pow(a,2) + 4*a*b + 4*pow(b,2) - pow(1 + 2*c,2);
  double expr3 = pow(a,2) - 2*a*b + pow(b,2) + pow(-1 + c,2);
  double expr4 = -1 + a + 2*b - 2*c;
  double expr5 = pow(a,4) + 2*pow(a,3)*b + 4*pow(b,4) +
      pow(b,2)*(-5 + 4*c - 8*pow(c,2)) -
      2*a*b*(1 + 2*pow(b,2) - 8*c- 2*pow(c,2)) + pow(1 + c - 2*pow(c,2),2) -
      pow(a,2)*(2 + 3*pow(b,2) + 2*c + 5*pow(c,2));
  double expr6 = pow(a,4) - 2*pow(b,4) - 2*b*pow((-1 + c),3) -
      pow(b,3)*(1 + 2*c) - pow((-1 + c),3) * (1 + 2*c) +
      pow(a,3)*(1 - b + 2*c) - 3*pow(a,2)*b*(1 + b + 2*c) + a*(5*pow(b,3) -
      pow(-1 + c,3) + pow(b,2)*(3 + 6*c));
  double expr7 = pow(a,4) - 4*pow(a,3)*b + 6*pow(a,2)*pow(b,2) - 4*a*pow(b,3) +
      pow(b,4) + pow(-1 + c,4);
  double expr8 = pow(a,4) - 2*pow(b,4) - 3*pow(a,2)*b*(-1 + b - 2*c) +
      2*b*pow(-1 + c,3) + pow(b,3)*(1 + 2*c) - pow(-1 + c,3)*(1 + 2*c) -
      pow(a,3)*(1 + b + 2*c) + a*(5*pow(b,3) + pow(-1 + c,3) -
      3*pow(b,2)*(1 + 2*c));
  double expr9 = pow(a,2) - 2*a*b + pow(b,2) - pow(-1 + c,2);

  B->Set(0,0,0,0,pow(expr1,4));
  B->Set(0,0,0,1,pow(expr2,2));
  B->Set(0,0,0,2,2*expr3*pow(expr1,2));
  B->Set(0,0,1,0,pow(expr2,2));
  B->Set(0,0,1,1,pow(expr4,4));
  B->Set(0,0,1,2,2*expr3*pow(expr4,2));
  B->Set(0,0,2,0,2*expr3*pow(expr1,2));
  B->Set(0,0,2,1,2*expr3*pow(expr4,2));
  B->Set(0,0,2,2,4*pow(expr3,2));

  B->Set(0,2,0,0,8*pow(a - b,2)*pow(-1 + c,2));

  B->Set(1,1,0,0,pow(expr2,2));
  B->Set(1,1,0,1,pow(expr2,2));
  B->Set(1,1,0,2,2*expr5);
  B->Set(1,1,1,0,pow(expr2,2));
  B->Set(1,1,1,1,pow(expr2,2));
  B->Set(1,1,1,2,2*expr5);
  B->Set(1,1,2,0,2*expr5);
  B->Set(1,1,2,1,2*expr5);
  B->Set(1,1,2,2,4*pow(expr9,2));

  B->Set(2,0,0,0,8*pow(a - b,2)*pow(-1 + c,2));

  B->Set(2,2,0,0,expr3*pow(expr1,2));
  B->Set(2,2,0,1,expr3*pow(expr1,2));
  B->Set(2,2,0,2,sqrt(2)*expr6);
  B->Set(2,2,0,3,expr5);
  B->Set(2,2,0,4,expr5);
  B->Set(2,2,1,0,expr3*pow(expr1,2));
  B->Set(2,2,1,1,expr3*pow(expr1,2));
  B->Set(2,2,1,2,sqrt(2)*expr6);
  B->Set(2,2,1,3,expr5);
  B->Set(2,2,1,4,expr5);
  B->Set(2,2,2,0,sqrt(2)*expr6);
  B->Set(2,2,2,1,sqrt(2)*expr6);
  B->Set(2,2,2,2,2*expr7);
  B->Set(2,2,2,3,sqrt(2)*expr8);
  B->Set(2,2,2,4,sqrt(2)*expr8);
  B->Set(2,2,3,0,expr5);
  B->Set(2,2,3,1,expr5);
  B->Set(2,2,3,2,sqrt(2)*expr8);
  B->Set(2,2,3,3,expr3*pow(expr4,2));
  B->Set(2,2,3,4,expr3*pow(expr4,2));
  B->Set(2,2,4,0,expr5);
  B->Set(2,2,4,1,expr5);
  B->Set(2,2,4,2,sqrt(2)*expr8);
  B->Set(2,2,4,3,expr3*pow(expr4,2));
  B->Set(2,2,4,4,expr3*pow(expr4,2));

//  double B00[][3] = {{pow(expr1,4), pow(expr2,2), 2*expr3*pow(expr1,2)},
//      {pow(expr2,2), pow(expr4,4), 2*expr3*pow(expr4,2)},
//      {2*expr3*pow(expr1,2), 2*expr3*pow(expr4,2), 4*pow(expr3,2)}};
//  double B01 = 0;
//  double B02 = 8*pow(a - b,2)*pow(-1 + c,2);
//  double B10 = 0;
//  double B11[][3] = {{pow(expr2,2), pow(expr2,2), 2*expr5},
//      {pow(expr2,2), pow(expr2,2), 2*expr5},
//      {2*expr5, 2*expr5, 4*pow(expr9,2)}};
//  double B12 = 0;
//  double B20 = 8*pow(a - b,2)*pow(-1 + c,2);
//  double B21 = 0;
//  double B22[][5] = {
//      {expr3*pow(expr1,2), expr3*pow(expr1,2), sqrt(2)*expr6, expr5, expr5},
//      {expr3*pow(expr1,2), expr3*pow(expr1,2), sqrt(2)*expr6, expr5, expr5},
//      {sqrt(2)*expr6, sqrt(2)*expr6, 2*expr7, sqrt(2)*expr8, sqrt(2)*expr8},
//      {expr5, expr5, sqrt(2)*expr8, expr3*pow(expr4,2), expr3*pow(expr4,2)},
//      {expr5, expr5, sqrt(2)*expr8, expr3*pow(expr4,2), expr3*pow(expr4,2)}
//  };
//
//  double B[][3] = {{B00, B01, B02},{B10, B11, B12},{B20,B21,B22}};
}

//FIXME: get rid of the ugly globals!
static gsl_vector *S[3][3];
static uint8_t dim[] = {1,1,2};

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

void DoFirstSVD(DTensor5 result[2], DTensor4 *B, uint8_t dc, double condi) {
  uint8_t sv_list[9*dc][3];
  uint8_t sv_num = 0;
  gsl_matrix *U[3][3];
  gsl_matrix *V[3][3];
  //FIXME: make GetGSLMatrix find size from tensor
  uint8_t msize[3][3] = {{3,1,1},{1,3,1},{1,1,5}};
  for (uint8_t rho_M = 0; rho_M < 3; ++rho_M)
    for (uint8_t rho_N = 0; rho_N < 3; ++rho_N) {
//      if (msize[rho_M][rho_N] == 0)
//        continue;
      U[rho_M][rho_N] = B->GetGSLMatrix(rho_M, rho_N, 2, 3, msize[rho_M][rho_N],
          msize[rho_M][rho_N]);
      S[rho_M][rho_N] = gsl_vector_alloc(msize[rho_N][rho_M]);
      V[rho_M][rho_N] = gsl_matrix_alloc(msize[rho_N][rho_M],
          msize[rho_N][rho_M]);
      gsl_linalg_SV_decomp_jacobi(U[rho_M][rho_N], V[rho_M][rho_N],
          S[rho_M][rho_N]);
      for (uint8_t i = 0; i < std::min(msize[rho_M][rho_N], dc); ++i)
        if (abs(gsl_vector_get(S[rho_M][rho_N], i)) > condi) {
          sv_list[sv_num][0] = rho_M;
          sv_list[sv_num][1] = rho_N;
          sv_list[sv_num][2] = i;
          ++sv_num;
        }
    };
  qsort(sv_list, sv_num, sizeof(sv_list[0]), CompareSVs);
  uint8_t rho_A[3][3][5] = {{{0,1,2},{2},{2}},
      {{2},{0,1,2},{2}}, {{2},{2},{0,2,2,1,2}}};
  uint8_t rho_B[3][3][5] = {{{0,1,2},{2},{2}},
      {{2},{1,0,2},{2}}, {{2},{2},{2,0,2,2,1}}};
  uint8_t rho_M;
  uint8_t rho_N;
  uint8_t i;
  for (uint8_t n = 0; n < std::min(sv_num, dc); ++n) {
    rho_M = sv_list[n][0];
    rho_N = sv_list[n][1];
    i = sv_list[n][2];
    for (uint8_t m = 0; m < msize[rho_M][rho_N]; ++m) {
      result[0].Set(i, rho_M, rho_N, rho_A[rho_M][rho_N][m],
          rho_B[rho_M][rho_N][m],
          sqrt(gsl_vector_get(S[rho_M][rho_N], i) * dim[rho_M] * dim[rho_N]) *
          gsl_matrix_get(U[rho_M][rho_N], m, i));
      result[1].Set(i, rho_M, rho_N, rho_A[rho_M][rho_N][m],
          rho_B[rho_M][rho_N][m],
          sqrt(gsl_vector_get(S[rho_M][rho_N], i) * dim[rho_M] * dim[rho_N]) *
          gsl_matrix_get(V[rho_M][rho_N], m, i));
    }
  }
}

void DoFirstContraction(DTensor14 *C, const DTensor9 &K, const DTensor5 &SU,
    const DTensor5 &SV) {
  DTensor9 SUSU;
  Contract(&SUSU, SU, 3, SU, 4);
  DTensor17 KSUSU;
  Contract(&KSUSU, K, 1, SUSU, 1);
  DTensor16 KSUSU1;
  ContractSelf(&KSUSU1, KSUSU, 2, 14);
  DTensor15 KSUSU2;
  ContractSelf(&KSUSU2, KSUSU1, 5, 12);
  DTensor14 KSUSU3;
  ContractSelf(&KSUSU3, KSUSU2, 6, 11);
  DTensor13 KSUSU4;
  ContractSelf(&KSUSU4, KSUSU3, 7, 13);
  DTensor9 SVSV;
  Contract(&SVSV, SV, 4, SV, 3);
  DTensor17 KSVSV;
  Contract(&KSVSV, K, 0, SVSV, 2);
  DTensor16 KSVSV1;
  ContractSelf(&KSVSV1, KSVSV, 3, 15);
  DTensor15 KSVSV2;
  ContractSelf(&KSVSV2, KSVSV1, 4, 12);
  DTensor14 KSVSV3;
  ContractSelf(&KSVSV3, KSVSV2, 5, 11);
  DTensor13 KSVSV4;
  ContractSelf(&KSVSV4, KSVSV3, 7, 13);
  DTensor25 KKSUSUSVSV;
  Contract(&KKSUSUSVSV, KSUSU4, 0, KSVSV4, 10);
  DTensor24 KKSUSUSVSV1;
  ContractSelf(&KKSUSUSVSV1, KKSUSUSVSV, 3, 24);
  DTensor22 KKSUSUSVSV2;
  ContractSelf2(&KKSUSUSVSV2, KKSUSUSVSV1, 4, 17);
  DTensor20 KKSUSUSVSV3;
  ContractSelf2(&KKSUSUSVSV3, KKSUSUSVSV2, 4, 16);
  DTensor18 KKSUSUSVSV4;
  ContractSelf2(&KKSUSUSVSV4, KKSUSUSVSV3, 4, 15);
  DTensor16 KKSUSUSVSV5;
  ContractSelf2(&KKSUSUSVSV5, KKSUSUSVSV4, 4, 14);
  DTensor15 KKSUSUSVSV6;
  ContractSelf(&KKSUSUSVSV6, KKSUSUSVSV5, 6, 10);
  ContractSelf(C, KKSUSUSVSV6, 8, 10);
}

void TRGS3(double a, double b, double c, uint8_t dc, double condi,
    uint8_t iter) {
  DTensor9 K;
  Make9j(&K);
  std::cout << K << std::endl;
  DTensor4 B;
  MakeFirstBlocks(&B, a, b, c);
  std::cout << B << std::endl;
  DTensor5 SVD[2];
  DoFirstSVD(SVD, &B, dc, condi);
  DTensor5 SU = SVD[0];
  DTensor5 SV = SVD[1];
  std::cout << SU << std::endl;
  std::cout << SV << std::endl;
  DTensor14 C;
  DoFirstContraction(&C, K, SU, SV);
  std::cout << C << std::endl;
}

int main(int argc, char **argv) {
  TRGS3(0,0,0,9,1e-8,1);
}
