#include <stdint.h>
#include <math.h>

#include "tensor.h"

DTensor9 Make9j() {
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

  DTensor9 K;
  uint8_t ind1[] = {1,2,3};
  uint8_t ind2[] = {1,2,4};

#define F(x) \
  for (uint8_t x = 0; x < 3; ++x) \
    for (uint8_t a##x = ind1[x]; a##x <= ind2[x]; ++a##x)

  F(i) F(j) F(k) F(l) F(m) F(n) F(o) F(p) F(q)
    K.Set(i,j,k,l,m,n,o,p,q,
        K.Get(i,j,k,l,m,n,o,p,q)+
        PP.Get(ai,aj,aq)*PP.Get(ak,al,aq)*PP.Get(an,am,ai)*PP.Get(ao,an,aj)*
        PP.Get(ap,ao,ak)*PP.Get(am,ap,al));

  return K;
}

DTensor4 MakeFirstBlocks(double a, double b, double c) {
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

  DTensor4 B;

  B.Set(0,0,0,0,pow(expr1,4));
  B.Set(0,0,0,1,pow(expr2,2));
  B.Set(0,0,0,2,2*expr3*pow(expr1,2));
  B.Set(0,0,1,0,pow(expr2,2));
  B.Set(0,0,1,1,pow(expr4,4));
  B.Set(0,0,1,2,2*expr3*pow(expr4,2));
  B.Set(0,0,2,0,2*expr3*pow(expr1,2));
  B.Set(0,0,2,1,2*expr3*pow(expr4,2));
  B.Set(0,0,2,2,4*pow(expr3,2));

  B.Set(0,2,0,0,8*pow(a - b,2)*pow(-1 + c,2));

  B.Set(1,1,0,0,pow(expr2,2));
  B.Set(1,1,0,1,pow(expr2,2));
  B.Set(1,1,0,2,2*expr5);
  B.Set(1,1,1,0,pow(expr2,2));
  B.Set(1,1,1,1,pow(expr2,2));
  B.Set(1,1,1,2,2*expr5);
  B.Set(1,1,2,0,2*expr5);
  B.Set(1,1,2,1,2*expr5);
  B.Set(1,1,2,2,4*pow(expr9,2));

  B.Set(2,0,0,0,8*pow(a - b,2)*pow(-1 + c,2));

  B.Set(2,2,0,0,expr3*pow(expr1,2));
  B.Set(2,2,0,1,expr3*pow(expr1,2));
  B.Set(2,2,0,2,sqrt(2)*expr6);
  B.Set(2,2,0,3,expr5);
  B.Set(2,2,0,4,expr5);
  B.Set(2,2,1,0,expr3*pow(expr1,2));
  B.Set(2,2,1,1,expr3*pow(expr1,2));
  B.Set(2,2,1,2,sqrt(2)*expr6);
  B.Set(2,2,1,3,expr5);
  B.Set(2,2,1,4,expr5);
  B.Set(2,2,2,0,sqrt(2)*expr6);
  B.Set(2,2,2,1,sqrt(2)*expr6);
  B.Set(2,2,2,2,2*expr7);
  B.Set(2,2,2,3,sqrt(2)*expr8);
  B.Set(2,2,2,4,sqrt(2)*expr8);
  B.Set(2,2,3,0,expr5);
  B.Set(2,2,3,1,expr5);
  B.Set(2,2,3,2,sqrt(2)*expr8);
  B.Set(2,2,3,3,expr3*pow(expr4,2));
  B.Set(2,2,3,4,expr3*pow(expr4,2));
  B.Set(2,2,4,0,expr5);
  B.Set(2,2,4,1,expr5);
  B.Set(2,2,4,2,sqrt(2)*expr8);
  B.Set(2,2,4,3,expr3*pow(expr4,2));
  B.Set(2,2,4,4,expr3*pow(expr4,2));

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

  return B;
}

void TRGS3(double a, double b, double c, uint8_t dc, double condi, uint8_t iter) {
  DTensor9 K = Make9j();
  std::cout << K << std::endl;

  DTensor4 B = MakeFirstBlocks(a, b, c);
  std::cout << B << std::endl;
}

int main(int argc, char **argv) {
  TRGS3(0,0,0,9,1e-9,1);
}
