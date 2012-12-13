#include <stdint.h>
#include <math.h>

#include "tensor.h"

DTensor9 Make9j(const DTensor3 &PP) {
  DTensor9 K;
  uint8_t ind1[] = {1,2,3};
  uint8_t ind2[] = {1,2,4};

#define F(x) \
  for (uint8_t x = 0; x < 3; ++x) \
    for (uint8_t a##x = ind1[x]; a##x <= ind2[x]; ++a##x)

  F(i) F(j) F(k) F(l) F(m) F(n) F(o) F(p) F(q)
    K.Set(i,j,k,l,m,n,o,p,q,
        PP.Get(ai,aj,aq)*PP.Get(ak,al,aq)*PP.Get(an,am,ai)*PP.Get(ao,an,aj)*
        PP.Get(ap,ao,ak)*PP.Get(am,ap,al));

  return K;
}

void TRGS3(double a, double b, double c, double dc, double condi, double iter) {
  DTensor3 PP;
  PP.Set(1,1,1,1);
  PP.Set(1,2,2,1);
  PP.Set(1,3,3,1/sqrt(2));
  PP.Set(1,4,4,1/sqrt(2));
  PP.Set(2,1,2,1);
  PP.Set(2,2,1,1);
  PP.Set(2,3,4,-1/sqrt(2));
  PP.Set(2,4,3,1/sqrt(2));
  PP.Set(3,1,3,1/sqrt(2));
  PP.Set(3,2,4,-1/sqrt(2));
  PP.Set(3,3,1,1/sqrt(2));
  PP.Set(3,3,3,-1/2);
  PP.Set(3,4,2,1/sqrt(2));
  PP.Set(3,4,4,1/2);
  PP.Set(4,1,4,1/sqrt(2));
  PP.Set(4,2,3,1/sqrt(2));
  PP.Set(4,3,2,-1/sqrt(2));
  PP.Set(4,3,4,1/2);
  PP.Set(4,4,1,1/sqrt(2));
  PP.Set(4,4,3,1/2);
  DTensor9 K = Make9j(PP);
  std::cout << K << std::endl;
}

int main(int argc, char **argv) {
  TRGS3(1,0,0,9,1e-9,1);
}
