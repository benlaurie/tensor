/* -*- indent-tabs-mode: nil -*- */

#include <stdint.h>

#include "tensor.h"

int RandomInt(unsigned range) {
  return random() % range;
}

double RandomDouble(double range) {
  return range * ((double)random())/(1U << 31);
}

template <rank_t rank> void RandomFill(Tensor<rank, double> *t, unsigned range,
                                       unsigned entries) {
  Coordinate<rank> coord;

  while (entries--) {
    for (rank_t n = 0; n < rank; ++n)
      coord.Set(n, RandomInt(range));
    t->Set(coord, RandomDouble(2) - 1);
  }
}

void TestContract() {
  DTensor3 t1;

  RandomFill(&t1, 3, 10);
  std::cout << t1 << std::endl;

  DTensor4 t2;
  Contract2(&t2, t1, 0, t1, 1);
  std::cout << t2 << std::endl;

  DTensor6 t3;
  Multiply(&t3, t1, t1);

  DTensor4 t4;
  ContractSelf2(&t4, t3, 0, 4);
  std::cout << t4 << std::endl;

  assert(t2 == t4);

  DTensor5 t5;
  Contract(&t5, t1, 2, t1, 0);

  DTensor5 t6;
  ContractSelf(&t6, t3, 2, 3);

  assert(t5 == t6);

  DTensor3 t7;
  RandomFill(&t7, 3, 5);

  DTensor3 t8;
  RandomFill(&t8, 3, 5);

  DTensor3 t9;
  RandomFill(&t9, 3, 5);

  DTensor5 t10;
  Contract(&t10, t7, 0, t8, 0);

  DTensor5 t11;
  Contract(&t11, t7, 0, t9, 0);

  DTensor9 t12;
  Contract(&t12, t10, 1, t11, 3);

  DTensor8 t13;
  ContractSelf(&t13, t12, 2, 6);
  std::cout << t13 << std::endl;

  DTensor4 t14;
  ContractSelf(&t14, t11, 1, 2);

  DTensor8 t15;
  Contract(&t15, t10, 1, t14, 2);

  rank_t mapping[8] = {0, 1, 6, 3, 4, 5, 2, 7};

  DTensor8 t16;
  Rearrange(&t16, t15, mapping);
  std::cout << t16 << std::endl;
}

int main(int argc, char **argv) {
  DTensor3 t1;

  t1.Set(1, 1, 2, 15);
  t1.Set(1, 1, 2, 23.1);
  t1.Set(1, 1, 3, 10);
  t1.Set(100, 234, 123, 42);

  Tensor<2, double> t2;
  uint8_t c3[] = { 2, 1 };
  t2.Set(c3, 1.5);
  uint8_t c4[] = { 2, 2 };
  t2.Set(c4, 2);
  uint8_t c6[] = { 3, 1 };
  t2.Set(c6, 2.5);

  Tensor<4, double> t3;
  Contract(&t3, t1, 2, t2, 0);

  Tensor<3, double> t4;
  Contract2(&t4, t1, 2, t2, 0);

  DTensor1 t5;
  t5.Set(1,34);

  Tensor<0, double> t6;
  Contract2(&t6, t5, 0, t5, 0);

  DTensor3 t7;
  ContractSelf(&t7, t3, 0, 1);

  DTensor2 t8;
  ContractSelf2(&t8, t3, 0, 1);

  DTensor2 t9;
  ContractSelf2(&t9, t3, 1, 0);

  DTensor3 t10;
  ContractSelf(&t10, t3, 1, 0);

  DTensor3 t11;
  rank_t mapping[3] = {2, 0, 1};
  Rearrange(&t11, t1, mapping);

  std::cout << t1 << std::endl;
  std::cout << t2 << std::endl;
  std::cout << t3 << std::endl;
  std::cout << t4 << std::endl;
  std::cout << t5 << std::endl;
  std::cout << t6 << std::endl;
  std::cout << t7 << std::endl;
  std::cout << t8 << std::endl;
  std::cout << t9 << std::endl;
  std::cout << t10 << std::endl;
  std::cout << t11 << std::endl;

  srandom(time(NULL));
  TestContract();
}
