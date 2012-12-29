/* -*- indent-tabs-mode: nil -*- */

#include <stdint.h>

#include "tensor.h"

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
  Contract2(&t6, t5, 1, t5, 1);

  DTensor3 t7;
  ContractSelf(&t7, t3, 2, 3);

  DTensor2 t8;
  ContractSelf2(&t8, t3, 0, 3);

  std::cout << t1 << std::endl;
  std::cout << t2 << std::endl;
  std::cout << t3 << std::endl;
  std::cout << t4 << std::endl;
  std::cout << t5 << std::endl;
  std::cout << t6 << std::endl;
  std::cout << t7 << std::endl;
  std::cout << t8 << std::endl;
}
