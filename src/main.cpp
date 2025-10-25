#include "api/Api.hpp"
#include <iostream>

int main() {
  auto t = Tensor::api::make_scalar<int>(1);
  std::cout << *t.data() << std::endl;
  return 0;
}