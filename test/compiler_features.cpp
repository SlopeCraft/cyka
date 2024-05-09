#include "cyka/utils/size_t_literal.hpp"
#include <cstddef>
#include <cstdint>

#include <iostream>
#include <random>

int main(int, char **) {
  const auto size_literal = 0uz + 0zu;

  std::random_device rd;
  for (int i = 0; i < 10; i++) {
    std::cout << rd() << ", ";
  }
  std::cout << std::endl;
  return 0;
}