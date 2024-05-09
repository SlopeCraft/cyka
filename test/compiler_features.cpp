#include "cyka/utils/size_t_literal.hpp"
#include <cstddef>
#include <cstdint>

int main(int, char **) {
  const auto size_literal = 0uz + 0zu;
  return 0;
}