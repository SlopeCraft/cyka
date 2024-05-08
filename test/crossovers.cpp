#include <Eigen/Dense>
#include <cyka/genetic/crossover.hpp>
#include <iostream>
#include <random>

int main(int, char **) {
  namespace cg = cyka::genetic;
  using vxf = Eigen::ArrayXf;

  const size_t dim = 5;
  const Eigen::ArrayXf p1 = Eigen::ArrayXf::LinSpaced(dim, 0.0, 4.0);
  const Eigen::ArrayXf p2 = Eigen::ArrayXf::LinSpaced(dim, 5.0, 9.0);
  assert(p1.size() == dim);
  cg::arithmetic_crossover<vxf &, const vxf &> ac_ref;

  std::mt19937 rand{0};

  Eigen::ArrayXf c1, c2;
  ac_ref.crossover(p1, p2, c1, c2, rand);
  assert(c1.size() == p1.size());
  assert(c2.size() == c2.size());
  Eigen::Array<float, Eigen::Dynamic, 4> temp{dim, 4};
  temp.col(0) = p1;
  temp.col(1) = p2;
  temp.col(2) = c1;
  temp.col(3) = c2;
  std::cout << "Arithmetic crossover: \n" << temp << std::endl;

  return 0;
}