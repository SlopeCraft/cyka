#include <Eigen/Dense>
#include <cyka/genetic/crossover.hpp>
#include <iostream>
#include <random>

int main(int, char **) {
  namespace cg = cyka::genetic;
  using vxf = Eigen::ArrayXf;
  using mxf_mut = Eigen::Map<Eigen::ArrayXf>;
  using mxf_const = Eigen::Map<const Eigen::ArrayXf>;

  constexpr size_t dim = 5;
  const Eigen::ArrayXf p1 = Eigen::ArrayXf::LinSpaced(dim, 0.0, 4.0);
  const Eigen::ArrayXf p2 = Eigen::ArrayXf::LinSpaced(dim, 5.0, 9.0);
  assert(p1.size() == dim);

  std::mt19937 rand{std::random_device{}()};

  Eigen::ArrayXf c1, c2;
  auto print_value = [dim, &p1, &p2](const char *name, auto c1, auto c2) {
    Eigen::Array<float, Eigen::Dynamic, 4> temp{dim, 4};
    temp.col(0) = p1;
    temp.col(1) = p2;
    temp.col(2) = c1;
    temp.col(3) = c2;
    std::cout << name << " crossover: \n" << temp << std::endl;
  };
  {
    cg::arithmetic_crossover<vxf &, const vxf &> c;
    c.crossover(p1, p2, c1, c2, rand);
    print_value("[ref] arithmetic", c1, c2);
  }
  {
    cg::uniform_crossover<vxf &, const vxf &> c;
    c.crossover(p1, p2, c1, c2, rand);
    print_value("[ref] uniform", c1, c2);
  }
  {
    cg::single_point_crossover<vxf &, const vxf &> c;
    c.crossover(p1, p2, c1, c2, rand);
    print_value("[ref] single point", c1, c2);
  }
  {
    cg::multi_point_crossover<vxf &, const vxf &> c;
    c.crossover(p1, p2, c1, c2, rand);
    print_value("[ref] multi point", c1, c2);
  }

  mxf_const p1m{p1.data(), p1.size()}, p2m{p2.data(), p2.size()};
  c1.setZero(p1.size());
  c2.setZero(p2.size());
  mxf_mut c1m{c1.data(), c1.size()}, c2m{c2.data(), c2.size()};

  {
    cg::arithmetic_crossover<mxf_mut, mxf_const> c;
    c.crossover(p1m, p2m, c1m, c2m, rand);
    print_value("[map] arithmetic", c1m, c2m);
  }
  {
    cg::uniform_crossover<mxf_mut, mxf_const> c;
    c.crossover(p1m, p2m, c1m, c2m, rand);
    print_value("[map] uniform", c1, c2);
  }
  {
    cg::single_point_crossover<mxf_mut, mxf_const> c;
    c.crossover(p1m, p2m, c1m, c2m, rand);
    print_value("[map] single point", c1, c2);
  }
  {
    cg::multi_point_crossover<mxf_mut, mxf_const> c;
    c.crossover(p1m, p2m, c1m, c2m, rand);
    print_value("[map] multi point", c1, c2);
  }

  return 0;
}