//
// Created by Joseph on 2024/5/1.
//
#include <Eigen/Dense>
#include <cyka/genetic/GA.hpp>
#include <cyka/genetic/GA_system.hpp>
#include <cyka/genetic/crossover.hpp>
#include <cyka/genetic/population_in_map.hpp>
#include <cyka/genetic/population_in_matrix.hpp>
#include <cyka/genetic/population_in_vector.hpp>
#include <cyka/genetic/single_object_selector.hpp>
#include <iostream>

void initiate_SO_selectors() noexcept;

int main() {
  //  Eigen::Array<float, 5, 20> mat;
  //  //  mat.resize(5, 20);
  //  for (size_t i = 0; i < mat.size(); i++) {
  //    mat(i) = i;
  //  }
  //
  //  Eigen::Map<Eigen::Array<float, 5, 1>> m{&mat(0, 6), mat.rows()};
  //
  //  std::cout << "\nmat = \n" << mat;
  //  std::cout << "\nmat.col(6) = \n" << m;
  //
  //  return 0;
  cyka::genetic::population_in_matrix<float, 10> pop_10_1_col;
  cyka::genetic::population_in_map<Eigen::Array<float, 20, 1>> pop_map;
  cyka::genetic::population_in_vector<Eigen::Array<float, 20, 1>> pop_vec;

  cyka::genetic::GA_system<decltype(pop_10_1_col), 1> sys{
      [](auto) { return 0.0; }};
  initiate_SO_selectors();
}

void initiate_SO_selectors() noexcept {
  using namespace cyka::genetic::SO_selector;
  truncation trunc;
  tournament tour;
  monte_carlo mc;
  linear_rank lr;
  exponential_rank er;
  boltzmann b;
}

void initiate_crossovers() noexcept {
  using gene_t = Eigen::ArrayXd;
  cyka::genetic::arithmetic_crossover<Eigen::Map<gene_t>,
                                      Eigen::Map<const gene_t>>
      ac;
  cyka::genetic::uniform_crossover<Eigen::Map<gene_t>, Eigen::Map<const gene_t>>
      uc;
  cyka::genetic::single_point_crossover<Eigen::Map<gene_t>,
                                        Eigen::Map<const gene_t>>
      sc;
}