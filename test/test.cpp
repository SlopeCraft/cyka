//
// Created by Joseph on 2024/5/1.
//
#include <Eigen/Dense>
#include <cyka/genetic/GA.hpp>
#include <cyka/genetic/population_in_map.hpp>
#include <cyka/genetic/population_in_matrix.hpp>
#include <cyka/genetic/population_in_vector.hpp>
#include <iostream>

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

  std::vector<Eigen::ArrayXf> vec;

  cyka::genetic::population_in_vector<Eigen::Array<float, 20, 1>> pop_vec;
}