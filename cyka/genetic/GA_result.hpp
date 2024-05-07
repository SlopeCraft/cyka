//
// Created by Joseph on 2024/5/7.
//

#ifndef CYKA_GA_RESULT_HPP
#define CYKA_GA_RESULT_HPP

#include <deque>
#include <vector>

#include <Eigen/Dense>

#include "fitness_computer.hpp"

namespace cyka::genetic {

template <class fitness_type, class fitness_matrix> struct GA_result {

  struct fitness_pair {
    fitness_matrix population_fitness;
    size_t best_gene_index;
  };

  std::vector<fitness_pair> fitness_history;
};
} // namespace cyka::genetic

#endif // CYKA_GA_RESULT_HPP
