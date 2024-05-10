//
// Created by Joseph on 2024/5/7.
//

#ifndef CYKA_GA_RESULT_HPP
#define CYKA_GA_RESULT_HPP

#include <deque>
#include <vector>

#include <Eigen/Dense>

#include "loss_computer.hpp"

namespace cyka::genetic {

template <class loss_type, class loss_matrix> struct GA_result {

  struct loss_pair {
    loss_matrix population_loss;
    size_t best_gene_index;
  };

  std::vector<loss_pair> loss_history;
};
} // namespace cyka::genetic

#endif // CYKA_GA_RESULT_HPP
