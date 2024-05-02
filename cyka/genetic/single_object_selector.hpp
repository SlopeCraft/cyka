//
// Created by Joseph on 2024/5/2.
//

#ifndef CYKA_SINGLE_OBJECT_SELECTOR_HPP
#define CYKA_SINGLE_OBJECT_SELECTOR_HPP

#include "selector.hpp"
#include <algorithm>
#include <map>

namespace cyka::genetic::SO_selector {

/**
 * \brief The Boltzmann method works by exponential function. The probability of
 * a gene is in proportion with exp(b*f), where b is the selection strength and
 * f is the fitness value.
 *
 * \note This method introduced one parameter: the selection strength b. For
 * maximum problems, b should be positive, while for minimize problems, b should
 * be a negative number.
 */
class boltzmann : public selector_base<1> {
public:
  struct option {
    double boltzmann_select_strength{-1.0};
  };

protected:
  option option_;

public:
  [[nodiscard]] auto &option() const noexcept { return this->option_; }
  void set_option(struct option &opt) noexcept { this->option_ = opt; }

  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<bool> &is_kept,
              std::mt19937 &rand_engine) noexcept override {
    assert(fitness.rows() == 1);
    assert(fitness.cols() >= expected_group_size);
    const size_t pop_size_before = fitness.cols();
    const size_t num_to_eliminate = pop_size_before - expected_group_size;
    is_kept.resize(int64_t(pop_size_before));
    is_kept.fill(true);
    if (num_to_eliminate <= 0) {
      return;
    }

    auto fitness_col = fitness.transpose();
    Eigen::ArrayXd probability_score =
        (fitness_col * this->option_.boltzmann_select_strength).exp();
    //    double score_sum = probability_score.sum();

    std::uniform_real_distribution<double> rand{0, 1};

    size_t eliminated_num = 0;
    while (eliminated_num < num_to_eliminate) {
      double r = rand(rand_engine) * probability_score.sum();
      for (size_t i = 0; i < pop_size_before; i++) {
        if (!is_kept[i]) {
          assert(probability_score[i] == 0);
          //          continue;
        }
        r -= probability_score[i];

        if (r <= 0) { // selected
          probability_score[i] = 0;
          is_kept[i] = false;
          eliminated_num++;
          break;
        }
      }
      if (r > 0) {
        assert(("Impossible condition, r should be less or equal to the sum of "
                "whole array",
                0));
      }
    }

    assert(is_kept.count() == expected_group_size);
  }
};
} // namespace cyka::genetic::SO_selector

#endif // CYKA_SINGLE_OBJECT_SELECTOR_HPP
