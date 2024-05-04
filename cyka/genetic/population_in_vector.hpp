//
// Created by Joseph on 2024/5/1.
//

#ifndef CYKA_POPULATION_IN_VECTOR_HPP
#define CYKA_POPULATION_IN_VECTOR_HPP
#include "population_base.hpp"
#include <deque>
#include <vector>

namespace cyka::genetic {

template <template <class element, class allocator> class deque_like>
concept is_random_accessible =
    requires(deque_like<int, std::allocator<int>> array, size_t index) {
      array[index];
      array.erase(index);
      array.emplace_back(0);
    };

// template <class gene,
//           template <class element, class allocator> class deque_like =
//           std::vector, class allocator = std::allocator<gene>>
//   requires is_random_accessible<deque_like>

template <class gene, class allocator = std::allocator<gene>>
class population_in_vector
    : public population_base<gene, gene &, const gene &> {
protected:
  std::vector<gene, allocator> genes;

public:
  [[nodiscard]] size_t population_size() const noexcept override {
    return this->genes.size();
  }
  [[nodiscard]] gene &gene_at(size_t idx) noexcept override {
    return this->genes[idx];
  }
  [[nodiscard]] const gene &gene_at(size_t idx) const noexcept override {
    return this->genes[idx];
  }

  void set_gene_at(size_t index, const gene &g) noexcept override {
    this->genes[index] = g;
  }

  void
  reset(size_t num_population,
        const std::function<void(gene &)> &init_function) noexcept override {
    this->genes.clear();
    this->genes.reserve(num_population);
    for (size_t i = 0; i < num_population; i++) {
      gene g{};
      init_function(g);
      this->genes.emplace_back(std::move(g));
    }
  }

  void crossover(std::span<const std::pair<size_t, size_t>> crossover_list,
                 const std::function<crossover_function<const gene &, gene &>>
                     &crossover_function) noexcept override {
    const size_t size_before = this->genes.size();
    this->genes.reserve(size_before + 2 * crossover_list.size());
    for (const auto &[aidx, bidx] : crossover_list) {
      assert(aidx < size_before);
      assert(bidx < size_before);
      gene c{}, d{};
      crossover_function(this->genes[aidx], this->genes[bidx], c, d);
      this->genes.emplace_back(std::move(c));
      this->genes.emplace_back(std::move(d));
    }
  }

  void mutate(std::span<const size_t> mutate_list,
              const std::function<mutate_function<const gene &, gene &>>
                  &mutate_function) noexcept override {
    const size_t size_before = this->genes.size();
    this->genes.reserve(size_before + mutate_list.size());
    for (size_t src_idx : mutate_list) {
      assert(src_idx < size_before);
      gene b{};
      mutate_function(this->genes[src_idx], b);
      this->genes.emplace_back(std::move(b));
    }
  }

  void select(std::span<const uint16_t> LUT_selected_count) noexcept override {
    assert(LUT_selected_count.size() == this->population_size());

    std::vector<gene, allocator> new_genes;
    new_genes.reserve(this->genes.size());
    for (auto src_idx = 0zu; src_idx < LUT_selected_count.size(); src_idx++) {
      if (LUT_selected_count[src_idx] <= 0) {
        continue;
      }

      for (uint16_t i = 0; i + 1 < LUT_selected_count[src_idx]; i++) {
        new_genes.emplace_back(this->genes[src_idx]);
      }
      new_genes.emplace_back(std::move(this->genes[src_idx]));
    }
    this->genes = new_genes;

    static_assert(is_population<population_in_vector>);
  }
};

} // namespace cyka::genetic
#endif // CYKA_POPULATION_IN_VECTOR_HPP
