//
// Created by Joseph on 2024/5/1.
//

#ifndef CYKA_POPULATION_IN_MAP_HPP
#define CYKA_POPULATION_IN_MAP_HPP

#include "population.hpp"

namespace cyka::genetic {

template <class gene, class mut_gene_view, class const_gene_view,
          size_t n_objectives>
class population_in_map
    : public population<gene, mut_gene_view, const_gene_view, n_objectives> {
protected:
  std::map<size_t, gene> gene_map;

public:
  [[nodiscard]] size_t population_size() const noexcept override {
    return this->gene_map.size();
  }
  [[nodiscard]] mut_gene_view gene_at(size_t idx) noexcept override {
    return this->gene_map.at(idx);
  }
  [[nodiscard]] const_gene_view gene_at(size_t idx) const noexcept override {
    return this->gene_map.at(idx);
  }

  void set_gene_at(size_t index, const_gene_view g) noexcept override {
    this->gene_map[index] = g;
  }

  void reset(size_t num_population, const std::function<void(mut_gene_view)>
                                        &init_function) noexcept override {
    this->gene_map.clear();
    for (size_t i; i < num_population; i++) {
      gene g{};
      init_function.value()(g);

      this->gene_map.emplace(i, std::move(g));
    }
  }

  void crossover(
      std::span<const std::pair<size_t, size_t>> crossover_list,
      const std::function<crossover_function<const_gene_view, mut_gene_view>>
          &crossover_function) noexcept override {
    const size_t size_before = this->gene_map.size();
    for (const auto &[aidx, bidx] : crossover_list) {
      assert(aidx < size_before);
      assert(bidx < size_before);
      gene c{}, d{};
      crossover_function(this->gene_map.at(aidx), this->gene_map.at(bidx), c,
                         d);
      this->gene_map.emplace(this->gene_map.size(), std::move(c));
      this->gene_map.emplace(this->gene_map.size(), std::move(d));
    }
  }

  void
  mutate(std::span<const size_t> mutate_list,
         const std::function<mutate_function<const_gene_view, mut_gene_view>>
             &mutate_function) noexcept override {
    const size_t size_before = this->gene_map.size();
    for (size_t src_idx : mutate_list) {
      assert(src_idx < size_before);
      gene b{};
      mutate_function(this->gene_map.at(src_idx), b);
      this->gene_map.emplace(this->gene_map.size(), std::move(b));
    }
  }

  void select(std::span<const bool> LUT_is_selected) noexcept override {
    assert(LUT_is_selected.size() == this->population_size());
    for (size_t idx = 0; idx < this->population_size(); idx++) {
      if (!LUT_is_selected[idx]) {
        this->gene_map.erase(idx);
      }
    }

    size_t counter = 0;
    for (auto it = this->gene_map.begin();;) {
      if (it == this->gene_map.end()) {
        break;
      }
      assert(it->first >= counter);
      if (it->first == counter) {
        counter++;
        ++it;
        continue;
      }
      assert(it->first > counter);
      assert(!this->gene_map.contains(counter));

      this->gene_map.emplace(counter, std::move(it->second));
      it = this->gene_map.erase(it);
      counter++;
    }
  }
};
} // namespace cyka::genetic

#endif // CYKA_POPULATION_IN_MAP_HPP
