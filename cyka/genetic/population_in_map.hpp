//
// Created by Joseph on 2024/5/1.
//

#ifndef CYKA_POPULATION_IN_MAP_HPP
#define CYKA_POPULATION_IN_MAP_HPP

#include "population_base.hpp"
#include <map>

namespace cyka::genetic {

template <class gene>
class population_in_map : public population_base<gene, gene &, const gene &> {
protected:
  std::map<size_t, gene> gene_map;

public:
  using base_t = population_base<gene, gene &, const gene &>;
  using typename base_t::const_gene_view_type;
  using typename base_t::mut_gene_view_type;

  [[nodiscard]] size_t population_size() const noexcept override {
    return this->gene_map.size();
  }
  [[nodiscard]] mut_gene_view_type gene_at(size_t idx) noexcept override {
    return this->gene_map.at(idx);
  }
  [[nodiscard]] const_gene_view_type
  gene_at(size_t idx) const noexcept override {
    return this->gene_map.at(idx);
  }

  void set_gene_at(size_t index, const_gene_view_type g) noexcept override {
    this->gene_map[index] = g;
  }

  void reset(size_t num_population,
             const std::function<void(mut_gene_view_type)>
                 &init_function) noexcept override {
    this->gene_map.clear();
    for (size_t i = 0; i < num_population; i++) {
      gene g{};
      init_function(g);

      this->gene_map.emplace(i, std::move(g));
    }
  }

  void crossover(
      std::span<const std::pair<size_t, size_t>> crossover_list,
            const std::function<
                crossover_function<const_gene_view_type, mut_gene_view_type>>
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
              const std::function<
                  mutate_function<const_gene_view_type, mut_gene_view_type>>
                  &mutate_function) noexcept override {
    const size_t size_before = this->gene_map.size();
    for (size_t src_idx : mutate_list) {
      assert(src_idx < size_before);
      gene b{};
      mutate_function(this->gene_map.at(src_idx), b);
      this->gene_map.emplace(this->gene_map.size(), std::move(b));
    }
  }

  std::vector<size_t>
  select(std::span<const uint16_t> LUT_selected_count) noexcept override {
    assert(LUT_selected_count.size() == this->population_size());

    std::vector<size_t> LUT_new_to_old;
    LUT_new_to_old.reserve(this->population_size());
    std::map<size_t, gene> new_genes;
    size_t counter = 0;
    for (auto src_idx = 0zu; src_idx < this->population_size(); src_idx++) {
      if (LUT_selected_count[src_idx] <= 0) {
        continue;
      }
      // `this->gene_map.at(src_idx)` will be repeated N times, copy previous
      // N-1 genes
      for (auto i = 0zu; i + 1 < LUT_selected_count[src_idx]; i++) {
        new_genes.emplace(counter, this->gene_map.at(src_idx));
        LUT_new_to_old.emplace_back(src_idx);
        counter++;
      }
      // Move the last
      new_genes.emplace(counter, std::move(this->gene_map.at(src_idx)));
      LUT_new_to_old.emplace_back(src_idx);
      counter++;
    }

    //    for (size_t idx = 0; idx < this->population_size(); idx++) {
    //      if (!LUT_is_selected[idx]) {
    //        this->gene_map.erase(idx);
    //      }
    //    }
    //
    //    size_t counter = 0;
    //    for (auto it = this->gene_map.begin();;) {
    //      if (it == this->gene_map.end()) {
    //        break;
    //      }
    //      assert(it->first >= counter);
    //      if (it->first == counter) {
    //        counter++;
    //        ++it;
    //        continue;
    //      }
    //      assert(it->first > counter);
    //      assert(!this->gene_map.contains(counter));
    //
    //      this->gene_map.emplace(counter, std::move(it->second));
    //      it = this->gene_map.erase(it);
    //      counter++;
    //    }

    static_assert(is_population<population_in_map>);
    return LUT_new_to_old;
  }
};
} // namespace cyka::genetic

#endif // CYKA_POPULATION_IN_MAP_HPP
