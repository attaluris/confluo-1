#ifndef TEST_COUNT_SKETCH_TEST_H_
#define TEST_COUNT_SKETCH_TEST_H_

#include <functional>
#include <unordered_map>

#include "container/sketch/count_sketch.h"
#include "sketch_test_utils.h"
#include "gtest/gtest.h"

using namespace ::confluo::sketch;

class CountSketchTest : public testing::Test {
 public:
  typedef std::unordered_map<int, uint64_t> hist_t;

  template<typename T, typename P>
  static void bounded_pq_insert(thread_unsafe_pq<T, P> &queue, T key, P priority, size_t k) {
    assert_throw(queue.size() <= k, "Queue can't be larger than " + std::to_string(k) + " elements");
    assert_throw(!queue.contains(key), "Key " + std::to_string(key) + " is not unique");
    if (queue.size() < k) {
      queue.pushp(key, priority);
    }
    else if (queue.top().priority < priority) {
      queue.pop();
      queue.pushp(key, priority);
    }
  }

  static void run(hist_t &hist, double epsilon, double gamma, size_t k) {
    auto cs = count_sketch<int>::create_parameterized(epsilon, gamma);
    ASSERT_GT(cs.width(), 8 * k);

    auto start = utils::time_utils::cur_ns();
    for (auto p : hist) {
      cs.update(p.first, p.second);
    }
    auto stop = utils::time_utils::cur_ns();
    LOG_INFO << "Size: " << cs.storage_size() / 1024 <<" KB, Update Latency: " << (stop - start) / hist.size() << " ns";

    thread_unsafe_pq<int, int64_t> estimated_heavy_hitters, actual_heavy_hitters;
    for (auto p : hist) {
      int64_t est = cs.estimate(p.first);
      bounded_pq_insert<int, int64_t>(estimated_heavy_hitters, p.first, est, k);
      bounded_pq_insert<int, int64_t>(actual_heavy_hitters, p.first, p.second, k);
    }

    int64_t smallest_actual = actual_heavy_hitters.top().priority;
    std::vector<double> errors;

    // Invariant defined by Charikhar count-sketch paper:
    // k elements such that every element i has actual frequency ni > (1 - e)nk
    for (auto hh : estimated_heavy_hitters) {
      ASSERT_GT(hist[hh.key], (1 - epsilon) * smallest_actual);
      auto error = std::abs(int64_t(hist[hh.key]) - hh.priority) * 1.0 / hist[hh.key];
      errors.push_back(error);
    }
  }

  static void run_update(hist_t &hist, double epsilon, size_t depth, size_t k) {
    size_t width = count_sketch<int>::error_margin_to_width(epsilon);
    auto cs = count_sketch<int>(width, depth);
    ASSERT_GT(cs.width(), 8 * k);

    auto start = utils::time_utils::cur_ns();
    for (auto p : hist) {
      cs.update(p.first, p.second);
    }
    auto stop = utils::time_utils::cur_ns();
    LOG_INFO << "Depth: " << depth << ", Size: " << cs.storage_size() / 1024 <<" KB, Update Latency: " << (stop - start) / hist.size() << " ns";

    thread_unsafe_pq<int, int64_t> estimated_heavy_hitters, actual_heavy_hitters;
    for (auto p : hist) {
      int64_t est = cs.estimate(p.first);
      bounded_pq_insert<int, int64_t>(estimated_heavy_hitters, p.first, est, k);
      bounded_pq_insert<int, int64_t>(actual_heavy_hitters, p.first, p.second, k);
    }

    int64_t smallest_actual = actual_heavy_hitters.top().priority;
    std::vector<double> errors;

    // Invariant defined by Charikhar count-sketch paper:
    // k elements such that every element i has actual frequency ni > (1 - e)nk
    for (auto hh : estimated_heavy_hitters) {
      ASSERT_GT(hist[hh.key], (1 - epsilon) * smallest_actual);
      auto error = std::abs(int64_t(hist[hh.key]) - hh.priority) * 1.0 / hist[hh.key];
      errors.push_back(error);
    }
  }

  static void run_update_estimate(hist_t &hist, double epsilon, size_t depth, size_t k) {
    size_t width = count_sketch<int>::error_margin_to_width(epsilon);
    auto cs = count_sketch<int>(width, depth);
    ASSERT_GT(cs.width(), 8 * k);

    auto start = utils::time_utils::cur_ns();
    for (auto p : hist) {
      cs.update_and_estimate(p.first, p.second);
    }
    auto stop = utils::time_utils::cur_ns();
    LOG_INFO << "Depth: " << depth << ", Size: " << cs.storage_size() / 1024 <<" KB, Update_and_estimate Latency: " << (stop - start) / hist.size() << " ns";

    thread_unsafe_pq<int, int64_t> estimated_heavy_hitters, actual_heavy_hitters;
    for (auto p : hist) {
      int64_t est = cs.estimate(p.first);
      bounded_pq_insert<int, int64_t>(estimated_heavy_hitters, p.first, est, k);
      bounded_pq_insert<int, int64_t>(actual_heavy_hitters, p.first, p.second, k);
    }

    int64_t smallest_actual = actual_heavy_hitters.top().priority;
    std::vector<double> errors;

    // Invariant defined by Charikhar count-sketch paper:
    // k elements such that every element i has actual frequency ni > (1 - e)nk
    for (auto hh : estimated_heavy_hitters) {
      ASSERT_GT(hist[hh.key], (1 - epsilon) * smallest_actual);
      auto error = std::abs(int64_t(hist[hh.key]) - hh.priority) * 1.0 / hist[hh.key];
      errors.push_back(error);
    }
  }

};

/**
 * Tests that the CountSketch solves FindApproxTop(S, k, e) for b > 8k
 * http://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarCF.pdf
 */
TEST_F(CountSketchTest, InvariantTest) {
  size_t k = 10;
  hist_t hist;
  NormalGenerator(0, 100).sample(hist, 1000000);

  // TODO run multiple trials and determine gamma empirically to avoid randomness in testing
  // for now use very small gamma to enforce invariant and remove randomness
  double epsilon[] = { 0.01, 0.02, 0.04 };
  for (double e : epsilon)
    run(hist, e, 0.01, k);
}

TEST_F(CountSketchTest, LatencyTest) {
  size_t depth[] = { 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
  size_t k = 100;
  double e = 0.01;

  hist_t hist;
  NormalGenerator(0, 100).sample(hist, 1000000);

  for (size_t d : depth)
    run_update(hist, e, d, k);

  for (size_t d : depth)
    run_update_estimate(hist, e, d, k);
}

#ifdef STRESS_TEST
TEST_F(CountSketchTest, InvariantStressTest) {
  size_t k = 10;
  hist_t hist;
  ZipfGenerator().sample(hist, 1000000);

  double epsilon[] = { 0.01, 0.02, 0.04 };
  double gamma[] = { 0.01, 0.02, 0.04, 0.06, 0.08 };

  for (double e : epsilon)
    for (double g : gamma)
      run(hist, e, g, k);
}
#endif

#endif /* TEST_COUNT_SKETCH_TEST */
