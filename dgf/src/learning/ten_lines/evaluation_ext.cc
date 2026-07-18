// C++ implementation of evaluation metrics for classifiers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "dgf/src/util/nanobind_util.h"
#include "dgf/src/util/status_caster.h"

namespace nb = nanobind;

namespace dgf {

class ClassificationEvaluationAccumulator {
 public:
  ClassificationEvaluationAccumulator(int num_classes, int num_bins)
      : num_classes_(num_classes), num_bins_(num_bins) {
    per_classes_.reserve(num_classes);
    for (int i = 0; i < num_classes; ++i) {
      per_classes_.push_back(
          PerClassHistogram{std::vector<uint64_t>(num_bins, 0),
                            std::vector<uint64_t>(num_bins, 0)});
    }
  }

  absl::Status AddPredictions(
      const nb::ndarray<const float, nb::numpy, nb::shape<-1, -1>>& predictions,
      const nb::ndarray<const int32_t, nb::numpy, nb::shape<-1>>& targets) {
    auto pred_view = predictions.view();
    auto target_view = targets.view();

    size_t num_examples = pred_view.shape(0);
    size_t num_classes = pred_view.shape(1);

    if (num_classes != static_cast<size_t>(num_classes_)) {
      return absl::InvalidArgumentError(
          "Predictions shape(1) does not match num_classes: " +
          std::to_string(num_classes) + " vs " + std::to_string(num_classes_));
    }

    if (target_view.shape(0) != num_examples) {
      return absl::InvalidArgumentError(
          "Targets shape does not match predictions shape(0): " +
          std::to_string(target_view.shape(0)) + " vs " +
          std::to_string(num_examples));
    }

    for (size_t i = 0; i < num_examples; i++) {
      int32_t true_class = target_view(i);
      if (true_class < 0 || true_class >= num_classes_) {
        return absl::InvalidArgumentError("Target class index out of range: " +
                                          std::to_string(true_class));
      }
      for (int c = 0; c < num_classes_; c++) {
        float score = pred_view(i, c);
        score = std::clamp(score, 0.0f, 1.0f);
        int bin = std::min(static_cast<int>(score * num_bins_), num_bins_ - 1);
        if (true_class == c) {
          per_classes_[c].pos_histogram[bin]++;
        } else {
          per_classes_[c].neg_histogram[bin]++;
        }
      }
    }
    return absl::OkStatus();
  }

  nb::list ExtractMetrics() {
    nb::list result;
    for (int c = 0; c < num_classes_; c++) {
      result.append(ExtractClassMetrics(c));
    }
    return result;
  }

 private:
  struct PerClassHistogram {
    std::vector<uint64_t> pos_histogram;
    std::vector<uint64_t> neg_histogram;
  };

  int num_classes_;
  int num_bins_;
  std::vector<PerClassHistogram> per_classes_;

  nb::dict ExtractClassMetrics(int c) {
    const auto& pos_hist = per_classes_[c].pos_histogram;
    const auto& neg_hist = per_classes_[c].neg_histogram;

    uint64_t total_pos = 0;
    uint64_t total_neg = 0;
    for (int b = 0; b < num_bins_; b++) {
      total_pos += pos_hist[b];
      total_neg += neg_hist[b];
    }

    std::vector<double> tpr;
    std::vector<double> fpr;
    std::vector<double> recall;
    std::vector<double> thresholds;
    std::vector<uint64_t> tp_list;
    std::vector<uint64_t> fp_list;
    std::vector<uint64_t> tn_list;
    std::vector<uint64_t> fn_list;

    tpr.reserve(num_bins_ + 1);
    fpr.reserve(num_bins_ + 1);
    recall.reserve(num_bins_ + 1);
    thresholds.reserve(num_bins_ + 1);
    tp_list.reserve(num_bins_ + 1);
    fp_list.reserve(num_bins_ + 1);
    tn_list.reserve(num_bins_ + 1);
    fn_list.reserve(num_bins_ + 1);

    uint64_t tp = 0;
    uint64_t fp = 0;

    // Start with threshold > 1.0 (all negative predictions)
    tpr.push_back(0.0);
    fpr.push_back(0.0);
    recall.push_back(0.0);
    thresholds.push_back(1.0 + 1.0 / num_bins_);
    tp_list.push_back(0);
    fp_list.push_back(0);
    tn_list.push_back(total_neg);
    fn_list.push_back(total_pos);

    for (int b = num_bins_ - 1; b >= 0; b--) {
      tp += pos_hist[b];
      fp += neg_hist[b];

      double cur_tpr =
          total_pos > 0 ? static_cast<double>(tp) / total_pos : 0.0;
      double cur_fpr =
          total_neg > 0 ? static_cast<double>(fp) / total_neg : 0.0;
      double cur_recall = cur_tpr;

      tpr.push_back(cur_tpr);
      fpr.push_back(cur_fpr);
      recall.push_back(cur_recall);
      thresholds.push_back(static_cast<double>(b) / num_bins_);
      tp_list.push_back(tp);
      fp_list.push_back(fp);
      tn_list.push_back(total_neg - fp);
      fn_list.push_back(total_pos - tp);
    }

    // ROC AUC
    double auc = 0.0;
    for (size_t i = 1; i < tpr.size(); i++) {
      auc += 0.5 * (tpr[i] + tpr[i - 1]) * (fpr[i] - fpr[i - 1]);
    }

    // PR AUC (Davis-Goadrich)
    double pr_auc = 0.0;
    for (size_t i = 1; i < recall.size(); i++) {
      double r_prev = recall[i - 1];
      double r_curr = recall[i];

      if (r_curr == r_prev) {
        continue;
      }

      double tp_prev = static_cast<double>(tp_list[i - 1]);
      double tp_curr = static_cast<double>(tp_list[i]);
      double fp_prev = static_cast<double>(fp_list[i - 1]);
      double fp_curr = static_cast<double>(fp_list[i]);

      double d_tp = tp_curr - tp_prev;
      double d_fp = fp_curr - fp_prev;

      double slope = d_fp / d_tp;
      double u = 1.0 + slope;
      double h = fp_prev - slope * tp_prev;

      auto eval_integral = [u, h](double tp) {
        if (std::abs(h) < 1e-9) {
          return tp / u;
        }
        double val = u * tp + h;
        if (val <= 0.0) {
          return 0.0;
        }
        return (u * tp - h * std::log(val)) / (u * u);
      };

      double integral = eval_integral(tp_curr) - eval_integral(tp_prev);
      pr_auc += integral / total_pos;
    }

    nb::dict class_metrics;
    class_metrics["auc"] = auc;
    class_metrics["pr_auc"] = pr_auc;
    class_metrics["tp"] =
        CCVectorToNumpyArray<uint64_t>(std::span<const uint64_t>(tp_list));
    class_metrics["fp"] =
        CCVectorToNumpyArray<uint64_t>(std::span<const uint64_t>(fp_list));
    class_metrics["tn"] =
        CCVectorToNumpyArray<uint64_t>(std::span<const uint64_t>(tn_list));
    class_metrics["fn"] =
        CCVectorToNumpyArray<uint64_t>(std::span<const uint64_t>(fn_list));
    class_metrics["thresholds"] =
        CCVectorToNumpyArray<double>(std::span<const double>(thresholds));

    return class_metrics;
  }
};

}  // namespace dgf

NB_MODULE(evaluation_ext, m) {
  nb::class_<dgf::ClassificationEvaluationAccumulator>(
      m, "ClassificationEvaluationAccumulator")
      .def(nb::init<int, int>(), nb::arg("num_classes"), nb::arg("num_bins"))
      .def("add_predictions",
           dgf::ThrowIfErrorWrapper(
               &dgf::ClassificationEvaluationAccumulator::AddPredictions))
      .def("extract_metrics",
           &dgf::ClassificationEvaluationAccumulator::ExtractMetrics);
}
