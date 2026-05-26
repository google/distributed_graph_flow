
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "dgf/src/util/concurrency.h"
#include "dgf/src/util/nanobind_util.h"
#include "dgf/src/util/status_caster.h"
#include "dgf/src/util/util.h"

namespace nb = nanobind;

namespace dgf {
namespace {

// A class that indexes byte string identifiers.
// It takes an array of byte strings and builds a mapping from each string to
// its integer index within that array. This mapping can then be used to
// efficiently look up the indices for other arrays of byte strings.
//
// Usage example (python):
//   >>> mapper = ByteIdToIdxMapper(np.array([b"X", b"Y"]))
//   >>> print(mapper(np.array([b"Y", b"X", b"Z"])))
//   (array([ 1,  0, -1]), 2)
class ByteIdToIdxMapper {
 public:
  using Idx = int64_t;
  using Ids = nb::object;
  using Idxs = nb::ndarray<Idx, nb::numpy, nb::shape<-1>>;

  // Create a mapper with an array of ids.
  ByteIdToIdxMapper(const Ids& ids) {
    // Create a view on the values.
    auto ids_array_or = NumpyBytesArray::Create(ids);
    if (!ids_array_or.ok()) {
      throw std::invalid_argument(std::string(ids_array_or.status().message()));
    }

    // Index the values.
    auto ids_array = std::move(ids_array_or.value());
    for (size_t idx = 0; idx < ids_array.size(); idx++) {
      index_[ids_array[idx]] = idx;
    }

    // Keep a pointer to the values to make sure they are not deleted.
    raw_ids_ = ids;
  }

  Idx value(std::string_view id, Idx i, Idx& mismatch) const {
    auto it = index_.find(id);
    if (it == index_.end()) {
      mismatch = i;
      return -1;  // Missing value
    }
    return it->second;
  }

  // Maps each byte string in `ids` to an index based on the constructor's
  // `ids`. Returns a pair: (1) indices for each input `id` (-1 if not found),
  // and (2) the index of the last unmatched `id` in the input (or -1 if all the
  // values are matched).
  absl::StatusOr<std::pair<Idxs, Idx>> operator()(const Ids& ids) const {
    DGF_ASSIGN_OR_RETURN(auto ids_array, NumpyBytesArray::Create(ids));

    // Query index.
    const std::size_t n = ids_array.size();
    Idx* idxs_data = new Idx[n];
    Idx mismatch = -1;
    {
      nb::gil_scoped_release release;
      for (size_t i = 0; i < n; i++) {
        idxs_data[i] = value(ids_array[i], i, mismatch);
      }
    }

    // Pack result into a numpy array.
    nanobind::capsule owner(idxs_data,
                            [](void* p) noexcept { delete[] (Idx*)p; });
    return std::pair(nanobind::ndarray<Idx, nb::numpy, nb::shape<-1>>(
                         /* data = */ idxs_data,
                         /* shape = */ {n},
                         /* owner = */ owner),
                     mismatch);
  }

 private:
  absl::flat_hash_map<std::string_view, Idx> index_;
  nb::object raw_ids_;
};

// Runs two ByteIdToIdxMapper::operator() and returns a single 2-dimensional
// array with both outputs stacked. This is equivalent, but more efficient, than
// running ByteIdToIdxMapper::operator() twice and stacking the results after.
absl::StatusOr<std::tuple<ByteIdToIdxMapper::Idxs, ByteIdToIdxMapper::Idx,
                          ByteIdToIdxMapper::Idx>>
PairMapping(const ByteIdToIdxMapper& mapper1, const ByteIdToIdxMapper& mapper2,
            const ByteIdToIdxMapper::Ids& ids1,
            const ByteIdToIdxMapper::Ids& ids2, const int num_threads) {
  DGF_ASSIGN_OR_RETURN(auto ids_array1, NumpyBytesArray::Create(ids1));
  DGF_ASSIGN_OR_RETURN(auto ids_array2, NumpyBytesArray::Create(ids2));

  if (ids_array1.size() != ids_array2.size()) {
    return absl::InvalidArgumentError(
        "Input arrays ids1 and ids2 must have the same size.");
  }

  // Query index.
  const size_t cols = ids_array1.size();
  const size_t rows = 2;
  auto* idxs_data = new ByteIdToIdxMapper::Idx[rows * cols];
  std::atomic<ByteIdToIdxMapper::Idx> mismatch1 = -1;
  std::atomic<ByteIdToIdxMapper::Idx> mismatch2 = -1;

  {
    nb::gil_scoped_release release;
    util::concurrency::ThreadPool pool(num_threads);

    // Find the optimal parameters.
    const size_t target_block_size = 1000;
    const size_t num_blocks =
        std::clamp((cols + target_block_size - 1) / target_block_size,
                   size_t{1}, size_t{100});

    util::concurrency::ConcurrentForLoop(
        num_blocks, &pool, cols,
        [cols, &mapper1, &mapper2, idxs_data, &mismatch1, &mismatch2,
         &ids_array1, &ids_array2](size_t block_idx, size_t begin_item_idx,
                                   size_t end_item_idx) {
          ByteIdToIdxMapper::Idx local_mismatch1 = -1;
          ByteIdToIdxMapper::Idx local_mismatch2 = -1;

          for (size_t i = begin_item_idx; i < end_item_idx; i++) {
            idxs_data[i] = mapper1.value(ids_array1[i], i, local_mismatch1);
          }
          for (size_t i = begin_item_idx; i < end_item_idx; i++) {
            idxs_data[i + cols] =
                mapper2.value(ids_array2[i], i, local_mismatch2);
          }
          if (local_mismatch1 != -1) {
            mismatch1 = local_mismatch1;
          }
          if (local_mismatch2 != -1) {
            mismatch2 = local_mismatch2;
          }
        });
  }

  // Pack result into a numpy array.
  nanobind::capsule owner(
      idxs_data, [](void* p) noexcept { delete[] (ByteIdToIdxMapper::Idx*)p; });
  return std::tuple(
      nanobind::ndarray<ByteIdToIdxMapper::Idx, nb::numpy, nb::shape<-1>>(
          /* data = */ idxs_data,
          /* shape = */ {rows, cols},
          /* owner = */ owner),
      mismatch1.load(), mismatch2.load());
}

}  // namespace

NB_MODULE(io_ext, m) {
  nb::class_<ByteIdToIdxMapper>(m, "ByteIdToIdxMapper")
      .def(nb::init<const ByteIdToIdxMapper::Ids&>())
      .def("__call__", [](const ByteIdToIdxMapper& self,
                          const ByteIdToIdxMapper::Ids& ids) {
        return ValueOrThrow(self(ids));
      });

  m.def("PairMapping", [](const ByteIdToIdxMapper& mapper1,
                          const ByteIdToIdxMapper& mapper2,
                          const ByteIdToIdxMapper::Ids& ids1,
                          const ByteIdToIdxMapper::Ids& ids2,
                          const int num_threads) {
    return ValueOrThrow(PairMapping(mapper1, mapper2, ids1, ids2, num_threads));
  });
}

}  // namespace dgf
