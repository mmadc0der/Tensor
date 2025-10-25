#include "Tensor.hpp"
#include <cstdlib>
#include <cstring>
#include <new>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace Tensor {

static void *aligned_alloc_host(std::size_t alignment, std::size_t size) {
#if defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  // posix_memalign alternative for portability
  void *p = nullptr;
  if (alignment < sizeof(void *))
    alignment = sizeof(void *);
  // try posix_memalign if available
  if (posix_memalign(&p, alignment, size) == 0)
    return p;
  return nullptr;
#endif
}

std::shared_ptr<Storage> make_host_storage(std::size_t bytes,
                                           std::size_t alignment) {
  void *raw = aligned_alloc_host(alignment, bytes);
  if (!raw) {
    throw std::bad_alloc{};
  }
  // wrap in shared_ptr with custom deleter
#if defined(_WIN32)
  auto deleter = [](void *p) { _aligned_free(p); };
#else
  auto deleter = [](void *p) { std::free(p); };
#endif
  std::shared_ptr<void> ptr(raw, deleter);
  auto st =
      std::make_shared<Storage>(std::move(ptr), bytes, alignment,
                                Device{DeviceType::CPU, 0}, MemoryKind::host);
  return st;
}

} // namespace Tensor
