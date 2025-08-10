#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstddef>     // std::byte
#include <type_traits> // std::is_same_v

namespace Tensor {
    // Basic data type enumeration for DL workloads
    enum class DType : uint8_t {
        f16,
        bf16,
        f32,
        f64,
        i32,
        i64
    };

    // Layout tag for fast-paths and heuristics
    enum class Layout : uint8_t {
        contiguous,
        channels_last,
        blocked
    };

    // Device and memory kinds
    enum class DeviceType : uint8_t { CPU, CUDA };
    enum class MemoryKind : uint8_t { host, device, pinned_host };

    struct Device {
        DeviceType type {DeviceType::CPU};
        int32_t id {0}; // for CUDA: GPU ordinal
    };

    // Forward declarations
    class DTensor;
    struct AutogradNode;

    // Storage: shared buffer with deleter and placement metadata
    class Storage {
    public:
        using Deleter = void(*)(void*) ;

        Storage() = default;

        Storage(std::shared_ptr<void> ptr, std::size_t bytes, std::size_t alignment,
                Device device, MemoryKind kind)
            : data_(std::move(ptr)), bytes_(bytes), alignment_(alignment),
              device_(device), kind_(kind) {}

        bool valid() const noexcept { return static_cast<bool>(data_); }
        void* data() noexcept { return data_.get(); }
        const void* data() const noexcept { return data_.get(); }
        std::size_t size_bytes() const noexcept { return bytes_; }
        std::size_t alignment() const noexcept { return alignment_; }
        Device device() const noexcept { return device_; }
        MemoryKind memory_kind() const noexcept { return kind_; }

        // Placeholders for async coordination (to be implemented with CUDA later)
        // e.g., last write/read event handles per stream

    private:
        std::shared_ptr<void> data_{};
        std::size_t bytes_ {0};
        std::size_t alignment_ {64};
        Tensor::Device device_ {};
        MemoryKind kind_ {MemoryKind::host};
    };

    // Small helpers
    inline constexpr std::size_t dtype_size(DType dt) noexcept {
        switch (dt) {
            case DType::f16:  return 2;
            case DType::bf16: return 2;
            case DType::f32:  return 4;
            case DType::f64:  return 8;
            case DType::i32:  return 4;
            case DType::i64:  return 8;
        }
        return 0;
    }

    // Runtime-typed tensor object
    class DTensor {
    public:
        DTensor() = default;

        // Construct a view over existing storage and metadata
        DTensor(std::shared_ptr<Storage> storage,
                std::vector<int64_t> shape,
                std::vector<int64_t> stride,
                int64_t offset,
                DType dtype,
                Layout layout = Layout::contiguous,
                bool is_contiguous = false,
                bool requires_grad = false)
            : storage_(std::move(storage)),
              shape_(std::move(shape)),
              stride_(std::move(stride)),
              offset_(offset),
              dtype_(dtype),
              layout_(layout),
              is_contiguous_(is_contiguous),
              requires_grad_(requires_grad) {}

        // Basic metadata
        const std::vector<int64_t>& shape() const noexcept { return shape_; }
        const std::vector<int64_t>& stride() const noexcept { return stride_; }
        int64_t offset() const noexcept { return offset_; }
        DType dtype() const noexcept { return dtype_; }
        Layout layout() const noexcept { return layout_; }
        bool is_contiguous() const noexcept { return is_contiguous_; }
        bool requires_grad() const noexcept { return requires_grad_; }

        void set_requires_grad(bool v) noexcept { requires_grad_ = v; }
        void set_contiguous(bool v) noexcept { is_contiguous_ = v; }

        int32_t rank() const noexcept { return static_cast<int32_t>(shape_.size()); }

        int64_t numel() const noexcept {
            int64_t n = 1;
            for (auto d : shape_) n *= d;
            return n;
        }

        // Raw data access (byte-wise)
        void* data() noexcept {
            auto* base = static_cast<std::byte*>(storage_->data());
            return static_cast<void*>(base + offset_ * static_cast<int64_t>(dtype_size(dtype_)));
        }
        const void* data() const noexcept {
            auto* base = static_cast<const std::byte*>(storage_->data());
            return static_cast<const void*>(base + offset_ * static_cast<int64_t>(dtype_size(dtype_)));
        }

        std::shared_ptr<Storage> storage() const noexcept { return storage_; }

        // Autograd hooks (minimal scaffolding)
        std::shared_ptr<DTensor> grad() const noexcept { return grad_; }
        void set_grad(std::shared_ptr<DTensor> g) noexcept { grad_ = std::move(g); }

        std::weak_ptr<AutogradNode> grad_fn() const noexcept { return grad_fn_; }
        void set_grad_fn(std::weak_ptr<AutogradNode> fn) noexcept { grad_fn_ = std::move(fn); }

    private:
        std::shared_ptr<Storage> storage_{};
        std::vector<int64_t> shape_{};
        std::vector<int64_t> stride_{};
        int64_t offset_ {0};
        DType dtype_ {DType::f32};
        Layout layout_ {Layout::contiguous};
        bool is_contiguous_ {false};
        bool requires_grad_ {false};

        // Autograd state
        std::shared_ptr<DTensor> grad_{};             // accumulated gradient (same shape)
        std::weak_ptr<AutogradNode> grad_fn_{};       // node that produced this tensor
    };

    // Autograd node interface (skeleton)
    struct AutogradNode : std::enable_shared_from_this<AutogradNode> {
        // Saved tensors for backward (views to avoid copies)
        std::vector<DTensor> saved_tensors{};

        // Apply backward given upstream gradient(s). To be implemented later.
        virtual void backward(const DTensor& upstream) = 0;
        virtual ~AutogradNode() = default;
    };

    // Factory for host storage (implemented in Tensor.cpp)
    std::shared_ptr<Storage> make_host_storage(std::size_t bytes, std::size_t alignment = 64);

    // Compute default contiguous strides for a given shape
    inline std::vector<int64_t> default_strides(const std::vector<int64_t>& shape) {
        std::vector<int64_t> s(shape.size());
        int64_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            s[i] = stride;
            stride *= shape[i];
        }
        return s;
    }

    // Typed facade over DTensor with compile-time T
    template <typename T>
    class Tensor {
    public:
        Tensor() {
            // Create a 1-element host tensor to satisfy immediate usability
            constexpr DType dt = dtype_of();
            auto st = make_host_storage(sizeof(T), 64);
            std::vector<int64_t> shp{1};
            auto str = default_strides(shp);
            dt_ = DTensor(
                std::move(st),
                std::move(shp),
                std::move(str),
                /*offset*/0,
                dt,
                Layout::contiguous,
                /*is_contiguous*/true,
                /*requires_grad*/false
            );
        }
        explicit Tensor(DTensor dt) : dt_(std::move(dt)) { assert(matches_dtype()); }

        // Accessors
        const std::vector<int64_t>& shape() const noexcept { return dt_.shape(); }
        const std::vector<int64_t>& stride() const noexcept { return dt_.stride(); }
        int64_t offset() const noexcept { return dt_.offset(); }
        int32_t rank() const noexcept { return dt_.rank(); }
        int64_t numel() const noexcept { return dt_.numel(); }
        bool is_contiguous() const noexcept { return dt_.is_contiguous(); }

        T* data() noexcept { return static_cast<T*>(dt_.data()); }
        const T* data() const noexcept { return static_cast<const T*>(dt_.data()); }

        DTensor& as_dtensor() & noexcept { return dt_; }
        const DTensor& as_dtensor() const & noexcept { return dt_; }
        DTensor as_dtensor() && noexcept { return std::move(dt_); }

        // Autograd convenience
        bool requires_grad() const noexcept { return dt_.requires_grad(); }
        void set_requires_grad(bool v) noexcept { dt_.set_requires_grad(v); }

        std::shared_ptr<DTensor> grad() const noexcept { return dt_.grad(); }
        void set_grad(std::shared_ptr<DTensor> g) noexcept { dt_.set_grad(std::move(g)); }

    private:
        static constexpr DType dtype_of() noexcept {
            if constexpr (std::is_same_v<T, int>) return DType::i32;
            if constexpr (std::is_same_v<T, float>) return DType::f32;
            if constexpr (std::is_same_v<T, double>) return DType::f64;
            if constexpr (std::is_same_v<T, std::int32_t>) return DType::i32;
            if constexpr (std::is_same_v<T, std::int64_t>) return DType::i64;
            // fp16/bf16 will be provided via custom scalar types
            return DType::f32; // default fallback
        }
        bool matches_dtype() const noexcept { return dt_.dtype() == dtype_of(); }

        DTensor dt_{};
    };
}
