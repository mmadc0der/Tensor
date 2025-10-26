#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include "api/Api.hpp"

namespace py = pybind11;

static Tensor::DType parse_dtype(const std::string &s) {
    if (s == "float32" || s == "f32") return Tensor::DType::f32;
    if (s == "float64" || s == "f64") return Tensor::DType::f64;
    if (s == "int32" || s == "i32" || s == "int") return Tensor::DType::i32;
    if (s == "int64" || s == "i64") return Tensor::DType::i64;
    throw std::invalid_argument("unsupported dtype: " + s);
}

template <typename T>
static py::array_t<T> dtensor_to_numpy(const Tensor::DTensor &dt) {
    // Only contiguous tensors supported for zero-copy
    if (!Tensor::api::is_contiguous(dt)) {
        throw std::runtime_error("only contiguous tensors are supported for export");
    }
    std::vector<ssize_t> shape(dt.shape().begin(), dt.shape().end());
    std::vector<ssize_t> strides(shape.size());
    ssize_t itemsize = static_cast<ssize_t>(sizeof(T));
    for (size_t i = 0; i < shape.size(); ++i) strides[i] = dt.stride()[i] * itemsize;
    // Keep storage alive by attaching a capsule that owns a shared_ptr copy
    auto storage_sp = dt.storage();
    auto base = py::capsule(new std::shared_ptr<Tensor::Storage>(storage_sp), "tensor_storage_sp",
        [](void* p){ delete static_cast<std::shared_ptr<Tensor::Storage>*>(p); });
    return py::array_t<T>(shape, strides, static_cast<T*>(const_cast<void*>(dt.data())), base);
}

PYBIND11_MODULE(tensor_py, m) {
    m.doc() = "Tensor pybind11 module (experimental)";

    m.def("zeros", [](const std::vector<int64_t>& shape, const std::string &dtype) -> py::array {
        Tensor::DType dt = parse_dtype(dtype);
        Tensor::DTensor t = Tensor::api::zeros(shape, dt);
        switch (dt) {
            case Tensor::DType::f32: return py::array(dtensor_to_numpy<float>(t));
            case Tensor::DType::f64: return py::array(dtensor_to_numpy<double>(t));
            case Tensor::DType::i32: return py::array(dtensor_to_numpy<int32_t>(t));
            case Tensor::DType::i64: return py::array(dtensor_to_numpy<int64_t>(t));
            default: throw std::runtime_error("dtype not supported in zeros");
        }
    }, py::arg("shape"), py::arg("dtype") = "f32");

    m.def("ones", [](const std::vector<int64_t>& shape, const std::string &dtype) -> py::array {
        Tensor::DType dt = parse_dtype(dtype);
        switch (dt) {
            case Tensor::DType::f32: return py::array(dtensor_to_numpy<float>(Tensor::api::ones<float>(shape).as_dtensor()));
            case Tensor::DType::f64: return py::array(dtensor_to_numpy<double>(Tensor::api::ones<double>(shape).as_dtensor()));
            case Tensor::DType::i32: return py::array(dtensor_to_numpy<int32_t>(Tensor::api::ones<int32_t>(shape).as_dtensor()));
            case Tensor::DType::i64: return py::array(dtensor_to_numpy<int64_t>(Tensor::api::ones<int64_t>(shape).as_dtensor()));
            default: throw std::runtime_error("dtype not supported in ones");
        }
    }, py::arg("shape"), py::arg("dtype") = "f32");
}


