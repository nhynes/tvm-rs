# TVM Rust Runtime

[docs](https://docs.rs/tvm/0.1.0/tvm/)

This crate provides a static TVM  runtime which is compatible with the original C++ implementation.
In particular, it supports

* NNVM graphs (training and inference)
* threading
* convertions to and from [`ndarray`](https://github.com/bluss/ndarray)
* **Web Assembly**
* **SGX**
* integration with the TVM Python module system
