extern crate bindgen;

use std::{env, path::PathBuf};

fn main() {
  let bindings = bindgen::Builder::default()
    .header(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/tvm/include/tvm/runtime/c_runtime_api.h"
    ))
    .header(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/tvm/include/tvm/runtime/c_backend_api.h"
    ))
    .rust_target(bindgen::RustTarget::Nightly)
    .clang_arg(concat!(
      "-I",
      env!("CARGO_MANIFEST_DIR"),
      "/tvm/dlpack/include"
    ))
    .layout_tests(false)
    .generate()
    .expect("Unable to generate bindings.");

  let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
  bindings
    .write_to_file(out_path.join("c_runtime_api.rs"))
    .expect("Unable to write bindings.");
}
