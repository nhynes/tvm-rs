#![feature(
  alloc,
  allocator_api,
  box_syntax,
  extern_prelude,
  fn_traits,
  try_from,
  unboxed_closures,
  vec_remove_item
)]

#[cfg(target_env = "sgx")]
extern crate alloc;
extern crate bounded_spsc_queue;
#[cfg(target_env = "sgx")]
extern crate core;
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate ndarray;
#[macro_use]
extern crate nom;
#[cfg(not(target_env = "sgx"))]
extern crate num_cpus;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

pub mod ffi {
  #![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused
  )]

  pub mod runtime {
    use std::os::raw::{c_char, c_int, c_void};

    include!(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/src/runtime/c_runtime_api.rs"
    ));

    pub type BackendPackedCFunc =
      extern "C" fn(args: *const TVMValue, type_codes: *const c_int, num_args: c_int) -> c_int;
  }
}

pub mod errors;
pub mod runtime;

pub use errors::*;
