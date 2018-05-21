#![feature(box_syntax)]
#![feature(try_from)]

#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate lazy_static;
extern crate ndarray;

pub mod ffi {
  #![allow(non_camel_case_types)]
  #![allow(non_snake_case)]
  #![allow(non_upper_case_globals)]
  #![allow(unused)]

  pub mod runtime {
    use std::os::raw::{c_char, c_int, c_void};

    include!(concat!(env!("OUT_DIR"), "/c_runtime_api.rs"));

    pub type BackendPackedCFunc =
      extern "C" fn(args: *const TVMValue, type_codes: *const c_int, num_args: c_int) -> c_int;
  }
}

pub mod errors;
pub mod runtime;

pub use errors::*;
