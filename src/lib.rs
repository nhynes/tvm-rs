mod ffi {
  #![allow(non_camel_case_types)]
  #![allow(non_snake_case)]
  #![allow(non_upper_case_globals)]
  #![allow(unused)]

  mod runtime {
    include!(concat!(env!("OUT_DIR"), "/c_runtime_api.rs"));
  }
}

pub mod runtime;
