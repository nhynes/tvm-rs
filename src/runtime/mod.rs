mod array;
mod module;
#[macro_use]
mod packed_func;
mod graph;
mod threading;

use std::{ffi::CStr, os::raw::c_char};

use ffi::runtime::{DLContext, DLDeviceType_kDLCPU};

pub use self::{array::*, graph::*, module::*, packed_func::*, threading::*};

#[no_mangle]
pub extern "C" fn TVMAPISetLastError(cmsg: *const c_char) {
  unsafe {
    panic!(CStr::from_ptr(cmsg).to_str().unwrap());
  }
}

impl Default for DLContext {
  fn default() -> Self {
    DLContext {
      device_type: DLDeviceType_kDLCPU,
      device_id: 0,
    }
  }
}
