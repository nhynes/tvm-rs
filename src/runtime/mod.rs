#[macro_use]
mod packed_func;
mod array;

use ndarray::Array1;

use std::{
  collections::HashMap,
  ffi::CStr,
  os::raw::{c_char, c_void},
  ptr,
  string::String,
  sync::Mutex,
};

use ffi::runtime::{
  BackendPackedCFunc, DLContext, DLDeviceType_kDLCPU, DLTensor, TVMParallelGroupEnv,
};

pub use self::{array::*, packed_func::*};

lazy_static! {
  static ref SYSTEM_LIB_FUNCTIONS: Mutex<HashMap<String, BackendPackedCFunc>> =
    Mutex::new(HashMap::new());
}

#[no_mangle]
pub extern "C" fn TVMAPISetLastError(cmsg: *const c_char) {
  unsafe {
    panic!(CStr::from_ptr(cmsg).to_str().unwrap());
  }
}

#[no_mangle]
pub extern "C" fn TVMBackendParallelLaunch(
  cb: extern "C" fn(task_id: i32, penv: *const TVMParallelGroupEnv, cdata: *const c_void) -> i32,
  cdata: *const c_void,
  num_task: i32,
) {
  // TODO: threadpool
  let penv = TVMParallelGroupEnv {
    sync_handle: ptr::null_mut() as *mut c_void,
    num_task: 1,
  };
  cb(0, &penv as *const _, cdata);
}

#[no_mangle]
pub extern "C" fn TVMBackendRegisterSystemLibSymbol(
  cname: *const c_char,
  func: BackendPackedCFunc,
) {
  let name = unsafe { CStr::from_ptr(cname).to_str().unwrap() };
  SYSTEM_LIB_FUNCTIONS
    .lock()
    .unwrap()
    .insert(name.to_string(), func);

  // XXX: remove testing bit
  if name == "default_function" {
    let mut in_nd = Array1::<f32>::zeros(4);
    let mut out_nd = Array1::<f32>::zeros(4);
    println!("arr: {:?}", out_nd);
    let mut in_arr = DLTensor::from(&mut in_nd);
    let mut out_arr = DLTensor::from(&mut out_nd);
    call_packed!(wrap_backend_packed_func(func), &mut in_arr, &mut out_arr);
    // func(
    //   [
    //   in_arr
    //     &mut in_arr as *mut _ as *mut c_void,
    //     &mut out_arr as *mut _ as *mut c_void,
    //   ].as_ptr() as *const c_void,
    //   [7, 7].as_ptr(),
    //   2,
    // );
    println!("that seemed to work?",);
    println!("{:?}", Array1::<f32>::from(out_nd).scalar_sum())
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
