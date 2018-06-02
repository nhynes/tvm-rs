use std::{
  collections::HashMap, convert::AsRef, ffi::CStr, os::raw::c_char, string::String, sync::Mutex,
};

use ffi::runtime::BackendPackedCFunc;
use runtime::packed_func::{wrap_backend_packed_func, PackedFunc};

pub trait Module {
  fn get_function<S: AsRef<str>>(&self, name: S) -> Option<PackedFunc>;
}

pub struct SystemLibModule {}

lazy_static! {
  static ref SYSTEM_LIB_FUNCTIONS: Mutex<HashMap<String, BackendPackedCFunc>> =
    Mutex::new(HashMap::new());
}

impl Module for SystemLibModule {
  fn get_function<S: AsRef<str>>(&self, name: S) -> Option<PackedFunc> {
    SYSTEM_LIB_FUNCTIONS
      .lock()
      .unwrap()
      .get(name.as_ref())
      .map(|func| wrap_backend_packed_func(func.to_owned()))
  }
}

impl Default for SystemLibModule {
  fn default() -> Self {
    SystemLibModule {}
  }
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

  // // XXX: remove testing bit
  // if name == "default_function" {
  //   let mut in_nd = Array1::<f32>::zeros(4);
  //   let mut out_nd = Array1::<f32>::zeros(4);
  //   println!("arr: {:?}", out_nd);
  //   let mut in_arr = DLTensor::from(&mut in_nd);
  //   let mut out_arr = DLTensor::from(&mut out_nd);
  //   call_packed!(wrap_backend_packed_func(func), &mut in_arr, &mut out_arr);
  //   // func(
  //   //   [
  //   //   in_arr
  //   //     &mut in_arr as *mut _ as *mut c_void,
  //   //     &mut out_arr as *mut _ as *mut c_void,
  //   //   ].as_ptr() as *const c_void,
  //   //   [7, 7].as_ptr(),
  //   //   2,
  //   // );
  //   println!("that seemed to work?",);
  //   println!("{:?}", Array1::<f32>::from(out_nd).scalar_sum())
  // }
}
