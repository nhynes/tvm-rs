use std::{any::Any, convert::TryFrom, os::raw::c_void};

use ffi::runtime::{
  BackendPackedCFunc, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLTensor,
  TVMTypeCode_kArrayHandle, TVMTypeCode_kHandle, TVMValue,
};

use errors::*;

macro_rules! call_packed {
  ($fn:expr, $($args:expr),+) => {
    $fn(&[$($args.into(),)+])
  };
}

pub struct TVMArgValue {
  value: TVMValue,
  type_code: i64,
}

macro_rules! impl_prim_tvm_arg {
  ($type:ty, $field:ident, $code:expr, $as:ty) => {
    impl From<$type> for TVMArgValue {
      fn from(val: $type) -> Self {
        TVMArgValue {
          value: TVMValue { $field: val as $as },
          type_code: $code as i64,
        }
      }
    }
  };
  ($type:ty, $field:ident, $code:expr) => {
    impl_prim_tvm_arg!($type, $field, $code, $type);
  };
  ($type:ty,v_int64) => {
    impl_prim_tvm_arg!($type, v_int64, DLDataTypeCode_kDLInt, i64);
  };
  ($type:ty,v_float64) => {
    impl_prim_tvm_arg!($type, v_float64, DLDataTypeCode_kDLFloat, f64);
  };
}

impl_prim_tvm_arg!(f32, v_float64);
impl_prim_tvm_arg!(f64, v_float64);
impl_prim_tvm_arg!(i8, v_int64);
impl_prim_tvm_arg!(u8, v_int64);
impl_prim_tvm_arg!(i32, v_int64);
impl_prim_tvm_arg!(u32, v_int64);
impl_prim_tvm_arg!(i64, v_int64);
impl_prim_tvm_arg!(u64, v_int64);
impl_prim_tvm_arg!(bool, v_int64);

impl<T> From<*const T> for TVMArgValue {
  fn from(ptr: *const T) -> Self {
    TVMArgValue {
      value: TVMValue {
        v_handle: ptr as *mut T as *mut c_void,
      },
      type_code: TVMTypeCode_kArrayHandle as i64,
    }
  }
}

impl<T> From<*mut T> for TVMArgValue {
  fn from(ptr: *mut T) -> Self {
    TVMArgValue {
      value: TVMValue {
        v_handle: ptr as *mut c_void,
      },
      type_code: TVMTypeCode_kHandle as i64,
    }
  }
}

impl<'a> From<&'a mut DLTensor> for TVMArgValue {
  fn from(arr: &'a mut DLTensor) -> Self {
    TVMArgValue {
      value: TVMValue {
        v_handle: arr as *mut _ as *mut c_void,
      },
      type_code: TVMTypeCode_kArrayHandle as i64,
    }
  }
}

pub struct TVMRetValue {
  prim_value: u64,
  box_value: Box<Any>,
  type_code: i64,
}

impl Default for TVMRetValue {
  fn default() -> Self {
    TVMRetValue {
      prim_value: 0,
      box_value: box (),
      type_code: 0,
    }
  }
}

macro_rules! impl_prim_ret_value {
  ($type:ty, $code:expr) => {
    impl From<$type> for TVMRetValue {
      fn from(val: $type) -> Self {
        TVMRetValue {
          prim_value: val as u64,
          box_value: box (),
          type_code: $code,
        }
      }
    }
    impl TryFrom<TVMRetValue> for $type {
      type Error = Error;
      fn try_from(ret: TVMRetValue) -> Result<$type> {
        if ret.type_code == $code {
          Ok(ret.prim_value as $type)
        } else {
          bail!(ErrorKind::TryFromTVMRetValueError(
            stringify!($type).to_string(),
            ret.type_code
          ))
        }
      }
    }
  };
}

macro_rules! impl_boxed_ret_value {
  ($type:ty, $code:expr) => {
    impl From<$type> for TVMRetValue {
      fn from(val: $type) -> Self {
        TVMRetValue {
          prim_value: 0,
          box_value: box val,
          type_code: $code,
        }
      }
    }
    impl TryFrom<TVMRetValue> for $type {
      type Error = Error;
      fn try_from(ret: TVMRetValue) -> Result<$type> {
        if let Ok(val) = ret.box_value.downcast::<$type>() {
          Ok(*val)
        } else {
          bail!(ErrorKind::TryFromTVMRetValueError(
            stringify!($type).to_string(),
            ret.type_code
          ))
        }
      }
    }
  };
}

impl_prim_ret_value!(f64, 8);
impl_prim_ret_value!(i8, 3);
impl_boxed_ret_value!(String, 11);

type PackedFunc = Box<Fn(&[TVMArgValue]) -> TVMRetValue>;
pub fn wrap_backend_packed_func(func: BackendPackedCFunc) -> PackedFunc {
  Box::new(move |args: &[TVMArgValue]| {
    func(
      args
        .iter()
        .map(|ref arg| arg.value)
        .collect::<Vec<TVMValue>>()
        .as_ptr(),
      args
        .iter()
        .map(|ref arg| arg.type_code)
        .collect::<Vec<i64>>()
        .as_ptr() as *const i32,
      args.len() as i32,
    );
    TVMRetValue::default()
  })
}
