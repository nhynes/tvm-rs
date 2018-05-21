use ndarray;
use std::{
  mem,
  os::raw::{c_int, c_void},
};

use ffi::runtime::{
  DLContext, DLDataType, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLUInt,
  DLTensor,
};

macro_rules! impl_from_array {
  ($type:ty, $typecode:expr) => {
    impl<'a, D: ndarray::Dimension> From<&'a mut ndarray::Array<$type, D>> for DLTensor {
      fn from(arr: &'a mut ndarray::Array<$type, D>) -> Self {
        DLTensor {
          data: arr.as_mut_ptr() as *mut c_void,
          ctx: DLContext::default(),
          ndim: arr.ndim() as c_int,
          dtype: DLDataType {
            code: $typecode as u8,
            bits: 8 * mem::size_of::<$type>() as u8,
            lanes: 1,
          },
          shape: arr.shape().as_ptr() as *const i64 as *mut i64,
          strides: arr.strides().as_ptr() as *const isize as *mut i64,
          byte_offset: 0,
        }
      }
    }
  };
}

impl_from_array!(f32, DLDataTypeCode_kDLFloat);
impl_from_array!(f64, DLDataTypeCode_kDLFloat);
impl_from_array!(i32, DLDataTypeCode_kDLInt);
impl_from_array!(i64, DLDataTypeCode_kDLInt);
impl_from_array!(u32, DLDataTypeCode_kDLUInt);
impl_from_array!(u64, DLDataTypeCode_kDLUInt);
