use std::{
  mem,
  os::raw::{c_int, c_void},
};

use ndarray;

use ffi::runtime::{
  DLContext, DLDataType, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLUInt,
  DLDeviceType_kDLCPU, DLTensor,
};

#[derive(Debug)]
pub struct TVMContext {
  pub device_type: usize,
  pub device_id: usize,
}

impl Default for TVMContext {
  fn default() -> Self {
    Self {
      device_type: DLDeviceType_kDLCPU as usize,
      device_id: 0,
    }
  }
}

#[derive(Debug)]
pub struct Tensor<T> {
  pub(super) data: T,
  pub(super) ctx: TVMContext,
  pub(super) ndim: usize,
  pub(super) dtype: DLDataType,
  pub(super) shape: Vec<usize>,
  pub(super) strides: Option<Vec<usize>>,
  pub(super) byte_offset: usize,
}
pub type OwnedTensor = Tensor<Vec<u8>>;
pub type ViewTensor<'a> = Tensor<&'a [u8]>;

pub type DataType = DLDataType;

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

impl Default for DLContext {
  fn default() -> Self {
    DLContext {
      device_type: DLDeviceType_kDLCPU,
      device_id: 0,
    }
  }
}

impl_from_array!(f32, DLDataTypeCode_kDLFloat);
impl_from_array!(f64, DLDataTypeCode_kDLFloat);
impl_from_array!(i32, DLDataTypeCode_kDLInt);
impl_from_array!(i64, DLDataTypeCode_kDLInt);
impl_from_array!(u32, DLDataTypeCode_kDLUInt);
impl_from_array!(u64, DLDataTypeCode_kDLUInt);
