use std::{
  convert::TryFrom,
  mem,
  os::raw::{c_int, c_void},
  ptr,
};

use ndarray;

use super::allocator::Allocation;
use errors::*;
use ffi::runtime::{
  DLContext, DLDataType, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLUInt,
  DLDeviceType_kDLCPU, DLTensor,
};

#[derive(Debug, Clone, Copy)]
pub struct TVMContext {
  pub(super) device_type: usize,
  pub(super) device_id: usize,
}

impl<'a> From<&'a TVMContext> for DLContext {
  fn from(ctx: &'a TVMContext) -> Self {
    Self {
      device_type: ctx.device_type as u32,
      device_id: ctx.device_id as i32,
    }
  }
}

impl Default for TVMContext {
  fn default() -> Self {
    Self {
      device_type: DLDeviceType_kDLCPU as usize,
      device_id: 0,
    }
  }
}

pub enum Storage {
  Owned(Allocation),
  View(*mut u8),
}

impl Storage {
  pub fn new(size: usize, align: Option<usize>) -> Result<Storage> {
    Ok(Storage::Owned(Allocation::new(size, align)?))
  }

  pub fn as_mut_ptr(&self) -> *mut u8 {
    match self {
      Storage::Owned(alloc) => alloc.as_mut_ptr(),
      Storage::View(ptr) => *ptr,
    }
  }

  pub fn view(&self) -> Storage {
    match self {
      Storage::Owned(alloc) => Storage::View(alloc.as_mut_ptr()),
      Storage::View(ptr) => Storage::View(ptr.clone()),
    }
  }

  pub fn offset(&self, offset: isize) -> Storage {
    Storage::View(unsafe { self.as_mut_ptr().offset(offset) })
  }
}

impl<'a> TryFrom<&'a [u8]> for Storage {
  type Error = Error;
  fn try_from(slice: &'a [u8]) -> Result<Self> {
    let storage = Storage::new(slice.len(), None)?;
    unsafe { storage.as_mut_ptr().copy_from(slice.as_ptr(), slice.len()) }
    Ok(storage)
  }
}

pub struct Tensor {
  pub(super) data: Storage,
  pub(super) ctx: TVMContext,
  pub(super) dtype: DataType,
  pub(super) shape: Vec<usize>,
  pub(super) strides: Option<Vec<usize>>,
  pub(super) byte_offset: usize,
}

impl<'a> From<&'a mut Tensor> for DLTensor {
  fn from(tensor: &'a mut Tensor) -> Self {
    DLTensor {
      data: tensor.data.as_mut_ptr() as *mut c_void,
      ctx: DLContext::from(&tensor.ctx),
      ndim: tensor.shape.len() as i32,
      dtype: DLDataType::from(&tensor.dtype),
      shape: tensor.shape.as_ptr() as *mut i64,
      strides: tensor
        .strides
        .as_ref()
        .map(|strides| strides.as_ptr())
        .unwrap_or(ptr::null_mut()) as *mut i64,
      byte_offset: tensor.byte_offset as u64,
    }
  }
}

impl<'a> From<&'a Tensor> for DLTensor {
  fn from(tensor: &'a Tensor) -> Self {
    DLTensor {
      data: tensor.data.as_mut_ptr() as *mut c_void,
      ctx: DLContext::from(&tensor.ctx),
      ndim: tensor.shape.len() as i32,
      dtype: DLDataType::from(&tensor.dtype),
      shape: tensor.shape.as_ptr() as *mut i64,
      strides: tensor
        .strides
        .as_ref()
        .map(|strides| strides.as_ptr())
        .unwrap_or(ptr::null_mut()) as *mut i64,
      byte_offset: tensor.byte_offset as u64,
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DataType {
  pub(super) code: usize,
  pub(super) bits: usize,
  pub(super) lanes: usize,
}

impl<'a> From<&'a DataType> for DLDataType {
  fn from(dtype: &'a DataType) -> Self {
    Self {
      code: dtype.code as u8,
      bits: dtype.bits as u8,
      lanes: dtype.lanes as u16,
    }
  }
}

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
