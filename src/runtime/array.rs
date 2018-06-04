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

pub enum Storage {
  Owned(Allocation),
  View(*mut u8, usize),
}

impl Storage {
  pub fn new(size: usize, align: Option<usize>) -> Result<Storage> {
    Ok(Storage::Owned(Allocation::new(size, align)?))
  }

  pub fn as_mut_ptr(&self) -> *mut u8 {
    match self {
      Storage::Owned(alloc) => alloc.as_mut_ptr(),
      Storage::View(ptr, _size) => *ptr,
    }
  }

  pub fn size(&self) -> usize {
    match self {
      Storage::Owned(alloc) => alloc.size(),
      Storage::View(_ptr, size) => *size,
    }
  }

  pub fn as_ptr(&self) -> *const u8 {
    self.as_mut_ptr() as *const _
  }

  pub fn view(&self) -> Storage {
    match self {
      Storage::Owned(alloc) => Storage::View(alloc.as_mut_ptr(), self.size()),
      Storage::View(ptr, size) => Storage::View(ptr.clone(), *size),
    }
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
  pub(super) byte_offset: isize,
  pub(super) numel: usize,
}

impl Tensor {
  pub fn shape(&self) -> Vec<usize> {
    self.shape.clone()
  }

  pub(super) fn to_vec<T>(&self) -> Vec<T> {
    let mut vec: Vec<T> = Vec::with_capacity(self.numel * self.dtype.itemsize());
    unsafe {
      vec.as_mut_ptr().copy_from_nonoverlapping(
        self.data.as_ptr().offset(self.byte_offset) as *const T,
        self.numel,
      );
      vec.set_len(self.numel);
    }
    vec
  }

  pub fn is_contiguous(&self) -> bool {
    match self.strides {
      None => true,
      Some(ref strides) => {
        self
          .shape
          .iter()
          .zip(strides)
          .rfold(
            (true, 1),
            |(is_contig, expected_stride), (shape, stride)| {
              (
                is_contig && *stride == expected_stride,
                expected_stride * shape,
              )
            },
          )
          .0
      }
    }
  }

  pub fn copy(&mut self, other: &Tensor) {
    assert!(
      self.dtype == other.dtype && self.numel == other.numel,
      "Tensor shape/dtype mismatch."
    );
    assert!(
      self.is_contiguous() && other.is_contiguous(),
      "copy currently requires contiguous tensors\n`self.strides = {:?}` `other.strides = {:?}`",
      self.strides,
      other.strides
    );
    unsafe {
      self
        .data
        .as_mut_ptr()
        .offset(self.byte_offset as isize)
        .copy_from_nonoverlapping(
          other.data.as_mut_ptr(), //.offset(other.byte_offset),
          other.numel * other.dtype.itemsize(),
        );
    }
  }
}

impl<'a> TryFrom<&'a Tensor> for ndarray::ArrayD<f32> {
  type Error = Error;
  fn try_from(tensor: &'a Tensor) -> Result<ndarray::ArrayD<f32>> {
    ensure!(
      tensor.dtype == DTYPE_FLOAT32,
      "Cannot convert Tensor with dtype {:?} to ndarray",
      tensor.dtype
    );
    Ok(ndarray::Array::from_shape_vec(
      tensor.shape.clone(),
      tensor.to_vec::<f32>(),
    )?)
  }
}

impl DLTensor {
  pub(super) fn from_tensor<'a>(tensor: &'a Tensor, flatten: bool) -> Self {
    assert!(!flatten || tensor.is_contiguous());
    Self {
      data: unsafe { tensor.data.as_mut_ptr().offset(tensor.byte_offset) } as *mut c_void,
      ctx: DLContext::from(&tensor.ctx),
      ndim: if flatten { 1 } else { tensor.shape.len() } as i32,
      dtype: DLDataType::from(&tensor.dtype),
      shape: if flatten {
        &tensor.numel
      } else {
        tensor.shape.as_ptr()
      } as *mut i64,
      strides: if flatten || tensor.is_contiguous() {
        ptr::null_mut()
      } else {
        tensor.strides.as_ref().unwrap().as_ptr()
      } as *mut i64,
      byte_offset: 0,
    }
  }
}

impl<'a> From<&'a Tensor> for DLTensor {
  fn from(tensor: &'a Tensor) -> Self {
    DLTensor::from_tensor(tensor, false /* flatten */)
  }
}

impl<'a> From<&'a mut Tensor> for DLTensor {
  fn from(tensor: &'a mut Tensor) -> Self {
    DLTensor::from_tensor(tensor, false /* flatten */)
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataType {
  pub(super) code: usize,
  pub(super) bits: usize,
  pub(super) lanes: usize,
}

impl DataType {
  fn itemsize(&self) -> usize {
    (self.bits * self.lanes) >> 3
  }
}

const DTYPE_FLOAT32: DataType = DataType {
  code: DLDataTypeCode_kDLFloat as usize,
  bits: 32,
  lanes: 1,
};

impl<'a> From<&'a DataType> for DLDataType {
  fn from(dtype: &'a DataType) -> Self {
    Self {
      code: dtype.code as u8,
      bits: dtype.bits as u8,
      lanes: dtype.lanes as u16,
    }
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

macro_rules! impl_tensor_from_ndarray {
  ($type:ty, $typecode:expr) => {
    impl<'a, D: ndarray::Dimension> From<&'a ndarray::Array<$type, D>> for Tensor {
      fn from(arr: &'a ndarray::Array<$type, D>) -> Self {
        let dtype = DataType {
          code: $typecode as usize,
          bits: 8 * mem::size_of::<$type>() as usize,
          lanes: 1,
        };
        let numel = arr
          .shape()
          .into_iter()
          .map(|&v| v as usize)
          .product::<usize>() as usize;
        Self {
          data: Storage::View(arr.as_ptr() as *mut u8, numel * dtype.itemsize()),
          ctx: TVMContext::default(),
          dtype: dtype,
          numel: numel,
          shape: arr.shape().into_iter().map(|&v| v as usize).collect(),
          strides: Some(arr.strides().into_iter().map(|&v| v as usize).collect()),
          byte_offset: 0,
        }
      }
    }
  };
}

macro_rules! impl_dltensor_from_ndarray {
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

impl_dltensor_from_ndarray!(f32, DLDataTypeCode_kDLFloat);
impl_dltensor_from_ndarray!(f64, DLDataTypeCode_kDLFloat);
impl_dltensor_from_ndarray!(i32, DLDataTypeCode_kDLInt);
impl_dltensor_from_ndarray!(i64, DLDataTypeCode_kDLInt);
impl_dltensor_from_ndarray!(u32, DLDataTypeCode_kDLUInt);
impl_dltensor_from_ndarray!(u64, DLDataTypeCode_kDLUInt);

impl_tensor_from_ndarray!(f32, DLDataTypeCode_kDLFloat);
impl_tensor_from_ndarray!(f64, DLDataTypeCode_kDLFloat);
impl_tensor_from_ndarray!(i32, DLDataTypeCode_kDLInt);
impl_tensor_from_ndarray!(i64, DLDataTypeCode_kDLInt);
impl_tensor_from_ndarray!(u32, DLDataTypeCode_kDLUInt);
impl_tensor_from_ndarray!(u64, DLDataTypeCode_kDLUInt);
