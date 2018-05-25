use std::{
  collections::HashMap,
  convert::TryFrom,
  heap::{Alloc, Heap, Layout},
  iter::FromIterator,
  slice,
  str,
};

use nom::{alpha, digit, le_i32, le_i64, le_u16, le_u32, le_u64, le_u8};
use serde_json;

use super::{DataType, Module, OwnedTensor, TVMContext, Tensor, ViewTensor};
use errors::{self, Error, ErrorKind, Result};
use ffi::runtime::{
  DLContext, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLUInt, DLTensor,
};

const NDARRAY_MAGIC: u64 = 0xDD5E40F096B4A13F; // Magic number for NDArray file
const NDARRAY_LIST_MAGIC: u64 = 0xF7E58D4F05049CB7; // Magic number for NDArray list file
const DEFAULT_ALIGN_BYTES: usize = 4;

#[derive(Serialize, Deserialize, Debug)]
pub struct Graph {
  pub nodes: Vec<Node>,
  pub arg_nodes: Vec<usize>,
  pub heads: Vec<Entry>,
  pub node_row_ptr: Option<Vec<usize>>,
  pub attrs: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Entry {
  pub id: usize,
  pub index: usize,
  pub version: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Node {
  pub op: String,
  pub name: String,
  pub inputs: Vec<Entry>,
  pub attrs: Option<HashMap<String, String>>,
  pub control_deps: Option<Vec<Entry>>,
}

impl<'a> TryFrom<&'a String> for Graph {
  type Error = Error;
  fn try_from(graph_json: &String) -> Result<Self> {
    let graph = serde_json::from_str(graph_json)?;
    Ok(graph)
  }
}

pub struct GraphExecutor<'m> {
  op_execs: Vec<Box<Fn() + 'm>>,
  tensors: Tensors,
}

impl<'m> GraphExecutor<'m> {
  pub fn new<M: 'm + Module>(graph: &Graph, lib: &'m M) -> Result<Self> {
    let tensors = Self::setup_storages(graph)?;
    Ok(GraphExecutor {
      op_execs: Self::setup_op_execs(graph, lib, &tensors),
      tensors: tensors,
    })
  }

  fn setup_storages(graph: &Graph) -> Result<Tensors> {
    let graph_attrs = graph.attrs.as_ref().ok_or(ErrorKind::GraphFormatError(
      "Missing graph attrs".to_string(),
    ))?;

    let shapes = serde_json::from_value::<(String, Vec<Vec<usize>>)>(
      graph_attrs
        .get("shape")
        .ok_or(ErrorKind::GraphFormatError(
          "Missing shape attr".to_string(),
        ))?
        .to_owned(),
    )?.1;

    let dtypes = serde_json::from_value::<(String, Vec<String>)>(
      graph_attrs
        .get("dltype")
        .ok_or(ErrorKind::GraphFormatError(
          "Missing dltype attr".to_string(),
        ))?
        .to_owned(),
    )?.1
      .iter()
      .map(|dltype| {
        if let Ok((_, dtype)) = tvm_str_to_type(dltype) {
          Ok(dtype)
        } else {
          Err(ErrorKind::GraphFormatError(format!("Invalid dltype: {}", dltype).to_string()).into())
        }
      })
      .collect::<Result<Vec<DataType>>>()?;

    Tensors::new(shapes, dtypes)
  }

  fn setup_op_execs<M: 'm + Module>(
    graph: &Graph,
    lib: &'m M,
    tensors: &Tensors,
  ) -> Vec<Box<Fn() + 'm>> {
    let t = tensors.get(4);
    let mut op_execs = Vec::new();
    let func: Box<Fn()> = box move || {
      lib.get_function("asdf".to_string()).unwrap();
      return ();
    };
    op_execs.push(func);
    op_execs
  }

  fn load_params(&self, params: HashMap<String, OwnedTensor>) {
    // TODO
  }
}

named!(
  tvm_str_to_type<&str, DataType>,
  do_parse!(
    type_name: alpha >>
    bits: digit >>
    lanes: opt!(tuple!(tag!("x"), digit)) >>
    (DataType {
      code: match type_name {
        "int" => DLDataTypeCode_kDLInt,
        "uint" => DLDataTypeCode_kDLUInt,
        "float" => DLDataTypeCode_kDLFloat,
        _ => DLDataTypeCode_kDLFloat,
      } as u8,
      bits: bits.parse::<u8>().unwrap(),
      lanes: match lanes {
        Some(lanes) => lanes.1.parse::<u16>().unwrap(),
        None => 1,
      },
    })
  )
);

struct Tensors {
  arena_base: *mut u8,
  arena_layout: Layout,
  shapes: Vec<Vec<usize>>,
  dtypes: Vec<DataType>,
  ptrs: Vec<*mut u8>,
  num_bytes: Vec<usize>,
}

impl Tensors {
  pub fn new(shapes: Vec<Vec<usize>>, dtypes: Vec<DataType>) -> Result<Self> {
    ensure!(shapes.len() == dtypes.len(), "len(shapes) != len(dtypes)");

    let num_bytes: Vec<usize> = dtypes
      .iter()
      .zip(shapes.iter())
      .map(|(dtype, shape)| {
        let dtype_bytes = (dtype.bits as usize) * (dtype.lanes as usize) / 8;
        let arr_size: usize = shape.iter().product();
        dtype_bytes * arr_size
      })
      .collect();

    let align = dtypes
      .iter()
      .map(|dtype| dtype.bits as usize >> 3)
      .max()
      .unwrap_or(DEFAULT_ALIGN_BYTES);

    let layout = Layout::from_size_align(num_bytes.iter().sum(), align).unwrap();
    let ptr = unsafe { Heap::default().alloc(layout.clone())? };

    let mut head = ptr.clone();
    let mut ptrs = Vec::with_capacity(num_bytes.len());
    for i in 0..num_bytes.len() {
      ptrs.push(head);
      head = unsafe { head.add(num_bytes[i]) };
    }

    Ok(Tensors {
      arena_base: ptr,
      arena_layout: layout,
      shapes: shapes,
      dtypes: dtypes,
      ptrs: ptrs,
      num_bytes: num_bytes,
    })
  }

  pub fn get(&self, index: usize) -> ViewTensor {
    ViewTensor {
      data: unsafe { slice::from_raw_parts_mut(self.ptrs[index], self.num_bytes[index]) },
      ctx: TVMContext::default(),
      ndim: self.shapes[index].len(),
      dtype: self.dtypes[index].clone(),
      shape: self.shapes[index].clone(),
      strides: None,
      byte_offset: 0,
    }
  }
}

impl Drop for Tensors {
  fn drop(&mut self) {
    unsafe {
      Heap::default().dealloc(self.arena_base, self.arena_layout.clone());
    }
  }
}

named!(
  name<String>,
  map_res!(length_bytes!(le_u64), |b: &[u8]| {
    String::from_utf8(b.to_vec())
  })
);

named!(
  tvm_ctx<&[u8], TVMContext>,
  do_parse!(
    device_type: le_u32 >>
    device_id: le_i32 >>
    (TVMContext { device_type: device_type as usize, device_id: device_id as usize })
  )
);

named!(
  data_type<&[u8], DataType>,
  do_parse!(
    code: le_u8 >>
    bits: le_u8 >>
    lanes: le_u16 >>
    (DataType { code: code, bits: bits, lanes: lanes })
  )
);

named!(
  tensor<OwnedTensor>,
  do_parse!(
    take!(8) >> bits!(tag_bits!(u64, 64, 0)) >> ctx: tvm_ctx >> ndim: le_u32 >> dtype: data_type
      >> shape: count!(map!(le_i64, |sz| sz as usize), ndim as usize)
      >> data: length_count!(le_i64, le_u8) >> (Tensor {
      data: data,
      ctx: ctx,
      ndim: ndim as usize,
      dtype: dtype,
      shape: shape,
      strides: None,
      byte_offset: 0,
    })
  )
);

named!(
  parse_param_dict<HashMap<String, OwnedTensor>>,
  do_parse!(
    take!(8) >> bits!(tag_bits!(u64, 64, 0)) >> names: length_count!(le_u64, name)
      >> tensors: length_count!(le_u64, tensor)
      >> (HashMap::from_iter(names.into_iter().zip(tensors.into_iter())))
  )
);

pub fn load_param_dict(bytes: &[u8]) -> Result<HashMap<String, OwnedTensor>> {
  if let Ok((remaining_bytes, param_dict)) = parse_param_dict(bytes) {
    if remaining_bytes.len() > 0 {
      bail!(ErrorKind::LoadGraphParamsError("extra input".to_string()))
    } else {
      Ok(param_dict)
    }
  } else {
    bail!(ErrorKind::LoadGraphParamsError(
      "invalid parameters file".to_string()
    ))
  }
}
