use std::{collections::HashMap, convert::TryFrom, iter::FromIterator};

use nom::{le_i32, le_i64, le_u16, le_u32, le_u64, le_u8};
use serde_json;

use super::{DataType, TVMContext, Tensor};
use errors::{Error, ErrorKind, Result};

pub const NDARRAY_MAGIC: u64 = 0xDD5E40F096B4A13F; // Magic number for NDArray file
pub const NDARRAY_LIST_MAGIC: u64 = 0xF7E58D4F05049CB7; // Magic number for NDArray list file

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
    (TVMContext { device_type: device_type, device_id: device_id })
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
  tensor<Tensor>,
  do_parse!(
    take!(8) >> bits!(tag_bits!(u64, 64, 0)) >> ctx: tvm_ctx >> ndim: le_u32 >> dtype: data_type
      >> shape: count!(le_i64, ndim as usize) >> data: length_count!(le_i64, le_u8)
      >> (Tensor {
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
  parse_param_dict<HashMap<String, Tensor>>,
  do_parse!(
    take!(8) >> bits!(tag_bits!(u64, 64, 0)) >> names: length_count!(le_u64, name)
      >> tensors: length_count!(le_u64, tensor)
      >> (HashMap::from_iter(names.into_iter().zip(tensors.into_iter())))
  )
);

pub fn load_param_dict(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
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
