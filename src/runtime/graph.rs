use std::{
  cmp,
  collections::HashMap,
  convert::TryFrom,
  heap::{Alloc, Heap, Layout},
  iter::FromIterator,
  str,
};

use nom::{alpha1, digit1, le_i32, le_i64, le_u16, le_u32, le_u64, le_u8, types::CompleteStr};
use serde_json;

use super::{DataType, Module, Storage, TVMArgValue, TVMContext, Tensor};
use errors::{Error, ErrorKind, Result};
use ffi::runtime::{
  DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLUInt, DLTensor,
};

const NDARRAY_MAGIC: u64 = 0xDD5E40F096B4A13F; // Magic number for NDArray file
const NDARRAY_LIST_MAGIC: u64 = 0xF7E58D4F05049CB7; // Magic number for NDArray list file

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

impl Graph {
  fn entry_index(&self, entry: &Entry) -> Result<usize> {
    self
      .node_row_ptr
      .as_ref()
      .map(|nrp| nrp[entry.id] + entry.index)
      .ok_or("Missing node_row_ptr.".into())
  }
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
  tensors: Vec<Tensor>,
}

impl<'m> GraphExecutor<'m> {
  pub fn new<M: 'm + Module>(graph: &Graph, lib: &'m M) -> Result<Self> {
    let tensors = Self::setup_storages(graph)?;
    Ok(GraphExecutor {
      op_execs: Self::setup_op_execs(graph, lib, &tensors)?,
      tensors: tensors,
    })
  }

  pub fn run(&self) {
    self.op_execs.iter().for_each(|op_exec| {
      op_exec();
    });
  }

  fn setup_storages(graph: &Graph) -> Result<Vec<Tensor>> {
    let graph_attrs = graph.attrs.as_ref().ok_or(ErrorKind::GraphFormatError(
      "Missing graph attrs".to_string(),
    ))?;

    let storage_ids = serde_json::from_value::<(String, Vec<usize>)>(
      graph_attrs
        .get("storage_id")
        .ok_or(ErrorKind::GraphFormatError(
          "Missing storage_id attr".to_string(),
        ))?
        .to_owned(),
    )?.1;

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
        if let Ok((_, dtype)) = tvm_str_to_type(CompleteStr(dltype)) {
          Ok(dtype)
        } else {
          Err(ErrorKind::GraphFormatError(format!("Invalid dltype: {}", dltype).to_string()).into())
        }
      })
      .collect::<Result<Vec<DataType>>>()?;

    let align = dtypes.iter().map(|dtype| dtype.bits as usize >> 3).max();
    let mut storage_num_bytes = vec![0usize; *storage_ids.iter().max().unwrap_or(&1) + 1];
    for (i, &storage_id) in storage_ids.iter().enumerate() {
      let dtype_size = dtypes[i].bits * dtypes[i].lanes >> 3;
      let nbytes = dtype_size * shapes[i].iter().product::<usize>();
      storage_num_bytes[storage_id] = cmp::max(nbytes, storage_num_bytes[storage_id]);
    }

    let storage = Storage::new(storage_num_bytes.iter().sum(), align)?;
    let mut offsets = vec![0; storage_ids.len()]; //Vec::with_capacity(storage_ids.len());
    offsets[0] = 0;
    for i in 0..(offsets.len() - 2) {
      offsets[i + 1] = offsets[i] + storage_num_bytes[i];
    }

    let tensors = izip!(storage_ids, shapes, dtypes)
      .map(|(storage_id, shape, dtype)| Tensor {
        data: storage.clone(),
        ctx: TVMContext::default(),
        dtype: dtype,
        shape: shape,
        strides: None,
        byte_offset: offsets[storage_id],
      })
      .collect();

    Ok(tensors)
  }

  fn setup_op_execs<M: 'm + Module>(
    graph: &Graph,
    lib: &'m M,
    tensors: &Vec<Tensor>,
  ) -> Result<Vec<Box<Fn() + 'm>>> {
    ensure!(graph.node_row_ptr.is_some(), "Missing node_row_ptr.");
    let node_row_ptr = graph.node_row_ptr.as_ref().unwrap();
    graph
      .nodes
      .iter()
      .filter(|node| node.op != "null")
      .map(|node| {
        ensure!(node.op == "tvm_op", "Only TVM ops are supported.");
        ensure!(node.attrs.is_some(), "Missing node_row_ptr.");
        let node_attrs = node.attrs.as_ref().unwrap();
        let func_name = node_attrs.get("func_name").unwrap();
        let func = lib
          .get_function(func_name)
          .ok_or(format!("Missing function {}", func_name))?;
        let num_outputs = node_attrs
          .get("num_outputs")
          .unwrap()
          .parse::<usize>()
          .unwrap();
        let arg_indices = node
          .inputs
          .iter()
          .map(|entry| graph.entry_index(entry))
          .chain((0..num_outputs).map(|oi| Ok(node_row_ptr[oi].clone())));
        let args = arg_indices
          .map(|idx| Ok(TVMArgValue::from(&mut DLTensor::from(&tensors[idx?]))))
          .collect::<Result<Vec<TVMArgValue>>>()?;
        let op: Box<Fn()> = box move || {
          func(args.as_slice());
          // func.call_box((args.as_slice(),));
        };
        Ok(op)
      })
      .collect()
  }

  fn load_params(&self, params: HashMap<String, Tensor>) {
    // TODO
  }
}

named!(
  tvm_str_to_type<CompleteStr, DataType>,
  do_parse!(
    type_name: alpha1 >>
    bits: digit1 >>
    lanes: opt!(tuple!(tag!("x"), digit1)) >>
    (DataType {
      code: match type_name {
        CompleteStr("int") => DLDataTypeCode_kDLInt,
        CompleteStr("uint") => DLDataTypeCode_kDLUInt,
        CompleteStr("float") => DLDataTypeCode_kDLFloat,
        _ => DLDataTypeCode_kDLFloat,
      } as usize,
      bits: bits.parse::<u8>().unwrap() as usize,
      lanes: match lanes {
        Some(lanes) => lanes.1.parse::<u16>().unwrap() as usize,
        None => 1,
      },
    })
  )
);

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
    (DataType { code: code as usize, bits: bits as usize, lanes: lanes as usize })
  )
);

named!(
  tensor<Tensor>,
  do_parse!(
    take!(8) >> bits!(tag_bits!(u64, 64, 0)) >> ctx: tvm_ctx >> ndim: le_u32 >> dtype: data_type
      >> shape: count!(map!(le_i64, |sz| sz as usize), ndim as usize) >> length: le_i64
      >> data: take!(length) >> (Tensor {
      data: Storage::try_from(data).unwrap(),
      ctx: ctx,
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_str_to_type() {
    assert_eq!(
      tvm_str_to_type(CompleteStr("float24")).unwrap().1,
      DataType {
        code: DLDataTypeCode_kDLFloat as usize,
        bits: 24,
        lanes: 1
      }
    );
    assert_eq!(
      tvm_str_to_type(CompleteStr("uint111x44")).unwrap().1,
      DataType {
        code: DLDataTypeCode_kDLUInt as usize,
        bits: 111,
        lanes: 44
      }
    );
  }
}
