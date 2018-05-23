use std::{collections::HashMap, convert::TryFrom};

use serde_json;

use errors::{Error, Result};

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
    let graph = serde_json::from_str(graph_json.borrow())?;
    Ok(graph)
  }
}
