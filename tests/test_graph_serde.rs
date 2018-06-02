#![feature(fs_read_write, try_from)]

extern crate serde;
extern crate serde_json;

extern crate tvm;

use std::{collections::HashMap, convert::TryFrom, fs, io::Read};

use tvm::{
  errors::*,
  runtime::{Graph, Tensor},
};

fn load_graph() -> Result<Graph> {
  Graph::try_from(
    &fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/graph.json")).unwrap(),
  )
}

fn load_params() -> Result<HashMap<String, Tensor>> {
  let mut params_bytes = Vec::new();
  fs::File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/graph.params"))
    .unwrap()
    .read_to_end(&mut params_bytes)
    .unwrap();
  tvm::runtime::load_param_dict(&params_bytes)
}

#[test]
fn test_load_graph() {
  let graph = load_graph().unwrap();

  assert_eq!(graph.nodes[3].op, "tvm_op");
  assert_eq!(
    graph.nodes[3]
      .attrs
      .as_ref()
      .unwrap()
      .get("func_name")
      .unwrap(),
    "fuse_dense"
  );
  assert_eq!(graph.nodes[5].inputs[0].index, 0);
  assert_eq!(graph.nodes[6].inputs[0].index, 1);
  assert_eq!(graph.heads.len(), 2);
}

#[test]
fn test_load_params() {
  assert!(load_params().is_ok());
}
