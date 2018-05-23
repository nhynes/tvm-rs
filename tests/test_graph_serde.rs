#![feature(fs_read_write)]

extern crate serde;
#[macro_use]
extern crate serde_json;

extern crate tvm;

use std::fs;

#[test]
fn test_resnet_inference() {
  let graph: tvm::runtime::Graph = serde_json::from_str(&fs::read_string(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/resnet_inference.json"
  )).unwrap())
    .unwrap();

  assert_eq!(graph.nodes[0].name, "data");
  assert_eq!(
    graph.nodes[1]
      .attrs
      .as_ref()
      .unwrap()
      .get("epsilon")
      .cloned()
      .unwrap(),
    "2e-05"
  );
  assert_eq!(graph.nodes[5].inputs[3].version, 1);
  assert_eq!(
    json!(graph.attrs.as_ref().unwrap().get("shape").unwrap())[1][0][3],
    224
  );
}
