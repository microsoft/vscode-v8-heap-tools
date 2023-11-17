mod decoder;
mod graph;
mod perf;

#[cfg(target_arch = "wasm32")]
mod wasm;

pub use decoder::{decode_reader, decode_slice, decode_str, EdgeType, Node, NodeType};
pub use graph::*;

#[cfg(target_arch = "wasm32")]
pub use wasm::*;
