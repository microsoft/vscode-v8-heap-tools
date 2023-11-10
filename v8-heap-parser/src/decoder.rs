use ouroboros::self_referencing;
use std::{collections::HashMap, fmt::Display};
use typed_arena::Arena;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::{
    error::{Error, Result},
    graph::Graph,
};

use self::raw::StringOrArray;

mod raw {
    use serde::Deserialize;

    #[derive(Deserialize)]
    pub struct Root {
        pub snapshot: Snapshot,
        pub nodes: Vec<u64>,
        pub edges: Vec<u64>,
        pub trace_function_infos: Vec<u32>,
        pub trace_tree: Vec<u32>,
        pub samples: Vec<u32>,
        pub locations: Vec<u32>,
        pub strings: Vec<String>,
    }

    #[derive(Deserialize)]
    pub struct Snapshot {
        pub meta: Meta,
        pub node_count: u64,
        pub edge_count: u64,
        pub trace_function_count: u64,
        pub root_index: Option<usize>,
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    pub enum StringOrArray {
        Single(String),
        Arr(Vec<String>),
    }

    #[derive(Deserialize)]
    pub struct Meta {
        pub node_fields: Vec<String>,
        pub node_types: Vec<StringOrArray>,
        pub edge_fields: Vec<String>,
        pub edge_types: Vec<StringOrArray>,
        pub trace_function_info_fields: Vec<String>,
        pub trace_node_fields: Vec<String>,
        pub sample_fields: Vec<String>,
    }
}

pub(crate) type PetGraph<'a> = petgraph::Graph<Node<'a>, PGNodeEdge<'a>>;

#[self_referencing]
pub(crate) struct GraphInner {
    arena: Arena<String>,
    #[borrows(arena)]
    #[covariant]
    inner_nodes: Result<PetGraph<'this>>,
}

impl GraphInner {
    pub(crate) fn borrow(&self) -> &PetGraph<'_> {
        self.borrow_inner_nodes().as_ref().unwrap()
    }
}

#[derive(Debug)]
pub struct Node<'a> {
    pub typ: NodeType<'a>,
    pub name: &'a str,
    pub id: u32,
    pub self_size: u64,
    pub edge_count: usize,
    pub trace_node_id: u64,
    pub detachedness: u32,
}

impl<'a> Node<'a> {
    pub(crate) fn is_document_dom_trees_root(&self) -> bool {
        self.typ == NodeType::Syntheic && self.name == "(Document DOM trees)'"
    }

    pub fn class_name(&self) -> &'a str {
        match &self.typ {
            NodeType::Object | NodeType::Native => self.name,
            t => t.as_class_name(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[non_exhaustive]
pub enum NodeType<'a> {
    Hidden,
    Array,
    String,
    Object,
    Code,
    Closure,
    RegExp,
    Number,
    Native,
    Syntheic,
    ConcatString,
    SliceString,
    BigInt,
    Other(&'a str),
}

impl<'a> From<NodeType<'a>> for u8 {
    fn from(t: NodeType<'a>) -> u8 {
        match t {
            NodeType::Hidden => 0,
            NodeType::Array => 1,
            NodeType::String => 2,
            NodeType::Object => 3,
            NodeType::Code => 4,
            NodeType::Closure => 5,
            NodeType::RegExp => 6,
            NodeType::Number => 7,
            NodeType::Native => 8,
            NodeType::Syntheic => 9,
            NodeType::ConcatString => 10,
            NodeType::SliceString => 11,
            NodeType::BigInt => 12,
            NodeType::Other(_) => 13,
        }
    }
}

impl<'a> NodeType<'a> {
    fn as_class_name(&self) -> &'static str {
        match self {
            Self::Hidden => "(system)",
            Self::Array => "(array)",
            Self::String => "(string)",
            Self::Object => "(object)",
            Self::Code => "(compiled code)",
            Self::Closure => "(closure)",
            Self::RegExp => "(regexp)",
            Self::Number => "(number)",
            Self::Native => "(native)",
            Self::Syntheic => "(synthetic)",
            Self::ConcatString => "(concatenated string)",
            Self::SliceString => "(sliced string)",
            Self::BigInt => "(bigint)",
            Self::Other(_) => "(unknown)",
        }
    }

    fn from_str_in_arena(strings: &mut Strings<'a>, typ: &str) -> Self {
        match typ {
            "hidden" => Self::Hidden,
            "array" => Self::Array,
            "string" => Self::String,
            "object" => Self::Object,
            "code" => Self::Code,
            "closure" => Self::Closure,
            "regexp" => Self::RegExp,
            "number" => Self::Number,
            "native" => Self::Native,
            "synthetic" => Self::Syntheic,
            "concatenated string" => Self::ConcatString,
            "sliced string" => Self::SliceString,
            "bigint" => Self::BigInt,
            _ => Self::Other(strings.additional(typ)),
        }
    }
}

#[derive(Debug)]
pub struct NodeEdge<'a> {
    pub typ: EdgeType<'a>,
    pub from_index: usize,
    pub to_index: usize,
    pub name: NameOrIndex<'a>,
}

#[derive(Debug)]
pub struct PGNodeEdge<'a> {
    pub typ: EdgeType<'a>,
    pub name: NameOrIndex<'a>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum NameOrIndex<'a> {
    Index(usize),
    Name(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u8)]
pub enum EdgeType<'a> {
    Context,
    Element,
    Property,
    Internal,
    Hidden,
    Shortcut,
    Weak,
    Invisible,
    Other(&'a str),
}

impl<'a> From<EdgeType<'a>> for u8 {
    fn from(t: EdgeType<'a>) -> u8 {
        match t {
            EdgeType::Context => 0,
            EdgeType::Element => 1,
            EdgeType::Property => 2,
            EdgeType::Internal => 3,
            EdgeType::Hidden => 4,
            EdgeType::Shortcut => 5,
            EdgeType::Weak => 6,
            EdgeType::Invisible => 7,
            EdgeType::Other(_) => 8,
        }
    }
}

impl<'a> Display for EdgeType<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeType::Context => write!(f, "context"),
            EdgeType::Element => write!(f, "element"),
            EdgeType::Property => write!(f, "property"),
            EdgeType::Internal => write!(f, "internal"),
            EdgeType::Hidden => write!(f, "hidden"),
            EdgeType::Shortcut => write!(f, "shortcut"),
            EdgeType::Weak => write!(f, "weak"),
            EdgeType::Invisible => write!(f, "invisible"),
            EdgeType::Other(s) => write!(f, "other<{}>", s),
        }
    }
}

impl<'a> EdgeType<'a> {
    fn from_str_in_arena(strings: &mut Strings<'a>, typ: &str) -> Self {
        match typ {
            "context" => Self::Context,
            "element" => Self::Element,
            "property" => Self::Property,
            "internal" => Self::Internal,
            "hidden" => Self::Hidden,
            "shortcut" => Self::Shortcut,
            "weak" => Self::Weak,
            "invisible" => Self::Invisible,
            _ => Self::Other(strings.additional(typ)),
        }
    }
}

pub fn decode_reader(input: impl std::io::Read) -> Result<Graph> {
    serde_json::from_reader(input)
        .map_err(Error::DecodeError)
        .and_then(decode_inner)
}

pub fn decode_slice(input: &[u8]) -> Result<Graph> {
    serde_json::from_slice(input)
        .map_err(Error::DecodeError)
        .and_then(decode_inner)
}

pub fn decode_str(input: &str) -> Result<Graph> {
    serde_json::from_str(input)
        .map_err(Error::DecodeError)
        .and_then(decode_inner)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn decode_bytes(input: &[u8]) -> std::result::Result<Graph, String> {
    decode_slice(input).map_err(|e| e.to_string())
}

struct Strings<'a> {
    bump: &'a Arena<String>,
    indexed: Vec<&'a str>,
    additional: HashMap<String, &'a str>,
}

impl<'a> Strings<'a> {
    pub fn indexed(&mut self, index: usize) -> Result<&'a str> {
        match self.indexed.get(index) {
            Some(i) => Ok(*i),
            None => Ok(self.additional("<missing>")),
        }
    }

    pub fn additional(&mut self, input: &str) -> &'a str {
        match self.additional.get(input) {
            Some(i) => i,
            None => {
                let str = self.bump.alloc(input.to_string());
                self.additional.insert(input.to_string(), str);
                str
            }
        }
    }
}

fn alloc_strs(bump: &Arena<String>, strs: Vec<String>) -> Strings<'_> {
    let mut v = Vec::with_capacity(strs.len());
    for s in strs {
        v.push(bump.alloc(s).as_str());
    }

    Strings {
        bump,
        indexed: v,
        additional: HashMap::new(),
    }
}

fn alloc_nodes<'a, 'snap>(root: &'snap raw::Root, strs: &mut Strings<'a>) -> Result<PetGraph<'a>> {
    let mut name_offset = None;
    let mut id_offset = None;
    let mut self_size_offset = None;
    let mut edge_count_offset = None;
    let mut trace_node_id_offset = None;
    let mut detachedness_offset = None;
    let mut type_offset = None;

    for (i, field) in root.snapshot.meta.node_fields.iter().enumerate() {
        match field.as_str() {
            "name" => name_offset = Some(i),
            "id" => id_offset = Some(i),
            "self_size" => self_size_offset = Some(i),
            "edge_count" => edge_count_offset = Some(i),
            "trace_node_id" => trace_node_id_offset = Some(i),
            "detachedness" => detachedness_offset = Some(i),
            "type" => type_offset = Some(i),
            _ => {}
        }
    }

    let name_offset = name_offset.ok_or(Error::MissingField("name"))?;
    let type_offset = type_offset.ok_or(Error::MissingField("type"))?;
    let type_types = match root.snapshot.meta.node_types.get(type_offset) {
        None => return Err(Error::ColumnOutOfBounds(type_offset)),
        Some(StringOrArray::Single(_)) => return Err(Error::ExpectedTypeArray),
        Some(StringOrArray::Arr(a)) => a,
    };

    let row_size = root.snapshot.meta.node_fields.len();

    let mut graph: PetGraph<'a> = petgraph::Graph::new();
    for i in (0..root.nodes.len()).step_by(row_size) {
        graph.add_node(Node {
            name: strs.indexed(root.nodes[i + name_offset] as usize)?,
            typ: NodeType::from_str_in_arena(
                strs,
                &type_types[root.nodes[i + type_offset] as usize],
            ),
            self_size: self_size_offset
                .map(|o| root.nodes[i + o])
                .unwrap_or_default(),
            edge_count: edge_count_offset
                .map(|o| root.nodes[i + o] as usize)
                .unwrap_or_default(),
            trace_node_id: trace_node_id_offset
                .map(|o| root.nodes[i + o])
                .unwrap_or_default(),
            detachedness: detachedness_offset
                .map(|o| root.nodes[i + o] as u32)
                .unwrap_or_default(),
            id: id_offset
                .map(|o| root.nodes[i + o] as u32)
                .unwrap_or_default(),
        });
    }

    let graph = alloc_edges(root, graph, strs)?;

    Ok(graph)
}

fn alloc_edges<'a>(
    root: &raw::Root,
    mut graph: PetGraph<'a>,
    strs: &mut Strings<'a>,
) -> Result<PetGraph<'a>> {
    let mut to_node_offset = None;
    let mut name_or_index_offset = None;
    let mut type_offset = None;

    for (i, field) in root.snapshot.meta.edge_fields.iter().enumerate() {
        match field.as_str() {
            "to_node" => to_node_offset = Some(i),
            "name_or_index" => name_or_index_offset = Some(i),
            "type" => type_offset = Some(i),
            _ => {}
        }
    }

    let to_node_offset = to_node_offset.ok_or(Error::MissingField("to_node"))?;
    let type_offset = type_offset.ok_or(Error::MissingField("type"))?;
    let name_or_index_offset = name_or_index_offset.ok_or(Error::MissingField("name_or_index"))?;
    let type_types = match root.snapshot.meta.edge_types.get(type_offset) {
        None => return Err(Error::ColumnOutOfBounds(type_offset)),
        Some(StringOrArray::Single(_)) => return Err(Error::ExpectedTypeArray),
        Some(StringOrArray::Arr(a)) => a,
    };

    let row_size = root.snapshot.meta.edge_fields.len();
    let node_row_size = root.snapshot.meta.node_fields.len();

    // Each node own the next "edge_count" edges in the array.
    let mut edge_index = 0;
    let mut edge_queue = Vec::with_capacity(root.edges.len() / row_size);
    for (from_index, node) in graph.raw_nodes().iter().enumerate() {
        for _ in 0..node.weight.edge_count {
            if edge_index + row_size > root.edges.len() {
                return Err(Error::MissingNodeEdge(node.weight.id));
            }

            let typ = EdgeType::from_str_in_arena(
                strs,
                &type_types[root.edges[edge_index + type_offset] as usize],
            );

            let to_index = petgraph::graph::NodeIndex::new(
                (root.edges[edge_index + to_node_offset] as usize) / node_row_size,
            );
            let from_index = petgraph::graph::NodeIndex::new(from_index);
            let name_or_index = root.edges[edge_index + name_or_index_offset];

            edge_queue.push((
                from_index,
                to_index,
                PGNodeEdge {
                    typ,
                    name: NameOrIndex::Name(strs.indexed(name_or_index as usize)?),
                },
            ));
            edge_index += row_size;
        }
    }

    for (from, to, edge) in edge_queue {
        graph.add_edge(from, to, edge);
    }

    Ok(graph)
}

fn decode_inner(mut root: raw::Root) -> Result<Graph> {
    let root_index = root.snapshot.root_index.unwrap_or_default();
    let mut built = GraphInnerBuilder {
        arena: Arena::new(),
        inner_nodes_builder: |arena| {
            let mut strs = Vec::new();
            std::mem::swap(&mut strs, &mut root.strings);

            let mut strs = alloc_strs(arena, strs);
            alloc_nodes(&root, &mut strs)
        },
    }
    .build();

    // slight hack here to pull out the error and return it alone if the
    // builder failed, as oroborous has no 'into_inner_nodes' method.
    if built.borrow_inner_nodes().is_err() {
        return built.with_mut(|fields| {
            let mut out = Err(Error::ExpectedTypeArray);
            std::mem::swap(&mut out, fields.inner_nodes);
            Err(unsafe { out.unwrap_err_unchecked() })
        });
    }

    Ok(Graph::new(built, root_index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_basic_heapsnapshot() {
        let contents = fs::read("test/basic.heapsnapshot").unwrap();
        decode_slice(&contents).expect("expect no errors");
    }
}
