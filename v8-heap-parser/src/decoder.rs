use std::{borrow::Cow, fmt, fmt::Display, rc::Rc};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::{graph::Graph, perf::PerfCounter};

use serde::{
    de::{self, DeserializeSeed, SeqAccess, Visitor},
    Deserialize, Deserializer,
};

#[allow(dead_code)]
pub(crate) struct Root {
    pub snapshot: Snapshot,
    pub graph: PetGraph,
    pub strings: Rc<Vec<String>>,
    pub trace_function_infos: Vec<u32>,
    pub trace_tree: Vec<u32>,
    pub samples: Vec<u32>,
    pub locations: Vec<u32>,
}

#[derive(Deserialize)]
pub(crate) struct Snapshot {
    pub meta: Meta,
    pub root_index: Option<usize>,
    // unused:
    // pub node_count: u64,
    // pub edge_count: u64,
    // pub trace_function_count: u64,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum StringOrArray {
    Single(String),
    Arr(Vec<String>),
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub(crate) struct Meta {
    pub node_fields: Vec<String>,
    pub node_types: Vec<StringOrArray>,
    pub edge_fields: Vec<String>,
    pub edge_types: Vec<StringOrArray>,
    pub trace_function_info_fields: Vec<String>,
    pub trace_node_fields: Vec<String>,
    pub sample_fields: Vec<String>,
}

impl<'de> Deserialize<'de> for Root {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(RootVisitor)
    }
}

struct RootVisitor;

impl<'de> Visitor<'de> for RootVisitor {
    // This Visitor constructs a single Vec<T> to hold the flattened
    // contents of the inner arrays.
    type Value = Root;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "an object map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: de::MapAccess<'de>,
    {
        let mut snapshot: Option<Snapshot> = None;
        let mut graph: Option<PetGraph> = None;
        let mut has_edges = false;

        let mut trace_function_infos = None;
        let mut trace_tree = None;
        let mut samples = None;
        let mut locations = None;
        let mut strings: Option<Vec<String>> = None;

        while let Some(key) = map.next_key::<Cow<'_, str>>()? {
            match key.as_ref() {
                "snapshot" => {
                    snapshot = map.next_value()?;
                }
                "nodes" => {
                    let snapshot = snapshot.as_ref().ok_or_else(|| {
                        de::Error::custom("expected 'snapshot' before 'nodes' field")
                    })?;

                    graph = Some(map.next_value_seed(NodesDeserializer(snapshot))?);
                }
                "edges" => {
                    let snapshot = snapshot.as_ref().ok_or_else(|| {
                        de::Error::custom("expected 'snapshot' before 'edges' field")
                    })?;
                    let graph = graph.as_mut().ok_or_else(|| {
                        de::Error::custom("expected 'nodes' before 'edges' field")
                    })?;

                    map.next_value_seed(EdgesDeserializer(snapshot, graph))?;
                    has_edges = true;
                }

                "trace_function_infos" => {
                    trace_function_infos = Some(map.next_value()?);
                }
                "trace_tree" => {
                    trace_tree = Some(map.next_value()?);
                }
                "samples" => {
                    samples = Some(map.next_value()?);
                }
                "locations" => {
                    locations = Some(map.next_value()?);
                }
                "strings" => {
                    strings = Some(map.next_value()?);
                }
                _ => {}
            }
        }

        if !has_edges {
            return Err(de::Error::missing_field("edges"));
        }
        let snapshot = snapshot.ok_or_else(|| de::Error::missing_field("snapshot"))?;
        let mut graph = graph.ok_or_else(|| de::Error::missing_field("nodes"))?;
        let strings = Rc::new(strings.ok_or_else(|| de::Error::missing_field("strings"))?);

        for node in graph.node_weights_mut() {
            node.strings = Some(strings.clone());
        }

        Ok(Root {
            snapshot,
            graph,
            trace_function_infos: trace_function_infos.unwrap_or_default(),
            trace_tree: trace_tree.unwrap_or_default(),
            samples: samples.unwrap_or_default(),
            locations: locations.unwrap_or_default(),
            strings,
        })
    }
}
struct NodesDeserializer<'a>(&'a Snapshot);

impl<'de, 'a> DeserializeSeed<'de> for NodesDeserializer<'a> {
    // The return type of the `deserialize` method. This implementation
    // appends onto an existing vector but does not create any new data
    // structure, so the return type is ().
    type Value = PetGraph;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Visitor implementation that will walk an inner array of the JSON
        // input.
        struct NodesVisitor<'a>(&'a Snapshot);

        impl<'de, 'a> Visitor<'de> for NodesVisitor<'a> {
            type Value = PetGraph;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "an array of integers")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut name_offset = None;
                let mut id_offset = None;
                let mut self_size_offset = None;
                let mut edge_count_offset = None;
                let mut trace_node_id_offset = None;
                let mut detachedness_offset = None;
                let mut type_offset = None;

                for (i, field) in self.0.meta.node_fields.iter().enumerate() {
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

                let name_offset = name_offset.ok_or(de::Error::missing_field("name"))?;
                let type_offset = type_offset.ok_or(de::Error::missing_field("type"))?;
                let type_types = match self.0.meta.node_types.get(type_offset) {
                    None => return Err(de::Error::missing_field("type")),
                    Some(StringOrArray::Single(_)) => {
                        return Err(de::Error::custom("node `type` should be an array"))
                    }
                    Some(StringOrArray::Arr(a)) => a,
                };

                let row_size = self.0.meta.node_fields.len();

                let mut graph: PetGraph = petgraph::Graph::new();
                let mut buf: Vec<u64> = vec![0; row_size];
                let mut buf_i /* the vampire slayer */ = 0;
                while let Some(elem) = seq.next_element()? {
                    buf[buf_i] = elem;
                    buf_i += 1;

                    if buf_i == row_size {
                        buf_i = 0;
                        graph.add_node(Node {
                            strings: None,
                            name_index: buf[name_offset] as usize,
                            typ: NodeType::from_str(type_types, buf[type_offset] as usize),
                            self_size: self_size_offset.map(|o| buf[o]).unwrap_or_default(),
                            edge_count: edge_count_offset
                                .map(|o| buf[o] as usize)
                                .unwrap_or_default(),
                            trace_node_id: trace_node_id_offset.map(|o| buf[o]).unwrap_or_default(),
                            detachedness: detachedness_offset
                                .map(|o| buf[o] as u32)
                                .unwrap_or_default(),
                            id: id_offset.map(|o| buf[o] as u32).unwrap_or_default(),
                        });
                    }
                }

                Ok(graph)
            }
        }

        deserializer.deserialize_seq(NodesVisitor(self.0))
    }
}

struct EdgesDeserializer<'a>(&'a Snapshot, &'a mut PetGraph);

impl<'de, 'a> DeserializeSeed<'de> for EdgesDeserializer<'a> {
    // The return type of the `deserialize` method. This implementation
    // appends onto an existing vector but does not create any new data
    // structure, so the return type is ().
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Visitor implementation that will walk an inner array of the JSON
        // input.
        struct EdgesVisitor<'a>(&'a Snapshot, &'a mut PetGraph);

        impl<'de, 'a> Visitor<'de> for EdgesVisitor<'a> {
            type Value = ();

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "an array of integers")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<(), A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut to_node_offset = None;
                let mut name_or_index_offset = None;
                let mut type_offset = None;

                for (i, field) in self.0.meta.edge_fields.iter().enumerate() {
                    match field.as_str() {
                        "to_node" => to_node_offset = Some(i),
                        "name_or_index" => name_or_index_offset = Some(i),
                        "type" => type_offset = Some(i),
                        _ => {}
                    }
                }

                let to_node_offset = to_node_offset.ok_or(de::Error::missing_field("to_node"))?;
                let type_offset = type_offset.ok_or(de::Error::missing_field("type"))?;
                let name_or_index_offset =
                    name_or_index_offset.ok_or(de::Error::missing_field("name_or_index"))?;
                let type_types = match self.0.meta.edge_types.get(type_offset) {
                    None => return Err(de::Error::missing_field("type")),
                    Some(StringOrArray::Single(_)) => {
                        return Err(de::Error::custom("edge `type` should be an array"))
                    }
                    Some(StringOrArray::Arr(a)) => a,
                };

                let row_size = self.0.meta.edge_fields.len();
                let node_row_size = self.0.meta.node_fields.len();

                // Each node own the next "edge_count" edges in the array.
                let unexpected_end = || de::Error::custom("unexpected end of edges");
                let nodes_len = self.1.raw_nodes().len();
                for from_index in 0..nodes_len {
                    let edge_count = self.1.raw_nodes()[from_index].weight.edge_count;
                    let from_index = petgraph::graph::NodeIndex::new(from_index);
                    for _ in 0..edge_count {
                        // we know that all the offsets exists and are within
                        // the row_size, so they must be assigned before getting
                        // to the add_edge method.
                        let mut typ: usize = unsafe { std::mem::zeroed() };
                        let mut to_index: usize = unsafe { std::mem::zeroed() };
                        let mut name_or_index: NameOrIndex = unsafe { std::mem::zeroed() };

                        for i in 0..row_size {
                            match i {
                                i if i == to_node_offset => {
                                    to_index = seq.next_element()?.ok_or_else(unexpected_end)?;
                                }
                                i if i == name_or_index_offset => {
                                    name_or_index =
                                        seq.next_element()?.ok_or_else(unexpected_end)?;
                                }
                                i if i == type_offset => {
                                    typ = seq.next_element()?.ok_or_else(unexpected_end)?;
                                }
                                _ => {}
                            }
                        }

                        self.1.add_edge(
                            from_index,
                            petgraph::graph::NodeIndex::new(to_index / node_row_size),
                            PGNodeEdge {
                                typ: EdgeType::from_str(type_types, typ),
                                name: name_or_index,
                            },
                        );
                    }
                }

                Ok(())
            }
        }

        deserializer.deserialize_seq(EdgesVisitor(self.0, self.1))
    }
}

pub(crate) type PetGraph = petgraph::Graph<Node, PGNodeEdge>;

#[derive(Debug)]
pub struct Node {
    name_index: usize,
    strings: Option<Rc<Vec<String>>>,

    pub typ: NodeType,
    pub id: u32,
    pub self_size: u64,
    pub edge_count: usize,
    pub trace_node_id: u64,
    pub detachedness: u32,
}

impl Node {
    pub fn name(&self) -> &str {
        let strs = unsafe { self.strings.as_ref().unwrap_unchecked() };
        &strs[self.name_index]
    }

    pub(crate) fn is_document_dom_trees_root(&self) -> bool {
        self.typ == NodeType::Syntheic && self.name() == "(Document DOM trees)'"
    }

    pub fn class_name(&self) -> &str {
        match &self.typ {
            NodeType::Object | NodeType::Native => self.name(),
            t => t.as_class_name(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[non_exhaustive]
pub enum NodeType {
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
    Other(usize),
}

impl NodeType {
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

    fn from_str(strings: &[String], typ: usize) -> Self {
        match strings.get(typ).map(|s| s.as_str()) {
            Some("hidden") => Self::Hidden,
            Some("array") => Self::Array,
            Some("string") => Self::String,
            Some("object") => Self::Object,
            Some("code") => Self::Code,
            Some("closure") => Self::Closure,
            Some("regexp") => Self::RegExp,
            Some("number") => Self::Number,
            Some("native") => Self::Native,
            Some("synthetic") => Self::Syntheic,
            Some("concatenated string") => Self::ConcatString,
            Some("sliced string") => Self::SliceString,
            Some("bigint") => Self::BigInt,
            _ => Self::Other(typ),
        }
    }
}

#[derive(Debug)]
pub struct NodeEdge {
    pub typ: EdgeType,
    pub from_index: usize,
    pub to_index: usize,
    pub name: NameOrIndex,
}

#[derive(Debug)]
pub struct PGNodeEdge {
    pub typ: EdgeType,
    pub name: NameOrIndex,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum NameOrIndex {
    Index(usize),
    Name(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u8)]
pub enum EdgeType {
    Context,
    Element,
    Property,
    Internal,
    Hidden,
    Shortcut,
    Weak,
    Invisible,
    Other(usize),
}

impl From<EdgeType> for u8 {
    fn from(t: EdgeType) -> u8 {
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

impl Display for EdgeType {
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

impl EdgeType {
    fn from_str(strings: &[String], index: usize) -> Self {
        match strings.get(index).map(|s| s.as_str()) {
            Some("context") => Self::Context,
            Some("element") => Self::Element,
            Some("property") => Self::Property,
            Some("internal") => Self::Internal,
            Some("hidden") => Self::Hidden,
            Some("shortcut") => Self::Shortcut,
            Some("weak") => Self::Weak,
            Some("invisible") => Self::Invisible,
            _ => Self::Other(index),
        }
    }
}

pub fn decode_reader(input: impl std::io::Read) -> Result<Graph, serde_json::Error> {
    // todo@connor412: parsing the JSON takes the majority of time when parsing
    // a graph. We might be faster if we use DeserializeSeed to parse data
    // directly into the graph structure.
    // https://docs.rs/serde/latest/serde/de/trait.DeserializeSeed.html
    let perf = PerfCounter::new("json_decode");
    serde_json::from_reader(input).map(|b| {
        drop(perf);
        to_graph(b)
    })
}

pub fn decode_slice(input: &[u8]) -> Result<Graph, serde_json::Error> {
    let perf = PerfCounter::new("json_decode");
    serde_json::from_slice(input).map(|b| {
        drop(perf);
        to_graph(b)
    })
}

pub fn decode_str(input: &str) -> Result<Graph, serde_json::Error> {
    let perf = PerfCounter::new("json_decode");
    serde_json::from_str(input).map(|b| {
        drop(perf);
        to_graph(b)
    })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn decode_bytes(input: &[u8]) -> std::result::Result<Graph, String> {
    decode_slice(input).map_err(|e| e.to_string())
}

fn to_graph(root: Root) -> Graph {
    let _perf = PerfCounter::new("init_graph");
    let root_index = root.snapshot.root_index.unwrap_or_default();
    Graph::new(root.graph, root_index, root.strings)
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
