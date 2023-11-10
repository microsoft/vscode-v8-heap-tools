use std::{cmp::min, collections::BinaryHeap, rc::Rc};

use petgraph::visit::EdgeRef;
use wasm_bindgen::prelude::*;

use crate::{decoder::GraphInner, ClassGroup, Graph, Node};

#[wasm_bindgen(js_name = ClassGroup)]
pub struct WasmClassGroup {
    graph: Rc<GraphInner>,
    first_index: usize,

    pub self_size: u64,
    pub retained_size: u64,
    pub children_len: usize,
}

impl WasmClassGroup {
    fn new(group: &ClassGroup, graph: Rc<GraphInner>) -> Self {
        Self {
            graph,
            first_index: group.index,
            self_size: group.self_size,
            retained_size: group.retained_size,
            children_len: group.nodes.len(),
        }
    }
}

#[wasm_bindgen(js_class = ClassGroup)]
impl WasmClassGroup {
    /// Gets the node's string name.
    pub fn name(&self) -> String {
        self.graph.borrow().raw_nodes()[self.first_index]
            .weight
            .class_name()
            .to_string()
    }
}

#[wasm_bindgen(js_name = Node)]
pub struct WasmNode {
    graph: Rc<GraphInner>,

    pub children_len: usize,
    pub self_size: u64,
    pub retained_size: u64,
    pub index: usize,
    pub typ: u8,
    pub id: u32,
}

#[wasm_bindgen(js_class = Node)]
impl WasmNode {
    /// Gets the node's string name.
    pub fn name(&self) -> String {
        self.graph.borrow().raw_nodes()[self.index]
            .weight
            .name
            .to_string()
    }
}

#[derive(Clone, Copy)]
#[wasm_bindgen]

pub enum WasmSortBy {
    SelfSize,
    RetainedSize,
    Name,
}

struct SortedNode<'a> {
    sort: WasmSortBy,
    index: usize,
    retained_size: u64,
    node: &'a Node<'a>,
}

impl<'a> PartialEq for SortedNode<'a> {
    fn eq(&self, other: &Self) -> bool {
        match self.sort {
            WasmSortBy::SelfSize => self.node.self_size == other.node.self_size,
            WasmSortBy::RetainedSize => self.retained_size == other.retained_size,
            WasmSortBy::Name => self.node.name == other.node.name,
        }
    }
}

impl<'a> Eq for SortedNode<'a> {}

impl<'a> Ord for SortedNode<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.sort {
            WasmSortBy::SelfSize => self.node.self_size.cmp(&other.node.self_size).reverse(),
            WasmSortBy::RetainedSize => self.retained_size.cmp(&other.retained_size).reverse(),
            WasmSortBy::Name => self.node.name.cmp(other.node.name),
        }
    }
}

impl<'a> PartialOrd for SortedNode<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[wasm_bindgen]
impl Graph {
    /// Gets a range of class groups, sorted by retained size.
    #[wasm_bindgen(js_name = get_class_groups)]
    pub fn get_class_groups_wasm(&self, start: usize, end: usize) -> Vec<WasmClassGroup> {
        let groups = self.get_class_groups(false);
        groups[start..min(end, groups.len())]
            .iter()
            .map(|g| WasmClassGroup::new(g, self.inner.clone()))
            .collect()
    }

    #[wasm_bindgen(js_name = class_children)]
    pub fn class_children_wasm(
        &self,
        index: usize,
        start: usize,
        end: usize,
        sort_by: WasmSortBy,
    ) -> Vec<WasmNode> {
        self.get_class_groups(false)
            .get(index)
            .map(|g| {
                self.children_of_something(
                    g.nodes.iter().map(|i| petgraph::graph::NodeIndex::new(*i)),
                    start,
                    end,
                    sort_by,
                )
            })
            .unwrap_or_default()
    }

    #[wasm_bindgen(js_name = node_children)]
    pub fn node_children_wasm(
        &self,
        parent: usize,
        start: usize,
        end: usize,
        sort_by: WasmSortBy,
    ) -> Vec<WasmNode> {
        let children = self.graph().edges(petgraph::graph::NodeIndex::new(parent));
        self.children_of_something(children.map(|c| c.target()), start, end, sort_by)
    }
}

impl Graph {
    /// Sorts the given iterator of node indices by the given order, and returns
    /// the nodes. This isn't spectacularly efficient; for reuse, we could
    /// return a structure and allow paging items data out of it  to avoid
    /// re-sorting, at the cost of memory usage.
    fn children_of_something(
        &self,
        iter: impl Iterator<Item = petgraph::graph::NodeIndex<u32>>,
        start: usize,
        end: usize,
        sort_by: WasmSortBy,
    ) -> Vec<WasmNode> {
        let graph = self.graph();
        // Use a binary heap to let us do an insertion sort. This is better than
        // cloning and quicksorting the entire list of nodes, because we know we
        // don't need anything past the 'end' index.
        let mut heap = BinaryHeap::with_capacity(min(end + 1, graph.node_count()));

        for child in iter {
            let sorted = SortedNode {
                node: graph.node_weight(child).unwrap(),
                index: child.index(),
                sort: sort_by,
                retained_size: 0,
            };

            heap.push(sorted);

            if heap.len() > end {
                heap.pop();
            }
        }

        // todo: use into_iter_sorted() when it's stable. For now we just pop
        // off the greatest nodes (which we know are the ones of interest
        // because we enforce the max heap size in the above loop) and then
        // reverse them.
        let mut result = Vec::with_capacity(min(start - end, heap.len()));
        for _ in start..end {
            if let Some(n) = heap.pop() {
                result.push(WasmNode {
                    graph: self.inner.clone(),
                    index: n.index,
                    id: n.node.id,
                    children_len: n.node.edge_count,
                    typ: n.node.typ.into(),
                    retained_size: n.retained_size,
                    self_size: n.node.self_size,
                });
            }
        }
        result.reverse();

        result
    }
}
