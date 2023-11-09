use std::{
    cell::UnsafeCell,
    collections::{HashMap, HashSet, VecDeque},
    rc::Rc,
};

use crate::decoder::*;
use petgraph::visit::EdgeRef;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Maps node indices to and back from the DFS order.
struct PostOrder {
    // Node index to the order in which they're iterated
    index_to_order: Vec<usize>,
    // Order in which nodes are iterated to the node index
    order_to_index: Vec<usize>,
}

struct DominatedNodes {
    first_dom_node_index: Vec<usize>,
    dominated_nodes: Vec<usize>,
}

struct Retaining {
    nodes: Vec<usize>,
    edges: Vec<usize>,
    first_retainer: Vec<usize>,
    first_edge: Vec<usize>,
}

#[derive(Clone)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter_with_clone))]
pub struct ClassGroup {
    pub index: usize,
    pub self_size: u64,
    pub retained_size: u64,
    pub nodes: Vec<usize>,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Graph {
    // Much of the code in the Graph is based on that in the Chrome DevTools
    // (see LICENSE). We used Github Copilot to do the bulk of the translation from
    // TypeScript to Rust, with ample hand-editing afterwards. Chrome DevTools
    // prefers to store all node/edge information in equivalently-lengthed arrays,
    // which makes V8 very happy. In native code, we can do things a little more
    // efficiently, but there's still chunks of the graph that mirror devtools and
    // could be made more idiomatic.
    pub(crate) inner: Rc<GraphInner>,
    pub root_index: usize,
    dominators: UnsafeCell<Option<Vec<usize>>>,
    retained_sizes: UnsafeCell<Option<Vec<u64>>>,
    flags: UnsafeCell<Option<Flags>>,
    retainers: UnsafeCell<Option<Retaining>>,
    post_order: UnsafeCell<Option<PostOrder>>,
    dominated_nodes: UnsafeCell<Option<DominatedNodes>>,
    class_groups: UnsafeCell<Option<Vec<ClassGroup>>>,
}

mod flag {
    pub const CAN_BE_QUERIED: u8 = 1 << 0;
    pub const DETACHED_DOM_TREE_NODE: u8 = 1 << 1;
    pub const PAGE_OBJECT: u8 = 1 << 2;
}

struct Flags(Vec<u8>);

impl Flags {
    pub fn test(&self, index: usize, flag: u8) -> bool {
        (self.0[index] & flag) != 0
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = Node)]
pub struct WasmNode {
    graph: Rc<GraphInner>,

    pub self_size: u64,
    pub index: usize,
    pub typ: u8,
}

#[cfg(target_arch = "wasm32")]
impl WasmNode {
    pub(crate) fn maybe_new(graph: Rc<GraphInner>, index: usize) -> Option<Self> {
        let (self_size, typ) = {
            let node = graph.borrow().raw_nodes().get(index)?;
            (node.weight.self_size, node.weight.typ.into())
        };

        Some(Self {
            index,
            graph,
            self_size,
            typ,
        })
    }
}

#[cfg(target_arch = "wasm32")]
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Graph {
    /// Gets a range of class groups, sorted by retained size.
    #[wasm_bindgen(js_name = get_class_groups)]
    pub fn get_class_groups_wasm(
        &self,
        start: usize,
        end: usize,
        sort_by_retained: bool,
    ) -> Vec<ClassGroup> {
        let groups = self.get_class_groups(false);
        let range = start..std::cmp::min(end, groups.len());
        if !sort_by_retained {
            let mut indices = (0..groups.len()).collect::<Vec<_>>();
            indices.sort_by_key(|g| std::cmp::Reverse(groups[*g].self_size));
            return indices[range].iter().map(|i| groups[*i].clone()).collect();
        }

        groups[range].to_vec()
    }

    /// Gets a list of nodes by their index.
    #[wasm_bindgen(js_name = get_nodes)]
    pub fn get_nodes_wasm(&self, nodes: &[usize]) -> Vec<WasmNode> {
        nodes
            .iter()
            .flat_map(|n| WasmNode::maybe_new(self.inner.clone(), *n))
            .collect()
    }

    /// Gets a list children of the node at the given index. The return value
    /// is a list of numbers where the top 8 bits are the edge type, and the
    /// remaining bits are the node index.
    #[wasm_bindgen(js_name = children)]
    pub fn children_wasm(&self, parent: usize) -> Vec<u64> {
        let graph = self.graph();

        graph
            .edges(petgraph::graph::NodeIndex::new(parent))
            .map(|n| {
                let typ: u8 = n.weight().typ.into();
                (n.target().index() as u64) | ((typ as u64) << (64 - 8))
            })
            .collect()
    }
}

impl Graph {
    pub(crate) fn new(inner: GraphInner, root_index: usize) -> Self {
        Self {
            inner: Rc::new(inner),
            root_index,
            dominators: UnsafeCell::new(None),
            retained_sizes: UnsafeCell::new(None),
            retainers: UnsafeCell::new(None),
            post_order: UnsafeCell::new(None),
            flags: UnsafeCell::new(None),
            dominated_nodes: UnsafeCell::new(None),
            class_groups: UnsafeCell::new(None),
        }
    }

    /// Gets a list of all nodes in the graph.
    pub fn nodes(&self) -> &[petgraph::graph::Node<Node<'_>>] {
        self.inner.borrow().raw_nodes()
    }

    /// Gets a node by its index.
    pub fn get_node(&self, index: usize) -> Option<&Node<'_>> {
        self.inner
            .borrow()
            .raw_nodes()
            .get(index)
            .map(|n| &n.weight)
    }

    fn graph(&self) -> &PetGraph<'_> {
        self.inner.borrow()
    }

    /// Gets an iterator over the graph nodes.
    pub fn iter(&self) -> NodeIterator<'_, '_> {
        NodeIterator {
            graph: self.graph(),
            index: 0,
        }
    }

    /// Gets a list children of the node at the given index.
    pub fn children(&self, parent: usize) -> Vec<usize> {
        let graph = self.graph();

        graph
            .neighbors(petgraph::graph::NodeIndex::new(parent))
            .map(|n| n.index())
            .collect()
    }

    /// Gets the retained size of a node in the graph. This is the size this
    /// node specifically requires the program to retain., i.e. the nodes
    /// the given node dominates https://en.wikipedia.org/wiki/Dominator_(graph_theory)
    pub fn retained_size(&self, node_index: usize) -> u64 {
        let retained_sizes = unsafe { &mut *self.retained_sizes.get() };

        if retained_sizes.is_none() {
            let graph = self.graph();
            let mut rs = Vec::with_capacity(graph.node_count());
            for n in self.graph().raw_nodes() {
                rs.push(n.weight.self_size);
            }

            let post_order = &self.get_post_order().order_to_index;
            let dom = self.get_dominators();
            for i in post_order.iter() {
                if *i != self.root_index {
                    rs[dom[*i]] += rs[*i];
                }
            }

            *retained_sizes = Some(rs);
        }

        retained_sizes.as_ref().unwrap()[node_index]
    }

    fn get_dominated_nodes(&self) -> &DominatedNodes {
        let dominated_nodes_o = unsafe { &mut *self.dominated_nodes.get() };

        if dominated_nodes_o.is_none() {
            let graph = self.graph();
            let mut dominated_nodes = vec![0; graph.node_count()];
            let mut first_dom_node_index = vec![0; graph.node_count() + 1];

            let dominators_tree = self.get_dominators();
            let range = match self.root_index {
                0 => 1..graph.node_count(),
                i if i == graph.node_count() - 1 => 0..graph.node_count() - 1,
                i => panic!("expected root index to be first or last, was {}", i),
            };

            // clone the range as it's used again later
            for node_index in range.clone() {
                first_dom_node_index[dominators_tree[node_index]] += 1;
            }

            let mut first_dominated_node_index = 0;
            #[allow(clippy::needless_range_loop)]
            for i in 0..graph.node_count() {
                let dominated_count = first_dom_node_index[i];
                if i < graph.node_count() - 1 {
                    dominated_nodes[first_dominated_node_index] = dominated_count;
                }
                first_dom_node_index[i] = first_dominated_node_index;
                first_dominated_node_index += dominated_count;
            }
            first_dom_node_index[graph.node_count()] = dominated_nodes.len() - 1;

            for node_index in range {
                let dominator_ordinal = dominators_tree[node_index];
                let mut dominated_ref_index = first_dom_node_index[dominator_ordinal];
                dominated_nodes[dominated_ref_index] -= 1;
                dominated_ref_index += dominated_nodes[dominated_ref_index];
                dominated_nodes[dominated_ref_index] = node_index;
            }

            *dominated_nodes_o = Some(DominatedNodes {
                dominated_nodes,
                first_dom_node_index,
            });
        }

        dominated_nodes_o.as_ref().unwrap()
    }

    /// Gets top-level groups for classes to show in a summary view. Sorted by
    /// retained size descending by default.
    pub fn get_class_groups(&self, no_retained: bool) -> &[ClassGroup] {
        let class_groups = unsafe { &mut *self.class_groups.get() };

        if class_groups.is_none() {
            let nodes = self.nodes();
            let mut groups = HashMap::new();
            for (index, node) in nodes.iter().enumerate() {
                let name = node.weight.class_name();
                let group = groups.entry(name).or_insert(ClassGroup {
                    index,
                    self_size: 0,
                    retained_size: 0,
                    nodes: vec![],
                });

                group.self_size += node.weight.self_size;
                group.nodes.push(index);
            }

            if !no_retained {
                let dominators = self.get_dominated_nodes();
                let mut queue = VecDeque::new();
                let mut sizes = vec![-1];
                let mut classes = vec![];
                queue.push_back(self.root_index);

                let mut seen_groups: HashSet<&str> = HashSet::new();
                while let Some(node_index) = queue.pop_back() {
                    let node = &nodes[node_index];
                    let name = node.weight.class_name();
                    let seen = seen_groups.contains(name);

                    let dominated_index_from = dominators.first_dom_node_index[node_index];
                    let dominated_index_to = dominators.first_dom_node_index[node_index + 1];

                    if !seen
                        && (node.weight.self_size != 0
                            || matches!(node.weight.typ, NodeType::Native))
                    {
                        // group must eexist from built groups above
                        let group = groups.get_mut(name).unwrap();
                        group.retained_size += self.retained_size(node_index);
                        if dominated_index_from != dominated_index_to {
                            seen_groups.insert(name);
                            sizes.push(queue.len() as isize);
                            classes.push(name);
                        }
                    }

                    for i in dominated_index_from..dominated_index_to {
                        queue.push_back(dominators.dominated_nodes[i]);
                    }

                    let l = queue.len();
                    while sizes.last().copied() == Some(l as isize) {
                        sizes.pop();
                        seen_groups.remove(classes.pop().unwrap());
                    }
                }
            }

            let mut groups: Vec<_> = groups.into_values().collect();
            groups.sort_by_key(|g| std::cmp::Reverse(g.retained_size));
            *class_groups = Some(groups);
        }

        class_groups.as_ref().unwrap()
    }

    fn get_retainers(&self) -> &Retaining {
        let retaining = unsafe { &mut *self.retainers.get() };

        if retaining.is_none() {
            let graph = self.graph();
            let mut r = Retaining {
                edges: vec![0; graph.edge_count()],
                nodes: vec![0; graph.edge_count()],
                first_retainer: vec![0; graph.node_count() + 1],
                first_edge: vec![0; graph.node_count() + 1],
            };

            r.first_edge[graph.node_count()] = graph.edge_count();
            let mut edge_index = 0;
            for (i, node) in graph.raw_nodes().iter().enumerate() {
                r.first_edge[i] = edge_index;
                edge_index += node.weight.edge_count as usize;
            }

            for edge in graph.raw_edges() {
                r.first_retainer[edge.target().index()] += 1;
            }

            let mut first_unused_retainer_slot = 0;
            for i in 0..graph.node_count() {
                let retainers_count = r.first_retainer[i];
                r.first_retainer[i] = first_unused_retainer_slot;
                r.nodes[first_unused_retainer_slot] = retainers_count;
                first_unused_retainer_slot += retainers_count;
            }
            r.first_retainer[graph.node_count()] = r.nodes.len();

            for node in graph.node_indices() {
                for edge in graph.edges(node) {
                    let first_retainer_slot_index = r.first_retainer[edge.target().index()];
                    r.nodes[first_retainer_slot_index] -= 1;
                    let next_unused_retainer_slot_index =
                        first_retainer_slot_index + r.nodes[first_retainer_slot_index];
                    r.nodes[next_unused_retainer_slot_index] = node.index();
                    r.edges[next_unused_retainer_slot_index] = edge.id().index();
                }
            }

            *retaining = Some(r);
        }

        retaining.as_ref().unwrap()
    }

    fn get_dominators(&self) -> &Vec<usize> {
        let dominators = unsafe { &mut *self.dominators.get() };
        if dominators.is_none() {
            *dominators = Some(self.build_dominators());
        }

        dominators.as_ref().unwrap()
    }

    fn is_essential_edge(&self, index: usize, typ: &EdgeType<'_>) -> bool {
        if let EdgeType::Weak = typ {
            return false;
        }
        if let EdgeType::Shortcut = typ {
            return index == self.root_index;
        }

        true
    }

    /// Gets the flags of metadata for the graph.
    fn get_flags(&self) -> &Flags {
        let flags = unsafe { &mut *self.flags.get() };

        if flags.is_none() {
            let mut f = vec![0; self.graph().node_count()];
            self.mark_detached_dom_tree_nodes(&mut f);
            self.mark_queriable_heap_objects(&mut f);
            self.mark_page_owned_nodes(&mut f);
            *flags = Some(Flags(f));
        }

        flags.as_ref().unwrap()
    }

    fn mark_detached_dom_tree_nodes(&self, flags: &mut [u8]) {
        for (index, node) in self.graph().raw_nodes().iter().enumerate() {
            if let NodeType::Native = node.weight.typ {
                if node.weight.name.starts_with("Detached ") {
                    flags[index] |= flag::DETACHED_DOM_TREE_NODE;
                }
            }
        }
    }

    fn mark_queriable_heap_objects(&self, flags: &mut [u8]) {
        let graph = self.graph();
        let mut list = VecDeque::new();
        let retained = self.get_retainers();

        while let Some(node_index) = list.pop_front() {
            if flags[node_index] & flag::CAN_BE_QUERIED != 0 {
                continue;
            }
            flags[node_index] |= flag::CAN_BE_QUERIED;
            let begin_edge = retained.first_edge[node_index];
            let end_edge = retained.first_edge[node_index + 1];
            for edge in graph.raw_edges()[begin_edge..end_edge].iter() {
                if flags[edge.target().index()] & flag::CAN_BE_QUERIED != 0 {
                    continue;
                }
                let typ = edge.weight.typ;
                if typ == EdgeType::Hidden
                    || typ == EdgeType::Invisible
                    || typ == EdgeType::Internal
                    || typ == EdgeType::Weak
                {
                    continue;
                }
                list.push_back(edge.target().index());
            }
        }
    }

    fn mark_page_owned_nodes(&self, flags: &mut [u8]) {
        let retainers = self.get_retainers();
        let graph = self.graph();
        let node_count = graph.raw_nodes().len();
        let mut nodes_to_visit = vec![0; node_count];
        let mut nodes_to_visit_length = 0;

        // Populate the entry points. They are Window objects and DOM Tree Roots.
        for edge_index in
            retainers.first_edge[self.root_index]..retainers.first_edge[self.root_index + 1]
        {
            let edge = &graph.raw_edges()[edge_index];
            if edge.weight.typ == EdgeType::Element {
                if !graph[edge.target()].is_document_dom_trees_root() {
                    continue;
                }
            } else if edge.weight.typ != EdgeType::Shortcut {
                continue;
            }

            nodes_to_visit[nodes_to_visit_length] = edge.target().index();
            flags[edge.target().index()] |= flag::PAGE_OBJECT;
            nodes_to_visit_length += 1;
        }

        // Mark everything reachable with the pageObject flag.
        while nodes_to_visit_length > 0 {
            let node_ordinal = nodes_to_visit[nodes_to_visit_length - 1];
            nodes_to_visit_length -= 1;

            let edge_begin = retainers.first_edge[node_ordinal];
            let edge_end = retainers.first_edge[node_ordinal + 1];
            for edge in graph.raw_edges()[edge_begin..edge_end].iter() {
                let child_index = edge.target().index();
                if flags[child_index] & flag::PAGE_OBJECT != 0 {
                    continue;
                }
                if edge.weight.typ == EdgeType::Weak {
                    continue;
                }
                nodes_to_visit[nodes_to_visit_length] = child_index;
                flags[child_index] |= flag::PAGE_OBJECT;
                nodes_to_visit_length += 1;
            }
        }
    }

    /// Gets information about the DFS order of nodes in the tree.
    fn get_post_order(&self) -> &PostOrder {
        let post_order = unsafe { &mut *self.post_order.get() };
        if post_order.is_none() {
            *post_order = Some(self.build_post_order());
        }

        post_order.as_ref().unwrap()
    }

    fn build_post_order(&self) -> PostOrder {
        let retainers = self.get_retainers();
        let graph = self.graph();
        let node_count = graph.raw_nodes().len();
        let flags = self.get_flags();
        let mut stack_nodes = vec![0; node_count];
        let mut stack_current_edge = vec![0; node_count];
        let mut order_to_index = vec![0; node_count];
        let mut index_to_order = vec![0; node_count];
        let mut visited = vec![false; node_count];
        let mut post_order_index = 0;

        let mut stack_top = 0;
        stack_nodes[0] = self.root_index;
        stack_current_edge[0] = retainers.first_edge[self.root_index];
        visited[self.root_index] = true;

        let mut iteration = 0;
        loop {
            iteration += 1;
            loop {
                let node_index = stack_nodes[stack_top];
                let edge_index = stack_current_edge[stack_top];
                let edges_end = retainers.first_edge[node_index + 1];

                if edge_index < edges_end {
                    stack_current_edge[stack_top] += 1;
                    let edge = &graph.raw_edges()[edge_index];
                    if !self.is_essential_edge(node_index, &edge.weight.typ) {
                        continue;
                    }
                    let child_node_index = edge.target().index();
                    if visited[child_node_index] {
                        continue;
                    }

                    if node_index != self.root_index
                        && flags.test(child_node_index, flag::PAGE_OBJECT)
                        && !flags.test(node_index, flag::PAGE_OBJECT)
                    {
                        continue;
                    }

                    stack_top += 1;
                    stack_nodes[stack_top] = child_node_index;
                    stack_current_edge[stack_top] = retainers.first_edge[child_node_index];
                    visited[child_node_index] = true;
                } else {
                    index_to_order[node_index] = post_order_index;
                    order_to_index[post_order_index] = node_index;
                    post_order_index += 1;

                    if stack_top == 0 {
                        break;
                    }

                    stack_top -= 1;
                }
            }

            if post_order_index == node_count || iteration > 1 {
                break;
            }

            post_order_index -= 1;
            stack_top = 0;
            stack_nodes[0] = self.root_index;
            stack_current_edge[0] = retainers.first_edge[self.root_index + 1];
        }

        if post_order_index != node_count {
            post_order_index -= 1;
            for i in 0..node_count {
                if visited[i] {
                    continue;
                }
                index_to_order[i] = post_order_index;
                order_to_index[post_order_index] = i;
                post_order_index += 1;
            }
            index_to_order[self.root_index] = post_order_index;
            order_to_index[post_order_index] = self.root_index;
        }

        PostOrder {
            index_to_order,
            order_to_index,
        }
    }

    fn build_dominators(&self) -> Vec<usize> {
        let post_order = self.get_post_order();
        let graph = self.graph();

        let flags = self.get_flags();
        let retaining = self.get_retainers();
        let nodes_count = self.nodes().len();
        let root_post_ordered_index = nodes_count - 1;
        let no_entry = nodes_count;
        let mut dominators = vec![no_entry; nodes_count];
        dominators[root_post_ordered_index] = root_post_ordered_index;

        // The affected vector is used to mark entries which dominators
        // have to be recalculated because of changes in their retainers.
        let mut affected: Vec<u8> = vec![0; nodes_count];

        // Mark the root direct children as affected.
        {
            let begin_index = retaining.first_edge[self.root_index];
            let end_index = retaining.first_edge[self.root_index + 1];
            for edge in graph.raw_edges()[begin_index..end_index].iter() {
                if !self.is_essential_edge(self.root_index, &edge.weight.typ) {
                    continue;
                }

                let index = post_order.index_to_order[edge.target().index()];
                affected[index] = 1;
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..root_post_ordered_index {
                let post_order_index = root_post_ordered_index - (i + 1);
                if affected[post_order_index] == 0 {
                    continue;
                }
                affected[post_order_index] = 0;
                if dominators[post_order_index] == root_post_ordered_index {
                    continue;
                }
                let node_index = post_order.order_to_index[post_order_index];
                let mut new_dominator_index = no_entry;
                let begin_retainer_index = retaining.first_retainer[node_index];
                let end_retainer_index = retaining.first_retainer[node_index + 1];
                let mut orphan_node = true;
                for retainer_index in begin_retainer_index..end_retainer_index {
                    let retainer_edge_index = retaining.edges[retainer_index];
                    let retainer_edge_type = &graph.raw_edges()[retainer_edge_index].weight.typ;
                    let retainer_node_index = retaining.nodes[retainer_index];
                    if !self.is_essential_edge(retainer_node_index, retainer_edge_type) {
                        continue;
                    }
                    orphan_node = false;
                    if retainer_node_index != self.root_index
                        && flags.test(node_index, flag::PAGE_OBJECT)
                        && !flags.test(retainer_node_index, flag::PAGE_OBJECT)
                    {
                        continue;
                    }
                    let mut retainer_post_order_index =
                        post_order.index_to_order[retainer_node_index];
                    if dominators[retainer_post_order_index] != no_entry {
                        if new_dominator_index == no_entry {
                            new_dominator_index = retainer_post_order_index;
                        } else {
                            while retainer_post_order_index != new_dominator_index {
                                while retainer_post_order_index < new_dominator_index {
                                    retainer_post_order_index =
                                        dominators[retainer_post_order_index];
                                }
                                while new_dominator_index < retainer_post_order_index {
                                    new_dominator_index = dominators[new_dominator_index];
                                }
                            }
                        }
                        if new_dominator_index == root_post_ordered_index {
                            break;
                        }
                    }
                }
                if orphan_node {
                    new_dominator_index = root_post_ordered_index;
                }
                if new_dominator_index != no_entry
                    && dominators[post_order_index] != new_dominator_index
                {
                    dominators[post_order_index] = new_dominator_index;
                    let node_index = post_order.order_to_index[post_order_index];
                    changed = true;
                    let begin_index = retaining.first_edge[node_index];
                    let end_index = retaining.first_edge[node_index + 1];
                    for edge in graph.raw_edges()[begin_index..end_index].iter() {
                        affected[post_order.index_to_order[edge.target().index()]] = 1;
                    }
                }
            }
        }

        let mut dominators_tree = vec![0; nodes_count];
        for (post_order_index, dominated_by) in dominators.into_iter().enumerate() {
            let node_index = post_order.order_to_index[post_order_index];
            dominators_tree[node_index] = post_order.order_to_index[dominated_by];
        }

        dominators_tree
    }
}

pub struct NodeIterator<'a, 'graph> {
    graph: &'graph PetGraph<'a>,
    index: usize,
}

impl<'a, 'graph> Iterator for NodeIterator<'a, 'graph> {
    type Item = &'graph Node<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.graph.raw_nodes().get(self.index).map(|n| &n.weight);
        self.index += 1;
        n
    }
}
