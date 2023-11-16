use std::fmt::Write;

use crate::cli::{Format, SortBy};
use num_format::{Locale, ToFormattedString};
use serde_json::json;
use v8_heap_parser::{ClassGroup, Graph, Node};

struct TopLevelGroup<'a> {
    group: &'a ClassGroup,
    root: &'a Node,
}

struct NodeWithSize<'a> {
    retained_size: u64,
    index: usize,
    node: &'a Node,
}

trait SummarizableNode {
    fn id(&self) -> u64;
    fn children(&self, graph: &Graph) -> Vec<usize>;
    fn name(&self) -> &str;
    fn retained_size(&self) -> u64;
    fn shallow_size(&self) -> u64;
}

impl<'a> SummarizableNode for TopLevelGroup<'a> {
    fn id(&self) -> u64 {
        self.group.index as u64
    }

    fn children(&self, _graph: &Graph) -> Vec<usize> {
        self.group.nodes.clone()
    }

    fn name(&self) -> &str {
        self.root.class_name()
    }

    fn retained_size(&self) -> u64 {
        self.group.retained_size
    }

    fn shallow_size(&self) -> u64 {
        self.group.self_size
    }
}

impl<'a> SummarizableNode for NodeWithSize<'a> {
    fn id(&self) -> u64 {
        self.node.id as u64
    }

    fn children(&self, graph: &Graph) -> Vec<usize> {
        graph.children(self.index)
    }

    fn name(&self) -> &str {
        self.node.name()
    }

    fn retained_size(&self) -> u64 {
        self.retained_size
    }

    fn shallow_size(&self) -> u64 {
        self.node.self_size
    }
}

pub enum QueryOpt {
    Top(usize),
    Id(u64),
    Name(String),
}

impl QueryOpt {
    fn test(&self, order_index: usize, node: &impl SummarizableNode) -> bool {
        match self {
            QueryOpt::Top(n) => order_index < *n,
            QueryOpt::Id(i) => *i == node.id(),
            QueryOpt::Name(n) => node.name() == n,
        }
    }
}

pub struct SummaryOptions<'a> {
    /// Sort order
    pub sort_by: SortBy,
    /// Output format.
    pub format: Format,
    /// Graph to display
    pub graph: &'a Graph,
    /// List of queries to apply at each level of the graph.
    pub query: Vec<QueryOpt>,
    /// Skip calculation of retained sizes.
    pub no_retained: bool,
}

pub fn print_summary(opts: &SummaryOptions) -> String {
    let groups = opts
        .graph
        .get_class_groups(opts.no_retained)
        .iter()
        .map(|g| TopLevelGroup {
            group: g,
            root: opts.graph.get_node(g.nodes[0]).unwrap(),
        })
        .collect::<Vec<_>>();

    let mut output = String::with_capacity(1024);
    format_print_start(&mut output, opts.format);
    print_summary_inner(
        opts.graph,
        opts.sort_by,
        opts.format,
        &groups,
        &opts.query,
        0,
        &mut output,
    );
    format_print_end(&mut output, opts.format);

    output
}

fn format_print_start(output: &mut String, format: Format) {
    if let Format::Json = format {
        output.push('[');
    }
}
fn format_print_end(output: &mut String, format: Format) {
    if let Format::Json = format {
        output.push_str("]\n");
    }
}

fn format_print_children_start(output: &mut String, format: Format) {
    if let Format::Json = format {
        output.pop(); // remove last '{'
        output.push_str(",\"children\":[");
    }
}
fn format_print_children_end(output: &mut String, format: Format) {
    if let Format::Json = format {
        output.push_str("]}");
    }
}

fn format_print_node(
    order_index: usize,
    node: &impl SummarizableNode,
    format: Format,
    depth: usize,
    output: &mut String,
    num_children: usize,
) {
    match format {
        Format::Json => {
            if order_index > 0 {
                output.push(',');
            }
            write!(
                output,
                "{}",
                json!({
                  "name": node.name(),
                  "self_size": node.shallow_size(),
                  "retained_size": node.retained_size(),
                  "id": node.id(),
                })
            )
            .unwrap();
        }
        Format::Text => {
            let name = node.name();
            for _ in 0..depth {
                output.push('\t');
            }

            output.push_str(&format!("{}. ", order_index + 1));
            if name.len() > 50 {
                output.push_str(&name[0..50]);
                output.push('…');
            } else {
                output.push_str(name);
            }
            output.push_str(" self size ");
            output.push_str(&node.shallow_size().to_formatted_string(&Locale::en));
            output.push_str(" / retained size ");
            output.push_str(&node.retained_size().to_formatted_string(&Locale::en));

            if num_children > 0 {
                output.push_str(&format!(" x{}", num_children));
            }
            output.push_str(&format!(" @ {}\n", node.id()));
        }
    }
}

fn print_summary_inner<T>(
    graph: &Graph,
    sort_by: SortBy,
    format: Format,
    nodes: &[T],
    query: &[QueryOpt],
    depth: usize,
    output: &mut String,
) where
    T: SummarizableNode,
{
    let mut node_indexes = (0..nodes.len()).collect::<Vec<_>>();
    match sort_by {
        SortBy::ShallowSize => {
            node_indexes.sort_by_key(|g| std::cmp::Reverse(nodes[*g].shallow_size()))
        }
        SortBy::RetainedSize => {
            node_indexes.sort_by_key(|g| std::cmp::Reverse(nodes[*g].retained_size()))
        }
    }

    let this_query = query.get(depth).unwrap();

    let mut order_index = 0;
    for (i, index) in node_indexes.iter().enumerate() {
        let node = &nodes[*index];
        if !this_query.test(i, node) {
            continue;
        }

        let children = node.children(graph);
        format_print_node(order_index, node, format, depth, output, children.len());
        order_index += 1;

        if depth + 1 < query.len() {
            let children: Vec<NodeWithSize> = children
                .into_iter()
                .map(|i| NodeWithSize {
                    index: i,
                    node: graph.get_node(i).unwrap(),
                    retained_size: graph.retained_size(i),
                })
                .collect();

            if !children.is_empty() {
                format_print_children_start(output, format);
                print_summary_inner(graph, sort_by, format, &children, query, depth + 1, output);
                format_print_children_end(output, format);
            }
        }
    }
}
