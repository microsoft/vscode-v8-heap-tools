use std::{io::BufReader, path::PathBuf};

use clap::Parser;
use thiserror::Error;
use v8_heap_parser::Graph;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input to read, a file or "-" for stdin.
    #[arg(default_value = "-")]
    input: String,

    /// How to sort printed nodes.
    #[clap(value_enum, default_value = "retained-size")]
    #[arg(long, short)]
    sort_by: SortBy,

    /// Show only nodes with the given name.
    #[arg(long, short)]
    grep: Option<String>,

    /// Show only the given top-level node.
    #[arg(long)]
    node_id: Option<u64>,

    /// Number of additional children to show for each node.
    #[arg(short, long, default_value = "0")]
    depth: usize,

    /// How many nodes to print.
    #[arg(short, long, default_value = "50")]
    top: usize,
}

#[derive(clap::ValueEnum, Clone, Copy)]
enum SortBy {
    ShallowSize,
    RetainedSize,
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("could not read input")]
    CouldNotReadInput(std::io::Error),
    #[error(transparent)]
    FromHeap(#[from] v8_heap_parser::Error),
}

fn main() -> Result<(), Error> {
    let cli = Cli::parse();

    let graph = match cli.input.as_str() {
        "-" => {
            let stdin = std::io::stdin();
            let stdin = stdin.lock();
            v8_heap_parser::decode_reader(BufReader::new(stdin))?
        }
        f => {
            let path = PathBuf::from(f);
            let file = std::fs::File::open(path).map_err(Error::CouldNotReadInput)?;
            v8_heap_parser::decode_reader(BufReader::new(file))?
        }
    };

    let mut nodes_indexes = match cli.node_id {
        None => (0..graph.nodes().len()).collect::<Vec<_>>(),
        Some(i) => graph
            .iter()
            .enumerate()
            .find(|(_, n)| n.id == i)
            .map(|(i, _)| vec![i])
            .unwrap(),
    };

    if let Some(g) = &cli.grep {
        nodes_indexes = nodes_indexes
            .into_iter()
            .filter(|i| graph.get_node(*i).unwrap().name.contains(g))
            .collect::<Vec<_>>();
    }

    print(&cli, &graph, nodes_indexes, 0);

    Ok(())
}

fn print(cli: &Cli, graph: &Graph, mut nodes_indexes: Vec<usize>, depth: usize) {
    let nodes = graph.nodes();

    match cli.sort_by {
        SortBy::ShallowSize => nodes_indexes.sort_by(|a, b| {
            graph
                .get_node(*b)
                .unwrap()
                .self_size
                .cmp(&graph.get_node(*a).unwrap().self_size)
        }),
        SortBy::RetainedSize => {
            nodes_indexes.sort_by_key(|n| std::cmp::Reverse(graph.retained_size(*n)))
        }
    }

    let node_indexes = &nodes_indexes[0..cli.top.min(nodes_indexes.len())];
    let indent = "  ".repeat(depth);
    for (i, index) in node_indexes.iter().enumerate() {
        let retained = graph.retained_size(*index);
        let node = &nodes[*index].weight;
        println!(
            "{}{}. {}, self size {} / retained size {} @ {}",
            indent,
            i + 1,
            if node.name.len() > 50 {
                &node.name[0..50]
            } else {
                node.name
            }
            .replace('\n', "\\n")
            .replace('\r', "\\r"),
            node.self_size,
            retained,
            node.id,
        );

        if depth < cli.depth {
            print(cli, graph, graph.children(i), depth + 1);
        }
    }

    // println!("Total retained size: {}", sum);
}
