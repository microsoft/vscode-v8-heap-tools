use std::{io::BufReader, path::PathBuf};

use clap::Parser;
use thiserror::Error;

mod cli;
mod summary;

use cli::*;
use summary::*;
use v8_heap_parser::Graph;

#[derive(Error, Debug)]
pub enum Error {
    #[error("could not read input")]
    CouldNotReadInput(std::io::Error),
    #[error(transparent)]
    FromHeap(#[from] serde_json::Error),
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

    if !cli.op.is_empty() {
        for op in cli.op {
            let mut sub_args = shlex::split(&op).unwrap_or_default();
            sub_args.insert(0, cli.input.clone()); // just to make the parser happy
            sub_args.push(format!("--format={}", cli.format));
            print_divider(cli.format, &sub_args);
            do_print_summary(&graph, Cli::parse_from(sub_args));
        }
    } else {
        do_print_summary(&graph, cli);
    }

    Ok(())
}

fn do_print_summary(graph: &Graph, cli: Cli) {
    let depth = match cli.depth {
        0 => cli.node_id.len(),
        n => n,
    };

    let r = print_summary(&SummaryOptions {
        sort_by: match cli.no_retained {
            true => SortBy::ShallowSize,
            false => cli.sort_by,
        },
        format: cli.format,
        graph,
        no_retained: cli.no_retained,
        query: (0..=depth)
            .map(|d| match (cli.node_id.get(d), &cli.grep) {
                (Some(id), _) => QueryOpt::Id(*id),
                (_, Some(g)) if d == cli.node_id.len() => QueryOpt::Name(g.to_string()),
                _ => QueryOpt::Top(cli.top),
            })
            .collect(),
    });

    print!("{}", r);
}

fn print_divider(format: Format, args: &[String]) {
    if let Format::Text = format {
        println!();
        println!("{}:", args.join(" "));
    }
}
