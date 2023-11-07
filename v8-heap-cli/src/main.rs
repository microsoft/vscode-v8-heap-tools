use std::{io::BufReader, path::PathBuf};

use clap::Parser;
use thiserror::Error;

mod cli;
mod summary;

use cli::*;
use summary::*;

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

    let depth = match cli.depth {
        0 => cli.node_id.len(),
        n => n,
    };

    let r = print_summary(&SummaryOptions {
        sort_by: cli.sort_by,
        format: cli.format,
        graph,
        query: (0..=depth)
            .map(|d| match (cli.node_id.get(d), &cli.grep) {
                (Some(id), _) => QueryOpt::Id(*id),
                (_, Some(g)) if d == cli.node_id.len() => QueryOpt::Name(g.to_string()),
                _ => QueryOpt::Top(cli.top),
            })
            .collect(),
    });

    println!("{}", r);

    Ok(())
}
