use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Input to read, a file or "-" for stdin.
    #[arg(default_value = "-")]
    pub input: String,

    /// How to sort printed nodes.
    #[clap(value_enum, default_value = "retained-size")]
    #[arg(long, short)]
    pub sort_by: SortBy,

    /// How to display output
    #[clap(value_enum, default_value = "format", default_value = "text")]
    #[arg(long, short)]
    pub format: Format,

    /// Show only nodes with the given name.
    #[arg(long, short)]
    pub grep: Option<String>,

    /// Filter node IDs to show. Can be repeated to select nested nodes.
    #[arg(long, short, short_alias = 'i')]
    pub node_id: Vec<u64>,

    /// Number of additional children to show for each node.
    #[arg(short, long, default_value = "0")]
    pub depth: usize,

    /// How many nodes to print.
    #[arg(short, long, default_value = "50")]
    pub top: usize,
}

#[derive(clap::ValueEnum, Clone, Copy)]
pub enum SortBy {
    ShallowSize,
    RetainedSize,
}

#[derive(clap::ValueEnum, Clone, Copy)]
pub enum Format {
    Text,
    JSON,
}
