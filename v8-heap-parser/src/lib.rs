mod decoder;
mod error;
mod graph;

pub use decoder::{decode_reader, decode_slice, decode_str, EdgeType, Node};

pub use error::*;
pub use graph::*;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
