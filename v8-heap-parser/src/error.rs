use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("invalid profile JSON")]
    DecodeError(#[from] serde_json::Error),
    #[error("profile referenced a string index ({0}) that was out of bounds")]
    StringOutOfBounds(usize),
    #[error("profile referenced a column ({0}) that was out of bounds")]
    ColumnOutOfBounds(usize),
    #[error("expected an array of types at index it was not found")]
    ExpectedTypeArray,
    #[error("expected to find a field for '{0}', but we didn't")]
    MissingField(&'static str),
    #[error("missing node {0} in an edge")]
    MissingNodeEdge(u32),
}

pub type Result<T> = std::result::Result<T, Error>;
