# @vscode/v8-heap-parser

[![crates.io](https://img.shields.io/crates/v/v8-heap-parser.svg)](https://crates.io/crates/v8-heap-parser)
[![Documentation](https://docs.rs/v8-heap-parser/badge.svg)](https://docs.rs/v8-heap-parser)
[![BSD-3-Clause](https://img.shields.io/crates/l/v8-heap-parser.svg)](./LICENSE)

This package contains a library to parse and analyze V8 heap snapshots. It is published both as a [Rust crate](https://crates.io/crates/v8-heap-parser) and [npm package](https://www.npmjs.com/package/@vscode/v8-heap-parser) with WebAssembly bindings. It takes as input in the V8 `.heapshot` format, parses it to a graph structure, and exposes information about that graph.

The APIs exposed by the Rust package and JavaScript package are separate: heap snapshots are very large, and we have optimized the JavaScript bindings for usage in VS Code's visualization tools. However, if there's more information you need from the bindings, please make a pull requeset!

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
