# @vscode/v8-heap-parser

This repository contains tools to view and parse V8 heap snapshots. This provides programmatic and command-line access to the data you would normally see in the browser devtools. There are two packages:

- `v8-heap-cli`: a command-line that allows parsing and summarizing a `.heapsnapshot` file
- `v8-heap-parser`: a Rust library, which is also compiled to WebAssembly and installable as `npm install @vscode/v8-heap-parser`, that allows inspecting the heap.

The exposed WebAssembly functions from the parser library are tailored around the needs of our tooling and visualizers. Please see the published type definitions for the latest API.

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
