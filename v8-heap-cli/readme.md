# v8-heap CLI

This is a command-line tool that allows querying information from a V8 heap snapshot. It also supports running multiple query operations. We can run this automatically in VS Code's continuosu integration tests to check for state and memory leaks.

## Usage

The basic usage, which outputs something that you'd get in the Chromium devtools, is fairly simple:

```bash
v8-heap my.heapsnapshot -t 10 # prints the top 10 most-retaining nodes
```

You can also specify the `--output` format as JSON or text, and also run multiple operations. This is useful if you want to look at a few different aspects of the data without the overhead of parsing the snapshot and interrogating the graph multiple times.

```bash
v8-heap my.heapsnapshot --output json \
  --op "-t 10" \ # print the top 10 nodes
  --op "-g MyObject" \ # grep for and show information about MyObject
```

See `--help` for full information.
