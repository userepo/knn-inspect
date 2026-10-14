# Fuzzing

One libFuzzer target, `parse_and_verify`, feeds arbitrary bytes to
`knn_format::parse` and `knn_check::verify`. The rule under test: any
input yields a result, never a panic or an out-of-bounds read.

cargo-fuzz needs a nightly toolchain and does not support Windows MSVC;
run it on Linux (WSL works). CI does not run it.

```
cargo install cargo-fuzz
cd fuzz
sh seed.sh
cargo +nightly fuzz run parse_and_verify -- -max_len=200000
```

`seed.sh` fills `corpus/parse_and_verify/` with the committed fixtures,
clean and corrupted, skipping any larger than the 200,000-byte
`-max_len`: libFuzzer would only see their first 200 KB, which is just
another truncated file, a case the truncation tests already cover.
libFuzzer adds to the corpus as it finds new coverage.

The build has debug assertions, overflow checks, and AddressSanitizer on
by default, so an integer overflow fails a run as surely as a panic
does. A crash lands in `artifacts/parse_and_verify/`; reproduce it with

```
cargo +nightly fuzz run parse_and_verify artifacts/parse_and_verify/<file>
```

and turn it into a regression test in `crates/knn-format/tests/` or
`crates/knn-check/tests/` once fixed.
