### Requirement
* Rust nightly
* Arrayfire 3.4.x

On windows, only msvc version is supported.

### How to use

Firstly, clone this project:

```
git clone https://github.com/zhaihj/nmf-arrayfire
```

compile

```
cd nmf-arrayfire; cargo build --release
```

Prepare the data:

data should be in csv format with space delimiter,
where the first line should contain __number of rows, number of columns__

For example:

```
3 5
1.1 0.1 1.5 2.1 0.01
0.0 1.8 0.2 2.5 0.0
1.1 1.2 1.3 1.4 1.5
```

Run:

```
./target/release/nmf-arrayfire ./your-data-file iterations terminate_condition dimension
```

Check the result:

Factorizated matrix will be saved into "w.mat" and "h.mat". You can open it with any editor.
