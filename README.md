# Little Sorry

A Rust library for exploring regret minimization algorithms, with a focus on game theory applications.

## Features

- Regret matching implementation
- Rock Paper Scissors (RPS) example game
- Highly performant using ndarray for numerical operations
- Thread-safe with no unsafe code (except for carefully bounded enum conversions)

## Getting Started

Add this to your `Cargo.toml`:

```toml
[dependencies]
little-sorry = "1.0.0"
```

## How It Works

The library implements regret minimization algorithms, which are used in game theory to find optimal strategies in imperfect-information games. The core algorithm tracks:

1. Action probabilities for each possible move
2. Cumulative regret for not taking alternative actions
3. Strategy updates based on regret matching

The RPS example demonstrates these concepts in a simple zero-sum game setting.

## Building and Testing

```bash
# Run all tests
cargo test

# Run benchmarks
cargo bench

# Build in release mode
cargo build --release
```

## License

Licensed under the Apache License, Version 2.0.
