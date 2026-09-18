# microgpt in Bend

A port of Karpathy's [`microgpt.py`](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py)
(pure-Python GPT: tokenizer, scalar autograd, 1-layer transformer, Adam,
sampling) to [Bend](https://bend-lang.com), run to a working 1000-step
training run with decreasing loss. Outputs are babble-grade — the point
proven is that it trains, not what it dreams.

## Result

1000 steps, same hyperparams as the gist (1 layer, 16-wide, 4 heads,
block 16, lr 0.01, Adam 0.85/0.99), 4192 params, vocab 27 (a–z + BOS).

Mean per-doc loss: **2.642 (first 100) → 2.249 (last 100)**,
perplexity ≈ 9.5. Random init opens at 3.28 = ln 27, as it should.

```
step    0-9: 3.28 3.11 2.92 3.18 2.86 3.42 2.23 2.40 3.49 3.34
step 990-999: 2.17 2.91 1.66 1.92 1.95 2.60 2.74 1.46 2.09 1.47
```

Inference after 1000 steps (greedy — collapses to the mode, all 20 identical):

```
?anala?a??a
```

Stochastic (Gumbel-max, T=0.5) samples from the same run family are diverse
but untrained-BOS-heavy, e.g. early in training:

```
?mqm?ux?xgz
?ms?xmztm?m
?mgm?uw?xnx
?mrw?myzql?
```

(`?` = BOS predicted mid-name; the model leans on token frequency at this
data budget — each of 32k docs is seen ~0.03× in 1000 steps.)

## Benchmarks

Box: 8× Intel Haswell, no GPU. Bend native binary (`clang 21`).
The port has no parallel lets, so it runs **single-threaded**;
`--threads 8` gives identical output, no speedup (measured).

| Run | Work | Wall | Throughput |
|---|---|---|---|
| 1000 train steps + 20 samples | 13,902 train positions (×2 forward paths) + 400 sample positions, 4192 params | ~10.5 min (~0.64 s/step) | ≈ 22 forward-positions/s end-to-end (forward ×2 + backward sweep + Adam) |
| 5 train steps + 20 samples | ~450 positions | 5.8 s (`--threads 1`) / 8.4 s (`--threads 8`) | same code path, threading irrelevant |

Losses print once at the end (one `IO.print`), so there is no per-step
telemetry — the curve above comes from the final list.

## Layout

| File | What |
|---|---|
| `tape.bend` | Scalar-tape autograd: `TS`/`TP` state monad, 8 ops (add/mul/pow/log/exp/relu/leaf/const), array-backed tape (2^18) + grads + params + Adam buffers |
| `bwd.bend` | Reverse-mode sweep: Bool dispatch chain, descending index list |
| `model.bend` | Transformer forward on the tape: embeddings, 4-head attention with KV cache (heads unrolled), MLP, rmsnorm, softmax, per-doc loss |
| `data.bend` | `File.read` input, line split, sorted-unique vocab, char→id encoder, BOS-wrapped docs |
| `train.bend` | Adam stages, whole-run TP fuel loop, argmax + Gumbel-max samplers, decode, IO main |
| `rng.bend`, `nn.bend` | Port milestones 1–2 (LCG/Box-Muller PRNG, list-level dot/linear/rmsnorm/softmax) |
| `test_*.bend` | Gradient check (`4 2` on x·y+x), forward smoke, data smoke |

## Run it

```bash
export PATH="$HOME/.bend/bin:$PATH"
bend microgpt/train.bend -o train   # needs clang 14+; ~30 s build
./train                              # ~11 min for the full 1000 + 20
```

`bend microgpt/train.bend` (JS backend) type-checks but cannot execute:
228KB of line recursion overflows its stack. The native binary has no
such limit (flat state machine, no C stack) and is the only way to run.

## How it works (the one idea)

Microgpt's `Value` graph mutates shared nodes in place — impossible in
affine Bend. So the tape is explicit: forward appends `Node` records to
an `Array`, backward sweeps top-down accumulating into grad arrays, all
sequenced through a state monad (`TP`, same shape as `IO`) with
`do`-notation. Activations are bare `U32` node ids; data is re-read,
never carried.

## Deliberate differences from the gist

- Docs capped at 10 tokens (tape budget at depth 2^18); file order, no shuffle.
- Forward runs **twice** per position (logits path + cache path). Bend
  cannot split one computed `Step` and recurse on the pieces, and the
  loop must recurse — so affinity forces the 2× cost.
- Sampling is Gumbel-max with hash-derived noise (stateless, reproducible,
  T=0.5) instead of `random.choices`. Greedy argmax is also present.
- Single-doc batches, lr schedule over the step budget — as in the gist.

## Bend notes (earned, not guessed)

- Callee must precede caller textually; no forward references, no mutual recursion.
- `match` scrutinizes params/fields only — never computed values or let/do binds.
- Reuse needs `+` (params, case fields); `Array` is `Type`, never `+`.
  Affinity is whole-body: a value used in two `case` branches needs `+`.
- Loops are single self-recursive defs over a shrinking first arg; helpers
  that split binds must be acyclic (a split-then-recurse cycle is unorderable).
- Matching a `+` scrutinee hands out reusable fields.
- `+List` cannot take a pair element type — use a named record.
- `do M<..>:` headers take `<>`, signatures take `()`; `IO` is a `def`.
- Literal-left `/` with a call on the right misparses — use `F32.div`.
- `is` is a keyword; `Nat` patterns are `0n` / `1n+p`; `Char.cmp`
  returns the pair *and* the result.
- Pure `main` normalizes but never executes foreign ops — effectful tests
  need `IO` mains. File paths must be absolute (backend CWD differs).
