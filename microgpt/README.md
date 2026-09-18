# microgpt in Bend

A line-for-line-shaped port of Karpathy's `microgpt.py` (pure-Python GPT:
tokenizer, scalar autograd, 1-layer transformer, Adam, sampling) to
[Bend](https://bend-lang.com), a proof-carrying parallel language.

## Layout

| File | What |
|---|---|
| `tape.bend` | Scalar-tape autograd: `TS`/`TP` state monad, 8 ops, array-backed tape + grads + params + Adam buffers |
| `bwd.bend` | Reverse-mode sweep (Bool dispatch chain, descending index list) |
| `model.bend` | Transformer forward on the tape: embed, 4-head attention + KV cache, MLP, rmsnorm, softmax, loss |
| `data.bend` | File IO, line split, sorted-unique vocab, char→id encoder |
| `train.bend` | Adam, whole-run TP fuel loop, Gumbel-max sampler, IO main |
| `rng.bend`, `nn.bend` | Port milestones 1–2 (functional PRNG, list-level NN ops) |
| `test_*.bend` | Gradient check (`4 2`), forward smoke (loss ≈ 3.95 @ V=4), data smoke |

## Run it

```bash
export PATH="$HOME/.bend/bin:$PATH"
bend microgpt/train.bend -o train   # needs clang; runs on CPU
./train                              # ~11 min: 1000 steps + 20 samples
```

`bend microgpt/train.bend` (JS backend) type-checks but cannot run the
pipeline: 228KB of line recursion overflows its stack. The native binary
has no such limit (flat state machine) and trains ~0.6 s/step.

## Result (2026-09-18)

1000 steps, same hyperparams as the gist (1 layer, 16-wide, 4 heads,
block 16, lr 0.01, Adam 0.85/0.99). Mean per-doc loss by block:
2.64 → 2.35 → 2.27 → 2.16 (random init ln 28 ≈ 3.33).

## Bend notes (earned, not guessed)

- Callee must precede caller textually; no forward references, no mutual recursion.
- `match` scrutinizes params/fields only — never computed values or let/do binds.
- Reuse needs `+` (params, case fields); `Array` is `Type`, never `+`.
- Loops are single self-recursive defs over a shrinking first arg; helpers
  that split binds must be acyclic (or the cycle is unorderable).
- `+List` cannot take a pair element type — use a named record.
- `do M<..>:` headers take `<>`, signatures take `()`; `IO` is a `def`.
- Literal-left `/` with a call on the right misparses — use `F32.div`.
- `is` is a keyword; `Nat` patterns are `0n` / `1n+p`.
- Pure `main` normalizes but never executes foreign ops — effectful tests
  need `IO` mains.

## Deliberate differences from the gist

- Docs capped at 10 tokens (tape budget at depth 2^18); file order, no shuffle.
- Forward runs twice per position (logits + cache paths) — Bend cannot
  split one computed `Step` and recurse, so affinity forces the 2× cost.
- Sampling is Gumbel-max with hash-derived noise (stateless, reproducible)
  at T=0.5 instead of `random.choices`.
