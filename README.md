# bend-experiments

Private Bend playground. Bend 2.0.5.

## Setup

```bash
curl -fsSL https://bend-lang.com/install.sh | sh
export PATH="$HOME/.bend/bin:$PATH"
bend guide   # the whole language
bend base    # stdlib reference
```

## Workflow (per AGENTS.md)

```bash
bend hello.bend    # run
bend pow2.bend     # parallel demo (1024)
bend PROOF.bend    # gate: must print "All terms check." before commit
```

Laws live in `LAWS.bend` (human-owned, AI does not touch).
Proofs + code live in `PROOF.bend` + `*.bend` (AI-written).
Breaking a law must fail `bend PROOF.bend`; no merge until green.
