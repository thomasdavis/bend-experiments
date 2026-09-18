import Base

# Functional PRNG for microgpt port. Base has no RNG, so we thread a U32
# state explicitly: every draw returns (new_state, sample).
# Uniformity source is an LCG (Numerical Recipes constants); U32 arithmetic wraps.
#
# Bend note: a `match`/`let` can only scrutinize a parameter, never a computed
# value — so every pair consumer takes the whole pair as one parameter
# (the same shape as Base's String.cmp / String.eq.fin).

def rng_next(s: U32) -> U32 & U32:
  +ns = (1664525 * s + 1013904223 : U32)
  (ns, ns)

# Uniform F32 in [0, 1).
def rng_uniform.f(p: U32 & U32) -> U32 & F32:
  (ns, r) = p
  (ns, (U32.to_f32(r) / 4294967296.0 : F32))

def rng_uniform(s: U32) -> U32 & F32:
  rng_uniform.f(rng_next(s))

# Standard-normal F32 via Box-Muller (compare: random.gauss(0, 1)).
# Second deviate is discarded to keep the state single-threaded.
def rng_gauss.f2(pp: (U32 & F32) & F32) -> U32 & F32:
  ((s2, u2), u1) = pp
  +u1c = F32.max(u1, 0.00000001)
  l = F32.log(u1c)
  r = (0.0 - 2.0 * l : F32)
  mag = F32.sqrt(r)
  ang = (6.2831853 * u2 : F32)
  g = (mag * F32.cos(ang) : F32)
  (s2, g)

def rng_gauss.f1(p: U32 & F32) -> U32 & F32:
  (s1, u1) = p
  rng_gauss.f2((rng_uniform(s1), u1))

def rng_gauss(s: U32) -> U32 & F32:
  rng_gauss.f1(rng_uniform(s))

# microgpt inits every weight with random.gauss(0, 0.08).
def rng_weight.f(p: U32 & F32) -> U32 & F32:
  (ns, g) = p
  (ns, (0.08 * g : F32))

def rng_weight(s: U32) -> U32 & F32:
  rng_weight.f(rng_gauss(s))
