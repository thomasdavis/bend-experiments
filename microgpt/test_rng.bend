import Base
import ./rng.bend as R

# Chain five draws; samples accumulate in a nested pair (same
# pass-the-pair idiom as rng.bend: no destructuring of computed values).
# main returns IO so the program RUNS on the backend instead of merely
# normalizing (a pure main never executes foreign F32 ops).
def t5(pp: (U32 & F32) & ((((F32 & F32) & F32) & F32))) -> IO(Unit):
  ((s5, w), (((g, a), b), c)) = pp
  do IO<Unit>:
    IO.print(F32.show(a) ++ " " ++ F32.show(b) ++ " " ++ F32.show(c) ++ " " ++ F32.show(g) ++ " " ++ F32.show(w))

def t4(pp: (U32 & F32) & (((F32 & F32) & F32))) -> IO(Unit):
  ((s4, g), ((a, b), c)) = pp
  t5((R.rng_weight(s4), (((g, a), b), c)))

def t3(pp: (U32 & F32) & ((F32 & F32))) -> IO(Unit):
  ((s3, c), (a, b)) = pp
  t4((R.rng_gauss(s3), ((a, b), c)))

def t2(pp: (U32 & F32) & F32) -> IO(Unit):
  ((s2, b), a) = pp
  t3((R.rng_uniform(s2), (a, b)))

def t1(p: U32 & F32) -> IO(Unit):
  (s1, a) = p
  t2((R.rng_uniform(s1), a))

def main() -> IO(Unit):
  t1(R.rng_uniform(42))
