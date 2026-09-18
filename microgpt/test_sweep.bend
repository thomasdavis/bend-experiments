import Base
import ./ptape.bend as P

# f = x*y + x at x=2, y=3: expect param grads (4, 2).
# Tape hand-built newest-first: add(2,0), mul(0,1), leaf(p1), leaf(p0).

def tape() -> +List<P.PN>:
  Con{P.PN{1, 2, 0, {0.0 : F32}, {8.0 : F32}}, Con{P.PN{2, 0, 1, {0.0 : F32}, {6.0 : F32}}, Con{P.PN{0, 1, 0, {0.0 : F32}, {3.0 : F32}}, Con{P.PN{0, 0, 0, {0.0 : F32}, {2.0 : F32}}, Nil{}}}}}

def seedgs() -> +List<F32>:
  Con{{1.0 : F32}, Con{{0.0 : F32}, Con{{0.0 : F32}, Con{{0.0 : F32}, Nil{}}}}}

def go(st: P.SW) -> IO(Unit):
  match st:
    case P.SW{gs, p0, p1}:
      do IO<Unit>:
        IO.print(F32.show(p0) ++ " " ++ F32.show(p1))

def main() -> IO(Unit):
  go(P.sweep.go(tape(), P.SW{seedgs(), {0.0 : F32}, {0.0 : F32}}, tape()))
