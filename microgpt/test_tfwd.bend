import Base
import ./gten.bend as G
import ./gfwd.bend as F
import ./tfwd.bend as T

# M2'b2 parity: tensor-recording loss vs pure M1 loss, zero params.
# Same values => expect exact match (4.158883 both).

def show2(+a: F32, +b: F32) -> IO(Unit):
  do IO<Unit>:
    IO.print(F32.show(a) ++ " " ++ F32.show(b))

def main.go2(+a: F32, p2: T.GS3) -> IO(Unit):
  match p2:
    case T.GS3{toks, pos, cache, acc, n, tp, nx, V, params}:
      show2(a, acc)

def main.go(p1: F32 & Nat, p2: T.GS3) -> IO(Unit):
  match p1:
    case (a, n1):
      main.go2(a, p2)

def main() -> IO(Unit):
  main.go(F.gloss.go([{3 : U32}, {0 : U32}, {1 : U32}, {3 : U32}], 0n, 4, G.zeros(3456n), F.pc0(), {0.0 : F32}, 0n), T.tloss.entry([{3 : U32}, {0 : U32}, {1 : U32}, {3 : U32}], 4, G.zeros(3456n)))
