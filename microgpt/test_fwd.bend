import Base
import ./tape.bend as T
import ./bwd.bend as B
import ./rng.bend as R
import ./model.bend as M

# Forward smoke test: V=4, random-ish params, tokens [BOS,a,b,BOS].
# Exercises embed -> block -> logits -> loss end to end.

 def init.wk(p: U32 & F32, +i: U32) -> T.TP<U32>:
   (st2, w) = p
   do T.TP<U32>:
     u : Unit <- T.pset(i, w)
     return st2

 def init.go(+ids: +List<U32>, st: U32) -> T.TP<U32>:
   match ids:
     case Nil{}:
       do T.TP<U32>:
         return st
     case Con{i, t}:
       do T.TP<U32>:
         st2 : U32 <- init.wk(R.rng_weight(st), i)
         init.go(t, st2)

def init.params(+V: U32) -> T.TP<U32>:
  init.go(B.bwd.n2u.go(List.range(U32.to_nat(M.base.np(V)))), 42)

def showD(nd: T.Node) -> F32:
  match nd:
    case T.N{o, a, b, k, d}:
      d

def after(r: (U32 & (Nat & M.AC))) -> T.TP<F32>:
  match r:
    case (sum, (n, ac)):
      do T.TP<F32>:
        nd : T.Node <- T.tget(sum)
        return showD(nd)

def prog() -> T.TP<F32>:
  do T.TP<F32>:
    st : U32 <- init.params(4)
    z : U32 <- T.vconst({0.0 : F32})
    r : (U32 & (Nat & M.AC)) <- M.loss.go([3, 0, 1, 3], 0n, 4, M.ac0(), z, 0n)
    after(r)

def go(p: T.TS & F32) -> IO(Unit):
  (st, v) = p
  do IO<Unit>:
    IO.print(F32.show(v))

def main() -> IO(Unit):
  go(T.TP.run(F32, prog(), T.ts_new()))
