import Base
import ./tape.bend as T
import ./bwd.bend as B
import ./model.bend as M
import ./gten.bend as G
import ./gfwd.bend as F

# M1 parity: scalar-tape loss vs pure-tensor loss on identical params.
# Both read the same phash table, same op order => expect exact match.

def pset.go(+ids: +List<U32>, +ps: +List<F32>) -> T.TP<Unit>:
  match ids ps:
    case Nil{} Nil{}:
      do T.TP<Unit>:
        return Unit{}
    case Con{i, t} Con{p, pt}:
      do T.TP<Unit>:
        u : Unit <- T.pset(i, p)
        pset.go(t, pt)
    case _ _:
      do T.TP<Unit>:
        return Unit{}

def after(r: (U32 & (Nat & M.AC))) -> T.TP<(F32 & (F32 & Nat))>:
  match r:
    case (sum, (n, ac)):
      do T.TP<(F32 & (F32 & Nat))>:
        nd : T.Node <- T.tget(sum)
        return (T.ndata(nd), F.gloss.go([3, 0, 1, 3], 0n, 4, G.zeros(U32.to_nat(M.base.np(4))), F.pc0(), {0.0 : F32}, 0n))

def prog() -> T.TP<(F32 & (F32 & Nat))>:
  do T.TP<(F32 & (F32 & Nat))>:
    u0 : Unit <- pset.go(B.bwd.n2u.go(List.range(U32.to_nat(M.base.np(4)))), G.zeros(U32.to_nat(M.base.np(4))))
    z : U32 <- T.vconst({0.0 : F32})
    r : (U32 & (Nat & M.AC)) <- M.loss.go([3, 0, 1, 3], 0n, 4, M.ac0(), z, 0n)
    after(r)

def go(p: T.TS & (F32 & (F32 & Nat))) -> IO(Unit):
  (st, (a, (b, n))) = p
  do IO<Unit>:
    IO.print(F32.show(a) ++ " " ++ F32.show(b))

def main() -> IO(Unit):
  go(T.TP.run((F32 & (F32 & Nat)), prog(), T.ts_new()))
