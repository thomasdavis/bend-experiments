import Base
import ./tape.bend as T
import ./bwd.bend as B
import ./model.bend as M
import ./gten.bend as G
import ./tfwd.bend as F
import ./tsweep.bend as S

# M2'b3 parity: one position [3,0], V=4, phash params.
# TP engine pgrads vs tensor sweep pgrads => print max abs diff.

def pset.go(+ix: +List<U32>, +ps: +List<F32>) -> T.TP<Unit>:
  match ix ps:
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

def dump.go(+ix: +List<U32>) -> T.TP<+List<F32>>:
  match ix:
    case Nil{}:
      do T.TP<+List<F32>>:
        return Nil{}
    case Con{i, t}:
      do T.TP<+List<F32>>:
        x : F32 <- T.pgget(i)
        rest : +List<F32> <- dump.go(t)
        return Con{x, rest}

def dump() -> T.TP<+List<F32>>:
  dump.go(B.bwd.n2u.go(List.range(U32.to_nat(M.base.np(4)))))

def mdiff.go(+a: +List<F32>, +b: +List<F32>, +m: F32) -> F32:
  match a b:
    case Nil{} Nil{}:
      m
    case Con{x, xt} Con{y, yt}:
      mdiff.go(xt, yt, F32.max(m, F32.abs((x - y : F32))))
    case _ _:
      m

def mdiff(+a: +List<F32>, +b: +List<F32>) -> F32:
  mdiff.go(a, b, {0.0 : F32})

def seedtop(+tp: +List<F.TN>, +sd: +List<F32>) -> +List<+List<F32>>:
  match tp:
    case Nil{}:
      Nil{}
    case Con{nd, t}:
      Con{sd, S.seed.gs(t)}

def tensorPG.sw(w: S.SW2) -> +List<F32>:
  match w:
    case S.SW2{gs, pg}:
      pg

def tensorPG.go(hb: F.HB, +params: +List<F32>) -> +List<F32>:
  match hb:
    case F.HB{out, ac}:
      match out:
        case F.FV3{lg, +tp, nx, id}:
           tensorPG.sw(S.sweep.go(tp, S.SW2{seedtop(tp, S.seed(G.softmax(lg), 0)), G.zeros(U32.to_nat(M.base.np(4)))}, params))

def tensorPG(+params: +List<F32>) -> +List<F32>:
  tensorPG.go(F.gstep(3, 0n, 4, params, F.pc30(), Nil{}, 0), params)

 def afterTP(r: (U32 & (Nat & M.AC))) -> T.TP<F32>:
   match r:
     case (sum, (n, ac)):
       do T.TP<F32>:
         u0 : Unit <- T.gset(sum, {1.0 : F32})
         cnt : U32 <- T.tcount()
         u1 : Unit <- B.bwd.go(B.bwd.ids(U32.to_nat(cnt)))
         tp : +List<F32> <- dump()
         return mdiff(tp, tensorPG(G.pinit.go(U32.to_nat(M.base.np(4)), 0)))

def prog() -> T.TP<F32>:
  do T.TP<F32>:
    u0 : Unit <- pset.go(B.bwd.n2u.go(List.range(U32.to_nat(M.base.np(4)))), G.pinit.go(U32.to_nat(M.base.np(4)), 0))
    z : U32 <- T.vconst({0.0 : F32})
    r : (U32 & (Nat & M.AC)) <- M.loss.go([{3 : U32}, {0 : U32}], 0n, 4, M.ac0(), z, 0n)
    afterTP(r)
 def go(p: T.TS & F32) -> IO(Unit):
   (st, d) = p
   do IO<Unit>:
     IO.print(F32.show(d))

 def main() -> IO(Unit):
   go(T.TP.run(F32, prog(), T.ts_new()))


