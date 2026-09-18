import Base
import ./tape.bend as T
import ./bwd.bend as B
import ./model.bend as M
import ./gten.bend as G
import ./tfwd.bend as F
import ./tsweep.bend as S
import ./train.bend as TR

# M2'b4 parity: one full Adam step, TP engine vs explicit engine.
# Same phash params, doc [3,0] => print max abs diff of new params.

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
        x : F32 <- T.pget(i)
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

 def lastlog.go(nd: F.TN, +params: +List<F32>) -> +List<F32>:
   match nd:
     case F.TN{op, a, b, c, xs, ys, ids}:
       G.tmatvec(params, a, U32.to_nat(b), U32.to_nat(c), xs)

def lastlog(+tp: +List<F.TN>, +params: +List<F32>) -> +List<F32>:
  match tp:
    case Nil{}:
      Nil{}
    case Con{nd, t}:
      lastlog.go(nd, params)

def seedtop.go(+ns: +List<+List<F32>>, +sd: +List<F32>) -> +List<+List<F32>>:
  match ns:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{sd, t}

def seedtop(+tp: +List<F.TN>, +sd: +List<F32>) -> +List<+List<F32>>:
  seedtop.go(S.seed.gs(tp), sd)

def exAdam.go2(pm: +List<F32> & +List<F32>) -> +List<F32>:
  match pm:
    case (pp, mp):
      pp

def exAdam.go(p: (+List<F32> & +List<F32>) & +List<F32>) -> +List<F32>:
  match p:
    case (pm, vp):
      exAdam.go2(pm)

def exFin(w: S.SW2, +params: +List<F32>) -> +List<F32>:
  match w:
    case S.SW2{gs, pg}:
      exAdam.go(S.adam.go(params, pg, G.zeros(U32.to_nat(3456)), G.zeros(U32.to_nat(3456)), {1.0 : F32}, {0.01 : F32}, {0.85 : F32}, {0.99 : F32}, {0.00000001 : F32}))

def exSweep(+tp: +List<F.TN>, +params: +List<F32>) -> +List<F32>:
  exFin(S.sweep.go(tp, S.SW2{seedtop(tp, S.seed(G.softmax(lastlog(tp, params)), 0)), G.zeros(U32.to_nat(3456))}, params), params)

def exGo(gs: F.GS3, +params: +List<F32>) -> +List<F32>:
  match gs:
    case F.GS3{toks, pos, cache, acc, n, tp, nx, V, params2}:
      exSweep(tp, params)

def exSide() -> +List<F32>:
  exGo(F.tloss.entry([{3 : U32}, {0 : U32}], 4, G.pinit.go(U32.to_nat(M.base.np(4)), 0)), G.pinit.go(U32.to_nat(M.base.np(4)), 0))

def afterTP(r: (U32 & (Nat & M.AC))) -> T.TP<F32>:
  match r:
    case (sum, (n, ac)):
      do T.TP<F32>:
        u0 : Unit <- T.gset(sum, {1.0 : F32})
        cnt : U32 <- T.tcount()
        u1 : Unit <- B.bwd.go(B.bwd.ids(U32.to_nat(cnt)))
        u2 : Unit <- TR.adam.go(B.bwd.n2u.go(List.range(U32.to_nat(M.base.np(4)))), 0n, {0.01 : F32}, {0.85 : F32}, {0.99 : F32}, {0.00000001 : F32})
        tp : +List<F32> <- dump()
        return mdiff(tp, exSide())

def progTP() -> T.TP<F32>:
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
  go(T.TP.run(F32, progTP(), T.ts_new()))
