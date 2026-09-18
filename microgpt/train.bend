import Base
import ./tape.bend as T
import ./bwd.bend as B
import ./rng.bend as R
import ./model.bend as M
import ./data.bend as D

# Training + sampling for the microgpt port.
# One TP run does everything: init params, fuel-looped train steps
# (tfresh -> forward x2 -> seed -> sweep -> Adam), argmax samples.
# Loops are single self-recursive defs; splits happen only in
# acyclic helpers. Callee-before-caller throughout.

 def adam.mn(+m: F32, +g: F32, +b1: F32) -> F32:
   ((b1 * m + ((1.0 - b1 : F32)) * g : F32))

 def adam.vn(+v: F32, +g: F32, +b2: F32) -> F32:
   ((b2 * v + ((1.0 - b2 : F32)) * g * g : F32))

 def adam.mh(+m: F32, +g: F32, +b1: F32, +t1: F32) -> F32:
   (adam.mn(m, g, b1) / ((1.0 - F32.pow(b1, t1) : F32)) : F32)

 def adam.vh(+v: F32, +g: F32, +b2: F32, +t1: F32) -> F32:
   (adam.vn(v, g, b2) / ((1.0 - F32.pow(b2, t1) : F32)) : F32)

 def adam.dl(+m: F32, +g: F32, +v: F32, +b1: F32, +b2: F32, +t1: F32, +lr: F32, +eps: F32) -> F32:
   ((lr * adam.mh(m, g, b1, t1) / (F32.sqrt(adam.vh(v, g, b2, t1)) + eps : F32) : F32))

 def adam.up(+p: F32, +g: F32, +m: F32, +v: F32, +i: U32, +t1: F32, +lr: F32, +b1: F32, +b2: F32, +eps: F32) -> T.TP<Unit>:
   do T.TP<Unit>:
     u0 : Unit <- T.mset(i, adam.mn(m, g, b1))
     u1 : Unit <- T.vset(i, adam.vn(v, g, b2))
     u2 : Unit <- T.pset(i, ((p - adam.dl(m, g, v, b1, b2, t1, lr, eps) : F32)))
     T.pgset(i, {0.0 : F32})

 def adam.one(+i: U32, +step: Nat, +lr: F32, +b1: F32, +b2: F32, +eps: F32) -> T.TP<Unit>:
   do T.TP<Unit>:
     p : F32 <- T.pget(i)
     g : F32 <- T.pgget(i)
     m : F32 <- T.mget(i)
     v : F32 <- T.vget(i)
     t1 : F32 = (F32.from_nat(step) + 1.0 : F32)
     adam.up(p, g, m, v, i, t1, lr, b1, b2, eps)

def adam.go(+ix: +List<U32>, +step: Nat, +lr: F32, +b1: F32, +b2: F32, +eps: F32) -> T.TP<Unit>:
  match ix:
    case Nil{}:
      do T.TP<Unit>:
        return Unit{}
    case Con{i, t}:
      do T.TP<Unit>:
        u : Unit <- adam.one(i, step, lr, b1, b2, eps)
        adam.go(t, step, lr, b1, b2, eps)

def train.pick(+ds: +List<+List<U32>>, n: Nat) -> +List<U32>:
  match ds n:
    case Nil{} _:
      Nil{}
    case Con{d, t} 0n:
      d
    case Con{d, t} 1n+p:
      train.pick(t, p)

 def train.cont(+mn: U32, +V: U32, +step: Nat, +num: Nat, +lr0: F32) -> T.TP<F32>:
   do T.TP<F32>:
     u0 : Unit <- T.gset(mn, {1.0 : F32})
     cnt : U32 <- T.tcount()
     u1 : Unit <- B.bwd.go(B.bwd.ids(U32.to_nat(cnt)))
     u2 : Unit <- adam.go(B.bwd.n2u.go(List.range(U32.to_nat(M.base.np(V)))), step, ((lr0 * ((1.0 - F32.div(F32.from_nat(step), F32.from_nat(num)) : F32)) : F32)), {0.85 : F32}, {0.99 : F32}, {0.00000001 : F32})
     nd : T.Node <- T.tget(mn)
     return T.ndata(nd)

 def train.full(+sum: U32, n: Nat, +V: U32, +step: Nat, +num: Nat, +lr0: F32) -> T.TP<F32>:
   match n:
     case 0n:
       do T.TP<F32>:
         nd : T.Node <- T.tget(sum)
         return T.ndata(nd)
     case 1n+m:
       do T.TP<F32>:
         invc : U32 <- T.vconst(F32.div({1.0 : F32}, F32.from_nat(1n + m)))
         mn : U32 <- T.vmul(sum, invc)
         train.cont(mn, V, step, num, lr0)
def train.after(r: (U32 & (Nat & M.AC)), +V: U32, +step: Nat, +num: Nat, +lr0: F32) -> T.TP<F32>:
  match r:
    case (sum, (n, ac)):
      train.full(sum, n, V, step, num, lr0)

def train.step(+doc: +List<U32>, +V: U32, +step: Nat, +num: Nat, +lr0: F32) -> T.TP<F32>:
  do T.TP<F32>:
    u : Unit <- T.tfresh()
    z : U32 <- T.vconst({0.0 : F32})
    r : (U32 & (Nat & M.AC)) <- M.loss.go(doc, 0n, V, M.ac0(), z, 0n)
    train.after(r, V, step, num, lr0)

def train.go(fuel: Nat, +step: Nat, +docs: +List<+List<U32>>, +V: U32, +num: Nat, +lr0: F32, +acc: +List<F32>) -> T.TP<+List<F32>>:
  match fuel:
    case 0n:
      do T.TP<+List<F32>>:
        return acc
    case 1n+f:
      do T.TP<+List<F32>>:
         l : F32 <- train.step(train.pick(docs, (step - step / List.length(&2, +List<U32>, docs) * List.length(&2, +List<U32>, docs) : Nat)), V, step, num, lr0)
         train.go(f, 1n + step, docs, V, num, lr0, Con{l, acc})
 type IDV is Data:
   IDV{id: U32, val: F32}

 def pairup.mk(xn: T.Node, +pos: U32, +rest: +List<IDV>) -> T.TP<+List<IDV>>:
   match xn:
     case T.N{o, a, b, k, xd}:
       do T.TP<+List<IDV>>:
         return Con{IDV{pos, xd}, rest}

 def pairup.go(+xs: +List<U32>, +pos: U32) -> T.TP<+List<IDV>>:
   match xs:
     case Nil{}:
       do T.TP<+List<IDV>>:
         return Nil{}
     case Con{x, xt}:
       do T.TP<+List<IDV>>:
         xn : T.Node <- T.tget(x)
         rest : +List<IDV> <- pairup.go(xt, (pos + 1 : U32))
         pairup.mk(xn, pos, rest)

 def pairup(+xs: +List<U32>) -> T.TP<+List<IDV>>:
   pairup.go(xs, 0)

 def argmax.go(+xs: +List<IDV>, +bi: U32, +bd: F32) -> U32:
   match xs:
     case Nil{}:
       bi
     case Con{pr, t}:
       match pr:
         case IDV{i, d}:
           argmax.go(t, Bool.pick(U32, F32.is_gt(d, bd), i, bi), F32.max(d, bd))

 def ghash(+smp: U32, +pos: Nat, +i: U32) -> F32:
   F32.max((U32.to_f32(U32.and((1103515245 * (smp + 1000003 * U32.from_nat(pos) + 9176 * i + 7) + 12345 : U32), 2147483647)) / 2147483648.0 : F32), {0.00000001 : F32})

 def ggum(+u: F32) -> F32:
   F32.neg(F32.log(F32.neg(F32.log(u))))

 def gpert.one(pr: IDV, +smp: U32, +pos: Nat) -> IDV:
   match pr:
     case IDV{+i, d}:
       IDV{i, F32.add(F32.log(d), ggum(ghash(smp, pos, i)))}

 def gpert.go(+ps: +List<IDV>, +smp: U32, +pos: Nat) -> +List<IDV>:
   match ps:
     case Nil{}:
       Nil{}
     case Con{pr, t}:
       Con{gpert.one(pr, smp, pos), gpert.go(t, smp, pos)}

 def samp.best(+pr: +List<IDV>) -> U32:
   argmax.go(pr, 0, ((0.0 - 1000000.0 : F32)))

 def samp.next(+lg: +List<U32>, +smp: U32, +pos: Nat) -> T.TP<U32>:
   do T.TP<U32>:
     pr : +List<IDV> <- pairup(lg)
     return samp.best(gpert.go(pr, smp, pos))
 def samp.go(fuel: Nat, +pos: Nat, +V: U32, +cache: M.AC, +acc: +List<U32>, +smp: U32) -> T.TP<+List<U32>>:
   match fuel:
     case 0n:
       do T.TP<+List<U32>>:
         return acc
     case 1n+f:
       match acc:
         case Nil{}:
           do T.TP<+List<U32>>:
             return acc
         case Con{+tok, rest}:
           do T.TP<+List<U32>>:
             lg : +List<U32> <- M.step.logits(tok, pos, V, cache)
             c2 : M.AC <- M.step.cache(tok, pos, V, cache)
             nt : U32 <- samp.next(lg, smp, pos)
             samp.go(f, 1n + pos, V, c2, Con{nt, acc}, smp)

 def samp.all(k: Nat, +V: U32, +j: U32) -> T.TP<+List<+List<U32>>>:
   match k:
     case 0n:
       do T.TP<+List<+List<U32>>>:
         return Nil{}
     case 1n+f:
       do T.TP<+List<+List<U32>>>:
         s : +List<U32> <- samp.go(10n, 0n, V, M.ac0(), Con{(V - 1 : U32), Nil{}}, j)
         rest : +List<+List<U32>> <- samp.all(f, V, (j + 1 : U32))
         return Con{s, rest}

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

 def dec.nth(+us: +List<Char>, n: Nat) -> Char:
   match us n:
     case Nil{} _:
       '?'
     case Con{u, t} 0n:
       u
     case Con{u, t} 1n+p:
       dec.nth(t, p)

 def dec.go(+ts: +List<U32>, +us: +List<Char>) -> String:
   match ts:
     case Nil{}:
       ""
     case Con{t, r}:
       SCon{dec.nth(us, U32.to_nat(t)), dec.go(r, us)}

 def dec.all(+ss: +List<+List<U32>>, +us: +List<Char>) -> +List<String>:
   match ss:
     case Nil{}:
       Nil{}
     case Con{s, t}:
       Con{dec.go(List.reverse(&2, U32, s), us), dec.all(t, us)}

 def showL.go(+xs: +List<F32>) -> String:
   match xs:
     case Nil{}:
       ""
     case Con{h, t}:
       F32.show(h) ++ " " ++ showL.go(t)

 def showL(+xs: +List<F32>) -> String:
   showL.go(List.reverse(&2, F32, xs))

 def showU.go(+xs: +List<U32>) -> String:
   match xs:
     case Nil{}:
       ""
     case Con{h, t}:
       U32.show(h) ++ " " ++ showU.go(t)

 def showU(+xs: +List<U32>) -> String:
   showU.go(List.reverse(&2, U32, xs))

 def printAll(+ss: +List<String>) -> IO(Unit):
   match ss:
     case Nil{}:
       do IO<Unit>:
         return Unit{}
     case Con{h, t}:
       do IO<Unit>:
         u : Unit <- IO.print(h)
         printAll(t)

 def out.ss(+ss: +List<+List<U32>>, +us: +List<Char>) -> IO(Unit):
   match ss:
     case Nil{}:
       do IO<Unit>:
         return Unit{}
     case Con{s, t}:
       do IO<Unit>:
         u : Unit <- IO.print(showU(s))
         printAll(dec.all(ss, us))
 def out(p: T.TS & (+List<F32> & +List<+List<U32>>), +us: +List<Char>) -> IO(Unit):
   (st, (ls, ss)) = p
   do IO<Unit>:
     u : Unit <- IO.print(showL(ls))
     printAll(dec.all(ss, us))

def prog(+docs: +List<+List<U32>>, +V: U32) -> T.TP<(+List<F32> & +List<+List<U32>>)>:
  do T.TP<(+List<F32> & +List<+List<U32>>)>:
    iu : U32 <- init.params(V)
    ls : +List<F32> <- train.go(1000n, 0n, docs, V, 1000n, {0.01 : F32}, Nil{})
    ss : +List<+List<U32>> <- samp.all(20n, V, 0)
    return (ls, ss)

def prep.docs(+docs: +List<+List<U32>>, +us: +List<Char>, +V: U32) -> IO(Unit):
  do IO<Unit>:
     r : (T.TS & (+List<F32> & +List<+List<U32>>)) = T.TP.run((+List<F32> & +List<+List<U32>>), prog(docs, V), T.ts_new())
    out(r, us)

def prep.vocab(+us: +List<Char>, +ls: +List<String>) -> IO(Unit):
  prep.docs(D.enc.docs(ls, us, (D.vof(us) - 1 : U32)), us, D.vof(us))

def prep.go(+ls: +List<String>) -> IO(Unit):
  prep.vocab(D.vocab(ls), ls)

def prep(s: String) -> IO(Unit):
  prep.go(D.lines(s))

def main() -> IO(Unit):
  do IO<Unit>:
    s : String <- D.readInput("/home/ajax/tmp/bend-experiments/microgpt/input.txt")
    prep(s)
