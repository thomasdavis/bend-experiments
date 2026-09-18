import Base
import ./gten.bend as G
import ./tfwd.bend as F
import ./tsweep.bend as S

# M3: minibatch driver (sequential map; `!` comes after CPU proof).
# Per-item forward+sweep are independent pure values sharing read-only
# params -- the shape `!` needs. Records everywhere; one self-recursive
# driver over DS states; splits only in acyclic helpers.

type BT is Data:
  BT{loss: F32, pg: +List<F32>}

type BR is Data:
  BR{loss: F32, pg: +List<F32>}

type DS is Data:
  DS{params: +List<F32>, m: +List<F32>, v: +List<F32>, step: Nat, docs: +List<+List<U32>>, V: U32, pn: Nat, lr: F32, losses: +List<F32>}

def seedNils(+tp: +List<F.TN>) -> +List<+List<F32>>:
  match tp:
    case Nil{}:
      Nil{}
    case Con{nd, t}:
      Con{Nil{}, seedNils(t)}

 def bitem.si(+sds: +List<F.SD>, +gs: +List<+List<F32>>) -> +List<+List<F32>>:
   match sds:
     case Nil{}:
       gs
     case Con{s, t}:
       match s:
         case F.SD{id, tgt, lg}:
           bitem.si(t, S.paddcvec(gs, U32.to_nat(id), S.seed(G.softmax(lg), tgt)))

 def bitem.pgof(w: S.SW2) -> +List<F32>:
   match w:
     case S.SW2{gs, pg}:
       pg
 def bitem.go(gs: F.GS3, +params: +List<F32>, +pn: Nat) -> BT:
   match gs:
     case F.GS3{toks, pos, cache, acc, n, +tp, nx, V, params2, seeds}:
       BT{acc, bitem.pgof(S.sweep.go(tp, S.SW2{bitem.si(seeds, seedNils(tp)), G.zeros(pn)}, params))}

def bitem(+doc: +List<U32>, +V: U32, +params: +List<F32>, +pn: Nat) -> BT:
  bitem.go(F.tloss.entry(doc, V, params), params, pn)
 def bred.go(+bs: +List<BT>, +ls: F32, +pg: +List<F32>) -> BR:
   match bs:
     case Nil{}:
       BR{ls, pg}
     case Con{b, t}:
       match b:
         case BT{loss, pgb}:
           bred.go(t, (ls + loss : F32), G.tadd.zip(pg, pgb))

 def bred(+bs: +List<BT>, +pn: Nat) -> BR:
   bred.go(bs, {0.0 : F32}, G.zeros(pn))

 def bmap.go(+xs: +List<+List<U32>>, +V: U32, +params: +List<F32>, +pn: Nat) -> +List<BT>:
   match xs:
     case Nil{}:
       Nil{}
     case Con{d, t}:
       Con{bitem(d, V, params, pn), bmap.go(t, V, params, pn)}

 type DT is Data:
   DN{}
   DLF{doc: +List<U32>}
   DBR{l: DT, r: DT}

 def dleaf(+xs: +List<+List<U32>>) -> +List<DT>:
   match xs:
     case Nil{}:
       Nil{}
     case Con{d, t}:
       Con{DLF{d}, dleaf(t)}

 def dpair(+xs: +List<DT>) -> +List<DT>:
   match xs:
     case Nil{}:
       Nil{}
     case Con{a, Nil{}}:
       Con{a, Nil{}}
     case Con{a, Con{b, t}}:
       Con{DBR{a, b}, dpair(t)}

 def dlevel(fuel: Nat, +ts: +List<DT>) -> DT:
   match fuel ts:
     case 0n _:
       DN{}
     case 1n+f Nil{}:
       DN{}
     case 1n+f Con{t, Nil{}}:
       t
     case 1n+f Con{a, t2}:
       dlevel(f, dpair(ts))

 def dbuild(+xs: +List<+List<U32>>) -> DT:
   dlevel(List.length(&2, +List<U32>, xs), dleaf(xs))

 def bmap.join(+a: +List<BT>, +b: +List<BT>) -> +List<BT>:
   List.append(&2, BT, a, b)

 def bmap.tree(dt: DT, +V: U32, +params: +List<F32>, +pn: Nat) -> +List<BT>:
   match dt:
     case DN{}:
       Nil{}
     case DLF{d}:
       Con{bitem(d, V, params, pn), Nil{}}
     case DBR{l, r}:
       a b = bmap.tree(l, V, params, pn) bmap.tree(r, V, params, pn)
       bmap.join(a, b)
 def bstep.adm(a: (+List<F32> & +List<F32>) & +List<F32>, +step: Nat, +docs: +List<+List<U32>>, +V: U32, +pn: Nat, +lr: F32, +losses: +List<F32>, +loss: F32) -> DS:
   match a:
     case ((pp, mp), vp):
       DS{pp, mp, vp, 1n + step, docs, V, pn, lr, Con{loss, losses}}

 def bstep.red(r: BR, +params: +List<F32>, +m: +List<F32>, +v: +List<F32>, +step: Nat, +docs: +List<+List<U32>>, +V: U32, +pn: Nat, +lr: F32, +losses: +List<F32>) -> DS:
   match r:
     case BR{loss, pg}:
       bstep.adm(S.adam.go(params, pg, m, v, (F32.from_nat(step) + 1.0 : F32), lr, {0.85 : F32}, {0.99 : F32}, {0.00000001 : F32}), step, docs, V, pn, lr, losses, loss)

 def bstep(+docs: +List<+List<U32>>, +V: U32, +params: +List<F32>, +m: +List<F32>, +v: +List<F32>, +step: Nat, +lr: F32, +pn: Nat, +losses: +List<F32>) -> DS:
   bstep.red(bred(bmap.tree!(dbuild(docs), V, params, pn), pn), params, m, v, step, docs, V, pn, lr, losses)

 def bdriver(fuel: Nat, st: DS) -> DS:
   match fuel:
     case 0n:
       st
     case 1n+f:
       match st:
         case DS{params, m, v, step, docs, V, pn, lr, losses}:
           bdriver(f, bstep(docs, V, params, m, v, step, lr, pn, losses))

 def showl(+xs: +List<F32>) -> String:
   match xs:
     case Nil{}:
       ""
     case Con{h, t}:
       F32.show(h) ++ " " ++ showl(t)

 def out(st: DS) -> IO(Unit):
   match st:
     case DS{params, m, v, step, docs, V, pn, lr, losses}:
       do IO<Unit>:
         IO.print(showl(losses))

 def main() -> IO(Unit):
   out(bdriver(2n, DS{G.zeros(U32.to_nat(3456)), G.zeros(U32.to_nat(3456)), G.zeros(U32.to_nat(3456)), 0n, Con{Con{3, Con{0, Nil{}}}, Con{Con{1, Con{3, Nil{}}}, Nil{}}}, 4, U32.to_nat(3456), {0.01 : F32}, Nil{}}))
