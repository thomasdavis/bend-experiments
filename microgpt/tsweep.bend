import Base
import ./gten.bend as G
import ./tfwd.bend as T

# M2'b3: tensor-level reverse sweep (pure, no arrays).
# Tape newest-first + gradvecs aligned alongside (matched together);
# pgrads dense +List<F32> with O(n) scatters. State packs in SW2
# (records, never pairs). VJPs recompute what they need from stored
# vectors + params (no extra storage).

type SW2 is Data:
  SW2{gs: +List<+List<F32>>, pg: +List<F32>}

def paddcvec.go(+gs: +List<+List<F32>>, k: Nat, +dv: +List<F32>) -> +List<+List<F32>>:
  match gs k:
    case Nil{} _:
      Nil{}
    case Con{g, t} 0n:
      Con{G.tadd.zip(g, dv), t}
    case Con{g, t} 1n+p:
      Con{g, paddcvec.go(t, p, dv)}

def paddcvec(+gs: +List<+List<F32>>, +k: Nat, +dv: +List<F32>) -> +List<+List<F32>>:
  List.reverse(&2, +List<F32>, paddcvec.go(List.reverse(&2, +List<F32>, gs), k, dv))

def paddsc.go(+pg: +List<F32>, k: Nat, +x: F32) -> +List<F32>:
  match pg k:
    case Nil{} _:
      Nil{}
    case Con{g, t} 0n:
      Con{(g + x : F32), t}
    case Con{g, t} 1n+p:
      Con{g, paddsc.go(t, p, x)}

def paddsc(+pg: +List<F32>, +k: Nat, +x: F32) -> +List<F32>:
  List.reverse(&2, F32, paddsc.go(List.reverse(&2, F32, pg), k, x))

def vadd.go(+tp: +List<T.TN>, +a: U32, +b: U32, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>) -> SW2:
  SW2{paddcvec(paddcvec(gs, U32.to_nat(a), g), U32.to_nat(b), g), pg}

def vmul.go(+xs: +List<F32>, +ys: +List<F32>, +g: +List<F32>, +gs: +List<+List<F32>>, +a: U32, +b: U32, +pg: +List<F32>) -> SW2:
  SW2{paddcvec(paddcvec(gs, U32.to_nat(a), G.tmul.zip(ys, g)), U32.to_nat(b), G.tmul.zip(xs, g)), pg}

def vlin.dx.go(+rows: +List<+List<F32>>, +dy: +List<F32>, +acc: +List<F32>) -> +List<F32>:
  match rows dy:
    case Nil{} Nil{}:
      acc
    case Con{wr, wt} Con{dr, dt}:
      vlin.dx.go(wt, dt, G.tadd.zip(acc, G.tscale.map(wr, dr)))
    case _ _:
      acc

 def vlin.rows2(+rs: +List<Nat>, +tab: +List<F32>, +base: U32, +cols: Nat) -> +List<+List<F32>>:
   match rs:
     case Nil{}:
       Nil{}
     case Con{h, t}:
       Con{G.tgather(tab, base, h, cols), vlin.rows2(t, tab, base, cols)}

 def vlin.rows(+tab: +List<F32>, +base: U32, rows: Nat, +cols: Nat) -> +List<+List<F32>>:
   vlin.rows2(List.range(rows), tab, base, cols)

 def vlin.dw.row(+x: +List<F32>, +dr: F32, +b: U32, +pg: +List<F32>) -> +List<F32>:
   match x:
     case Nil{}:
       pg
     case Con{xj, xt}:
       vlin.dw.row(xt, dr, (b + 1 : U32), paddsc(pg, U32.to_nat(b), (dr * xj : F32)))

 def vlin.dw.go(+dy: +List<F32>, +x: +List<F32>, +base: U32, +cols: Nat, +pg: +List<F32>) -> +List<F32>:
   match dy:
     case Nil{}:
       pg
     case Con{dj, dt}:
       vlin.dw.go(dt, x, (base + U32.from_nat(cols) : U32), cols, vlin.dw.row(x, dj, base, pg))

 def vlin.dx(+tab: +List<F32>, +base: U32, rows: Nat, +cols: Nat, +dy: +List<F32>) -> +List<F32>:
   vlin.dx.go(vlin.rows(tab, base, rows, cols), dy, G.zeros(cols))

 def vlin.dw(+dy: +List<F32>, +x: +List<F32>, +base: U32, +cols: Nat, +pg: +List<F32>) -> +List<F32>:
   vlin.dw.go(dy, x, base, cols, pg)

def vlin(+tab: +List<F32>, +base: U32, rows: Nat, +cols: Nat, +x: +List<F32>, +dy: +List<F32>, +gs: +List<+List<F32>>, +a: U32, +pg: +List<F32>) -> SW2:
  SW2{paddcvec(gs, U32.to_nat(a), vlin.dx(tab, base, rows, cols, dy)), vlin.dw(dy, x, base, cols, pg)}
 def vrms.m2(+ss: F32, +n: Nat) -> F32:
   ((F32.div(ss, F32.from_nat(n)) + 0.00001 : F32))

 def vrms.cf(+s: F32, +ss: F32, +n: Nat) -> F32:
   F32.div(s, (F32.from_nat(n) * vrms.m2(ss, n) * F32.sqrt(vrms.m2(ss, n)) : F32))

 def vrms.dx.go(+xs: +List<F32>, +ys: +List<F32>, +s: F32, +cf: F32) -> +List<F32>:
   match xs ys:
     case Nil{} Nil{}:
       Nil{}
     case Con{x, xt} Con{y, yt}:
       Con{((s * y - cf * x : F32)), vrms.dx.go(xt, yt, s, cf)}
     case _ _:
       Nil{}
 def vrms.bwd.go(+x: +List<F32>, +dy: +List<F32>, +ss: F32, +n: Nat) -> +List<F32>:
   vrms.dx.go(x, dy, G.rmsnorm.sc(ss, n), vrms.cf(G.dot(dy, x), ss, n))


 def vrms.bwd.ss(p: Nat & F32, +x: +List<F32>, +dy: +List<F32>) -> +List<F32>:
   match p:
     case (n, ss):
       vrms.bwd.go(x, dy, ss, n)

 def vrms.bwd(+x: +List<F32>, +dy: +List<F32>) -> +List<F32>:
   vrms.bwd.ss(G.sumsq(x), x, dy)

 def vsm.bwd.go(+p: +List<F32>, +dy: +List<F32>, +s: F32) -> +List<F32>:
   match p dy:
     case Nil{} Nil{}:
       Nil{}
     case Con{pi, pt} Con{di, dt}:
       Con{(pi * (di - s : F32) : F32), vsm.bwd.go(pt, dt, s)}
     case _ _:
       Nil{}

 def vsm.bwd(+x: +List<F32>, +dy: +List<F32>) -> +List<F32>:
   vsm.bwd.go(G.softmax(x), dy, G.dot(G.softmax(x), dy))

 def vrelu.bwd.go(+x: +List<F32>, +dy: +List<F32>) -> +List<F32>:
   match x dy:
     case Nil{} Nil{}:
       Nil{}
     case Con{xi, xt} Con{di, dt}:
       Con{Bool.pick(F32, F32.is_gt(xi, {0.0 : F32}), di, {0.0 : F32}), vrelu.bwd.go(xt, dt)}
     case _ _:
       Nil{}

 def vrelu.bwd(+x: +List<F32>, +dy: +List<F32>) -> +List<F32>:
   vrelu.bwd.go(x, dy)

 def vgat.bwd.go(+dy: +List<F32>, +b: U32, +pg: +List<F32>) -> +List<F32>:
   match dy:
     case Nil{}:
       pg
     case Con{dj, dt}:
       vgat.bwd.go(dt, (b + 1 : U32), paddsc(pg, U32.to_nat(b), dj))

 def vgat.bwd(+dy: +List<F32>, +base: U32, +pg: +List<F32>) -> +List<F32>:
   vgat.bwd.go(dy, base, pg)

 def vsc.tr(+ds: +List<F32>, +ks: +List<F32>, +q: +List<F32>) -> +List<F32>:
   match ds:
     case Nil{}:
       Nil{}
     case Con{dt, dtt}:
       match ks:
         case Nil{}:
           Nil{}
         case Con{k0, t0}:
           match t0:
             case Nil{}:
               Nil{}
             case Con{k1, t1}:
               match t1:
                 case Nil{}:
                   Nil{}
                 case Con{k2, t2}:
                   match t2:
                     case Nil{}:
                       Nil{}
                     case Con{k3, t3}:
                       List.append(&2, F32, G.tscale.map(q, dt), vsc.tr(dtt, t3, q))
 def vsc.rt(+flat: +List<F32>, +ids: +List<U32>, +gs: +List<+List<F32>>) -> +List<+List<F32>>:
   match flat:
     case Nil{}:
       gs
     case Con{f0, t0}:
       match t0:
         case Nil{}:
           gs
         case Con{f1, t1}:
           match t1:
             case Nil{}:
               gs
             case Con{f2, t2}:
               match t2:
                 case Nil{}:
                   gs
                 case Con{f3, t3}:
                   match ids:
                     case Nil{}:
                       gs
                     case Con{i, ti}:
                       vsc.rt(t3, ti, paddcvec(gs, U32.to_nat(i), Con{f0, Con{f1, Con{f2, Con{f3, Nil{}}}}}))
 def vchunk4(+xs: +List<F32>) -> +List<+List<F32>>:
   match xs:
     case Nil{}:
       Nil{}
     case Con{a, t0}:
       match t0:
         case Nil{}:
           Con{Con{a, Nil{}}, Nil{}}
         case Con{b, t1}:
           match t1:
             case Nil{}:
               Con{Con{a, Con{b, Nil{}}}, Nil{}}
             case Con{c, t2}:
               match t2:
                 case Nil{}:
                   Con{Con{a, Con{b, Con{c, Nil{}}}}, Nil{}}
                 case Con{d, t3}:
                   Con{Con{a, Con{b, Con{c, Con{d, Nil{}}}}}, vchunk4(t3)}

 def vws.dv.go(+ws: +List<F32>, +vs: +List<+List<F32>>) -> +List<F32>:
   match ws vs:
     case Nil{} Nil{}:
       Nil{}
     case Con{w, wt} Con{v, vt}:
       List.append(&2, F32, G.tscale.map(v, w), vws.dv.go(wt, vt))
     case _ _:
       Nil{}

 def pick2.go(z: Bool) -> F32:
   match z:
     case True{}:
       {1.0 : F32}
     case False{}:
       {0.0 : F32}

 def pick2(+i: U32, +tgt: U32) -> F32:
   pick2.go(U32.is_eq(i, tgt))

 def seed.go(+ps: +List<F32>, +tgt: U32, +i: U32) -> +List<F32>:
   match ps:
     case Nil{}:
       Nil{}
     case Con{p, t}:
       Con{(p - pick2(i, tgt) : F32), seed.go(t, tgt, (i + 1 : U32))}

 def seed(+ps: +List<F32>, +tgt: U32) -> +List<F32>:
   seed.go(ps, tgt, 0)

 def seed.gs(+tp: +List<T.TN>) -> +List<+List<F32>>:
   match tp:
     case Nil{}:
       Nil{}
     case Con{nd, t}:
       Con{Nil{}, seed.gs(t)}

 def sweep.is10(z: Bool, +op: U32, +a: U32, +b: U32, +xs: +List<F32>, +ys: +List<F32>, +ids: +List<U32>, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>, +params: +List<F32>) -> SW2:
   match z:
     case True{}:
       SW2{vsc.rt(vws.dv.go(xs, vchunk4(ys)), ids, paddcvec(gs, U32.to_nat(a), vlin.dx(ys, 0, U32.to_nat(b), 4n, g))), pg}
     case False{}:
       SW2{gs, pg}
 def sweep.is9(z: Bool, +op: U32, +a: U32, +b: U32, +xs: +List<F32>, +ys: +List<F32>, +ids: +List<U32>, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>, +params: +List<F32>) -> SW2:
   match z:
     case True{}:
       SW2{vsc.rt(vsc.tr(g, ys, xs), ids, paddcvec(gs, U32.to_nat(a), vlin.dx(ys, 0, U32.to_nat(b), 4n, g))), pg}
     case False{}:
       sweep.is10(U32.is_eq(op, 10), op, a, b, xs, ys, ids, g, gs, pg, params)


 def sweep.is8(z: Bool, +op: U32, +a: U32, +b: U32, +c: U32, +xs: +List<F32>, +ys: +List<F32>, +ids: +List<U32>, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>, +params: +List<F32>) -> SW2:
   match z:
     case True{}:
       SW2{gs, vgat.bwd(g, a, pg)}
     case False{}:
       sweep.is9(U32.is_eq(op, 9), op, a, b, xs, ys, ids, g, gs, pg, params)

 def sweep.is7(z: Bool, +op: U32, +a: U32, +b: U32, +c: U32, +xs: +List<F32>, +ys: +List<F32>, +ids: +List<U32>, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>, +params: +List<F32>) -> SW2:
   match z:
     case True{}:
       SW2{gs, pg}
     case False{}:
       sweep.is8(U32.is_eq(op, 8), op, a, b, c, xs, ys, ids, g, gs, pg, params)

 def sweep.is6(z: Bool, +op: U32, +a: U32, +b: U32, +c: U32, +xs: +List<F32>, +ys: +List<F32>, +ids: +List<U32>, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>, +params: +List<F32>) -> SW2:
   match z:
     case True{}:
       SW2{paddcvec(gs, U32.to_nat(a), vrelu.bwd(xs, g)), pg}
     case False{}:
       sweep.is7(U32.is_eq(op, 7), op, a, b, c, xs, ys, ids, g, gs, pg, params)

 def sweep.is5(z: Bool, +op: U32, +a: U32, +b: U32, +c: U32, +xs: +List<F32>, +ys: +List<F32>, +ids: +List<U32>, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>, +params: +List<F32>) -> SW2:
   match z:
     case True{}:
       SW2{paddcvec(gs, U32.to_nat(a), vsm.bwd(xs, g)), pg}
     case False{}:
       sweep.is6(U32.is_eq(op, 6), op, a, b, c, xs, ys, ids, g, gs, pg, params)

 def sweep.is4(z: Bool, +op: U32, +a: U32, +b: U32, +c: U32, +xs: +List<F32>, +ys: +List<F32>, +ids: +List<U32>, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>, +params: +List<F32>) -> SW2:
   match z:
     case True{}:
       SW2{paddcvec(gs, U32.to_nat(a), vrms.bwd(xs, g)), pg}
     case False{}:
       sweep.is5(U32.is_eq(op, 5), op, a, b, c, xs, ys, ids, g, gs, pg, params)

 def sweep.is3(z: Bool, +op: U32, +a: U32, +b: U32, +c: U32, +xs: +List<F32>, +ys: +List<F32>, +ids: +List<U32>, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>, +params: +List<F32>) -> SW2:
   match z:
     case True{}:
       SW2{paddcvec(gs, U32.to_nat(a), vlin.dx(params, a, U32.to_nat(b), U32.to_nat(c), g)), vlin.dw(g, xs, a, U32.to_nat(c), pg)}
     case False{}:
       sweep.is4(U32.is_eq(op, 4), op, a, b, c, xs, ys, ids, g, gs, pg, params)

 def sweep.is2(z: Bool, +op: U32, +a: U32, +b: U32, +c: U32, +xs: +List<F32>, +ys: +List<F32>, +ids: +List<U32>, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>, +params: +List<F32>) -> SW2:
   match z:
     case True{}:
       SW2{paddcvec(paddcvec(gs, U32.to_nat(a), G.tmul.zip(ys, g)), U32.to_nat(b), G.tmul.zip(xs, g)), pg}
     case False{}:
       sweep.is3(U32.is_eq(op, 3), op, a, b, c, xs, ys, ids, g, gs, pg, params)

 def sweep.is1(z: Bool, +op: U32, +a: U32, +b: U32, +c: U32, +xs: +List<F32>, +ys: +List<F32>, +ids: +List<U32>, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>, +params: +List<F32>) -> SW2:
   match z:
     case True{}:
       SW2{paddcvec(paddcvec(gs, U32.to_nat(a), g), U32.to_nat(b), g), pg}
     case False{}:
       sweep.is2(U32.is_eq(op, 2), op, a, b, c, xs, ys, ids, g, gs, pg, params)

 def sweep.one(nd: T.TN, +g: +List<F32>, +gs: +List<+List<F32>>, +pg: +List<F32>, +params: +List<F32>) -> SW2:
   match nd:
     case T.TN{+op, +a, +b, +c, +xs, +ys, +ids}:
       sweep.is1(U32.is_eq(op, 1), op, a, b, c, xs, ys, ids, g, gs, pg, params)

 def sweep.go(+tp: +List<T.TN>, acc: SW2, +params: +List<F32>) -> SW2:
   match tp:
     case Nil{}:
       acc
     case Con{nd, t}:
       match acc:
         case SW2{gs, pg}:
           match gs:
             case Nil{}:
               acc
             case Con{g, gt}:
               sweep.go(t, sweep.one(nd, g, gt, pg, params), params)
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

 def adam.up(+p: F32, +g: F32, +m: F32, +v: F32, +t1: F32, +lr: F32, +b1: F32, +b2: F32, +eps: F32) -> F32 & (F32 & F32):
   ((p - adam.dl(m, g, v, b1, b2, t1, lr, eps) : F32), (adam.mn(m, g, b1), adam.vn(v, g, b2)))

 def adam.nx(u: F32 & (F32 & F32), r: (+List<F32> & +List<F32>) & +List<F32>) -> (+List<F32> & +List<F32>) & +List<F32>:
   match u:
     case (p2, (m2, v2)):
       match r:
         case ((pt, mt), vt):
           ((Con{p2, pt}, Con{m2, mt}), Con{v2, vt})

 def adam.go(+ps: +List<F32>, +gs: +List<F32>, +ms: +List<F32>, +vs: +List<F32>, +t1: F32, +lr: F32, +b1: F32, +b2: F32, +eps: F32) -> (+List<F32> & +List<F32>) & +List<F32>:
   match ps gs ms vs:
     case Nil{} Nil{} Nil{} Nil{}:
       ((Nil{}, Nil{}), Nil{})
     case Con{p, pt} Con{g, gt} Con{m, mt} Con{v, vt}:
       adam.nx(adam.up(p, g, m, v, t1, lr, b1, b2, eps), adam.go(pt, gt, mt, vt, t1, lr, b1, b2, eps))
     case _ _ _ _:
       ((Nil{}, Nil{}), Nil{})
