import Base
import ./tape.bend as T

# Transformer forward on the tape. Activations are bare U32 node ids in
# +List<U32> vectors. Weight matrices are implicit: element (r,c) of the
# matrix at param base B is param B + r*C + c, so dots walk the input vector
# while incrementing a +U32 cursor — no counter lists, no reuse traps.
# Dims match microgpt (E=16 H=4 D=4 L=1 B=16); heads and the layer are
# unrolled. V (vocab) is threaded; all other sizes are literals.
# Order: every callee precedes its callers (Bend has no forward references).

def idx.map(+xs: +List<Nat>, +base: U32, +stride: U32) -> +List<U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{(base + U32.mul(U32.from_nat(h), stride) : U32), idx.map(t, base, stride)}

def erow(+tok: U32, +base: U32) -> +List<U32>:
  idx.map(List.range(16n), (base + U32.mul(tok, 16) : U32), 1)

def embs(+ws: +List<U32>) -> T.TP<+List<U32>>:
  match ws:
    case Nil{}:
      do T.TP<+List<U32>>:
        return Nil{}
    case Con{w, wt}:
      do T.TP<+List<U32>>:
        v : U32 <- T.vleaf(w)
        rest : +List<U32> <- embs(wt)
        return Con{v, rest}

def vdot.term(+x: U32, +w: U32, acc: U32) -> T.TP<U32>:
  do T.TP<U32>:
    wl : U32 <- T.vleaf(w)
    m : U32 <- T.vmul(wl, x)
    T.vadd(m, acc)

def vdot.go(+xs: +List<U32>, +w: U32, acc: U32) -> T.TP<U32>:
  match xs:
    case Nil{}:
      do T.TP<U32>:
        return acc
    case Con{x, xt}:
      do T.TP<U32>:
        s1 : U32 <- vdot.term(x, w, acc)
        vdot.go(xt, (w + 1 : U32), s1)

def vdot(+xs: +List<U32>, +w: U32) -> T.TP<U32>:
  do T.TP<U32>:
    z : U32 <- T.vconst({0.0 : F32})
    vdot.go(xs, w, z)

def vlin.go(+rs: +List<Nat>, +base: U32, +cols: U32, +x: +List<U32>, +acc: +List<U32>) -> T.TP<+List<U32>>:
  match rs:
    case Nil{}:
      do T.TP<+List<U32>>:
        return List.reverse(&2, U32, acc)
    case Con{h, t}:
      do T.TP<+List<U32>>:
        z : U32 <- T.vconst({0.0 : F32})
        y : U32 <- vdot.go(x, (base + U32.mul(U32.from_nat(h), cols) : U32), z)
        vlin.go(t, base, cols, x, Con{y, acc})

def vlin(+base: U32, rows: Nat, +cols: U32, +x: +List<U32>) -> T.TP<+List<U32>>:
  vlin.go(List.range(rows), base, cols, x, Nil{})

def vadd.zip(+xs: +List<U32>, +ys: +List<U32>) -> T.TP<+List<U32>>:
  match xs ys:
    case Nil{} Nil{}:
      do T.TP<+List<U32>>:
        return Nil{}
    case Con{x, xt} Con{y, yt}:
      do T.TP<+List<U32>>:
        z : U32 <- T.vadd(x, y)
        rest : +List<U32> <- vadd.zip(xt, yt)
        return Con{z, rest}
    case _ _:
      do T.TP<+List<U32>>:
        return Nil{}

def vrelu.map(+xs: +List<U32>) -> T.TP<+List<U32>>:
  match xs:
    case Nil{}:
      do T.TP<+List<U32>>:
        return Nil{}
    case Con{x, xt}:
      do T.TP<+List<U32>>:
        y : U32 <- T.vrelu(x)
        rest : +List<U32> <- vrelu.map(xt)
        return Con{y, rest}

def vrms.term(+x: U32, acc: U32) -> T.TP<U32>:
  do T.TP<U32>:
    s : U32 <- T.vmul(x, x)
    T.vadd(s, acc)

def vrms.sq(+xs: +List<U32>, n: Nat, acc: U32) -> T.TP<(U32 & Nat)>:
  match xs:
    case Nil{}:
      do T.TP<(U32 & Nat)>:
        return (acc, n)
    case Con{x, xt}:
      do T.TP<(U32 & Nat)>:
        q : U32 <- vrms.term(x, acc)
        vrms.sq(xt, 1n + n, q)

def vrms.map2(+xs: +List<U32>, +s: U32) -> T.TP<+List<U32>>:
  match xs:
    case Nil{}:
      do T.TP<+List<U32>>:
        return Nil{}
    case Con{x, xt}:
      do T.TP<+List<U32>>:
        y : U32 <- T.vmul(x, s)
        rest : +List<U32> <- vrms.map2(xt, s)
        return Con{y, rest}

def vrms.sc4(mse: U32, +xs: +List<U32>) -> T.TP<+List<U32>>:
  do T.TP<+List<U32>>:
    sc : U32 <- T.vpow(mse, ((0.0 - 0.5 : F32)))
    vrms.map2(xs, sc)

def vrms.sc3(mean: U32, +xs: +List<U32>) -> T.TP<+List<U32>>:
  do T.TP<+List<U32>>:
    e : U32 <- T.vconst({0.00001 : F32})
    mse : U32 <- T.vadd(mean, e)
    vrms.sc4(mse, xs)

def vrms.sc2(+sum: U32, +nc: U32, +xs: +List<U32>) -> T.TP<+List<U32>>:
  do T.TP<+List<U32>>:
    mean : U32 <- T.vdiv(sum, nc)
    vrms.sc3(mean, xs)

def vrms.sc(sum: U32, n: Nat, +xs: +List<U32>) -> T.TP<+List<U32>>:
  do T.TP<+List<U32>>:
    nc : U32 <- T.vconst(F32.from_nat(n))
    vrms.sc2(sum, nc, xs)

def vrms.mapX(sn: U32 & Nat, +xs: +List<U32>) -> T.TP<+List<U32>>:
  (sum, n) = sn
  vrms.sc(sum, n, xs)

def vrms(+xs: +List<U32>) -> T.TP<+List<U32>>:
  do T.TP<+List<U32>>:
    z : U32 <- T.vconst({0.0 : F32})
    sn : U32 & Nat <- vrms.sq(xs, 0n, z)
    vrms.mapX(sn, xs)

def vmax.mx(xn: T.Node, +best: F32) -> F32:
  match xn:
    case T.N{o, a, b, k, xd}:
      F32.max(xd, best)

def vmax.go(+xs: +List<U32>, +best: F32) -> T.TP<F32>:
  match xs:
    case Nil{}:
      do T.TP<F32>:
        return best
    case Con{x, xt}:
      do T.TP<F32>:
        xn : T.Node <- T.tget(x)
        vmax.go(xt, vmax.mx(xn, best))

def vmax.hd2(xn: T.Node) -> F32:
  match xn:
    case T.N{o, a, b, k, xd}:
      xd

def vmax.hd(+xs: +List<U32>) -> T.TP<F32>:
  match xs:
    case Nil{}:
      do T.TP<F32>:
        return {0.0 : F32}
    case Con{x, xt}:
      do T.TP<F32>:
        xn : T.Node <- T.tget(x)
        vmax.go(xt, vmax.hd2(xn))

def vmax(+xs: +List<U32>) -> T.TP<F32>:
  vmax.hd(xs)

def vsm.exp1(+x: U32, +m: F32) -> T.TP<U32>:
  do T.TP<U32>:
    c : U32 <- T.vconst(m)
    d : U32 <- T.vsub(x, c)
    T.vexp(d)

def vsm.exps(+xs: +List<U32>, +m: F32) -> T.TP<+List<U32>>:
  match xs:
    case Nil{}:
      do T.TP<+List<U32>>:
        return Nil{}
    case Con{x, xt}:
      do T.TP<+List<U32>>:
        e : U32 <- vsm.exp1(x, m)
        rest : +List<U32> <- vsm.exps(xt, m)
        return Con{e, rest}

def vsum.go(+xs: +List<U32>, acc: U32) -> T.TP<U32>:
  match xs:
    case Nil{}:
      do T.TP<U32>:
        return acc
    case Con{x, xt}:
      do T.TP<U32>:
        s : U32 <- T.vadd(acc, x)
        vsum.go(xt, s)

def vsum(+xs: +List<U32>) -> T.TP<U32>:
  do T.TP<U32>:
    z : U32 <- T.vconst({0.0 : F32})
    vsum.go(xs, z)

def vsm.div(+xs: +List<U32>, +tot: U32) -> T.TP<+List<U32>>:
  match xs:
    case Nil{}:
      do T.TP<+List<U32>>:
        return Nil{}
    case Con{x, xt}:
      do T.TP<+List<U32>>:
        y : U32 <- T.vdiv(x, tot)
        rest : +List<U32> <- vsm.div(xt, tot)
        return Con{y, rest}

def vsm.top2(+es: +List<U32>) -> T.TP<+List<U32>>:
  do T.TP<+List<U32>>:
    tot : U32 <- vsum(es)
    vsm.div(es, tot)

def vsm.top(+xs: +List<U32>, +m: F32) -> T.TP<+List<U32>>:
  do T.TP<+List<U32>>:
    es : +List<U32> <- vsm.exps(xs, m)
    vsm.top2(es)

def vsoftmax(+xs: +List<U32>) -> T.TP<+List<U32>>:
  do T.TP<+List<U32>>:
    m : F32 <- vmax(xs)
    vsm.top(xs, m)
 def vdot2.go(+a: +List<U32>, +b: +List<U32>, acc: U32) -> T.TP<U32>:
   match a b:
     case Nil{} Nil{}:
       do T.TP<U32>:
         return acc
     case Con{x, xt} Con{y, yt}:
       do T.TP<U32>:
         m : U32 <- T.vmul(x, y)
         s : U32 <- T.vadd(m, acc)
         vdot2.go(xt, yt, s)
     case _ _:
       do T.TP<U32>:
         return acc

 def vdot2(+a: +List<U32>, +b: +List<U32>) -> T.TP<U32>:
   do T.TP<U32>:
     z : U32 <- T.vconst({0.0 : F32})
     vdot2.go(a, b, z)

 def split4.d(+h0: +List<U32>, +h1: +List<U32>, +h2: +List<U32>, +r3: +List<U32>) -> (+List<U32> & +List<U32>) & (+List<U32> & +List<U32>):
   ((h0, h1), (h2, List.take(&2, U32, r3, 4n)))

 def split4.c(+h0: +List<U32>, +h1: +List<U32>, +r2: +List<U32>) -> (+List<U32> & +List<U32>) & (+List<U32> & +List<U32>):
   split4.d(h0, h1, List.take(&2, U32, r2, 4n), List.drop(&2, U32, r2, 4n))

 def split4.b(+h0: +List<U32>, +r1: +List<U32>) -> (+List<U32> & +List<U32>) & (+List<U32> & +List<U32>):
   split4.c(h0, List.take(&2, U32, r1, 4n), List.drop(&2, U32, r1, 4n))

 def split4(+xs: +List<U32>) -> (+List<U32> & +List<U32>) & (+List<U32> & +List<U32>):
   split4.b(List.take(&2, U32, xs, 4n), List.drop(&2, U32, xs, 4n))
 def score.go(+ks: +List<+List<U32>>, +q: +List<U32>, +half: U32, +acc: +List<U32>) -> T.TP<+List<U32>>:
   match ks:
     case Nil{}:
       do T.TP<+List<U32>>:
         return acc
     case Con{k, kt}:
       do T.TP<+List<U32>>:
         d : U32 <- vdot2(q, k)
         s : U32 <- T.vmul(d, half)
         score.go(kt, q, half, Con{s, acc})

 def zeros4() -> T.TP<+List<U32>>:
   do T.TP<+List<U32>>:
     a : U32 <- T.vconst({0.0 : F32})
     b : U32 <- T.vconst({0.0 : F32})
     c : U32 <- T.vconst({0.0 : F32})
     d : U32 <- T.vconst({0.0 : F32})
     return Con{a, Con{b, Con{c, Con{d, Nil{}}}}}

 def wsum.add(+w: U32, +v: +List<U32>, +acc: +List<U32>) -> T.TP<+List<U32>>:
   match v acc:
     case Nil{} Nil{}:
       do T.TP<+List<U32>>:
         return Nil{}
     case Con{vj, vt} Con{aj, at}:
       do T.TP<+List<U32>>:
         m : U32 <- T.vmul(w, vj)
         s : U32 <- T.vadd(m, aj)
         rest : +List<U32> <- wsum.add(w, vt, at)
         return Con{s, rest}
     case _ _:
       do T.TP<+List<U32>>:
         return Nil{}

 def wsum.go(+ws: +List<U32>, +vs: +List<+List<U32>>, +acc: +List<U32>) -> T.TP<+List<U32>>:
   match ws vs:
     case Nil{} Nil{}:
       do T.TP<+List<U32>>:
         return acc
     case Con{w, wt} Con{v, vt}:
       do T.TP<+List<U32>>:
         a2 : +List<U32> <- wsum.add(w, v, acc)
         wsum.go(wt, vt, a2)
     case _ _:
       do T.TP<+List<U32>>:
         return acc

 def wsum(+ws: +List<U32>, +vs: +List<+List<U32>>) -> T.TP<+List<U32>>:
   do T.TP<+List<U32>>:
     z : +List<U32> <- zeros4()
     wsum.go(ws, vs, z)

 def attn.head(+q: +List<U32>, +k: +List<U32>, +v: +List<U32>, +ks: +List<+List<U32>>, +vs: +List<+List<U32>>, +half: U32) -> T.TP<(+List<U32> & (+List<+List<U32>> & +List<+List<U32>>))>:
   do T.TP<(+List<U32> & (+List<+List<U32>> & +List<+List<U32>>))>:
     sc : +List<U32> <- score.go(Con{k, ks}, q, half, Nil{})
     w : +List<U32> <- vsoftmax(sc)
     o : +List<U32> <- wsum(w, Con{v, vs})
     return (o, ((Con{k, ks}, Con{v, vs})))

 type AC is Data:
   AC{k0: +List<+List<U32>>, v0: +List<+List<U32>>, k1: +List<+List<U32>>, v1: +List<+List<U32>>, k2: +List<+List<U32>>, v2: +List<+List<U32>>, k3: +List<+List<U32>>, v3: +List<+List<U32>>}

 def ac0() -> AC:
   AC{Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}}

 def cat4(+a: +List<U32>, +b: +List<U32>, +c: +List<U32>, +d: +List<U32>) -> +List<U32>:
   List.append(&2, U32, a, List.append(&2, U32, b, List.append(&2, U32, c, d)))

 def blk.h3(r2: +List<U32> & (+List<+List<U32>> & +List<+List<U32>>), +o0: +List<U32>, +o1: +List<U32>, +o2: +List<U32>, +nk0: +List<+List<U32>>, +nv0: +List<+List<U32>>, +nk1: +List<+List<U32>>, +nv1: +List<+List<U32>>, +nk2: +List<+List<U32>>, +nv2: +List<+List<U32>>, +q3: +List<U32>, +k3: +List<U32>, +v3: +List<U32>, +k3c: +List<+List<U32>>, +v3c: +List<+List<U32>>, +half: U32) -> T.TP<(+List<U32> & AC)>:
   match r2:
     case (o3, (nk3, nv3)):
       do T.TP<(+List<U32> & AC)>:
         return (cat4(o0, o1, o2, o3), AC{nk0, nv0, nk1, nv1, nk2, nv2, nk3, nv3})

 def blk.r2(r2: +List<U32> & (+List<+List<U32>> & +List<+List<U32>>), +o0: +List<U32>, +o1: +List<U32>, +nk0: +List<+List<U32>>, +nv0: +List<+List<U32>>, +nk1: +List<+List<U32>>, +nv1: +List<+List<U32>>, +q2: +List<U32>, +k2: +List<U32>, +v2: +List<U32>, +k2c: +List<+List<U32>>, +v2c: +List<+List<U32>>, +q3: +List<U32>, +k3: +List<U32>, +v3: +List<U32>, +k3c: +List<+List<U32>>, +v3c: +List<+List<U32>>, +half: U32) -> T.TP<(+List<U32> & AC)>:
   match r2:
     case (o2, (nk2, nv2)):
       do T.TP<(+List<U32> & AC)>:
         r3 : +List<U32> & (+List<+List<U32>> & +List<+List<U32>>) <- attn.head(q3, k3, v3, k3c, v3c, half)
         blk.h3(r3, o0, o1, o2, nk0, nv0, nk1, nv1, nk2, nv2, q3, k3, v3, k3c, v3c, half)

 def blk.r1(r1: +List<U32> & (+List<+List<U32>> & +List<+List<U32>>), +o0: +List<U32>, +nk0: +List<+List<U32>>, +nv0: +List<+List<U32>>, +q2: +List<U32>, +k2: +List<U32>, +v2: +List<U32>, +k2c: +List<+List<U32>>, +v2c: +List<+List<U32>>, +q3: +List<U32>, +k3: +List<U32>, +v3: +List<U32>, +k3c: +List<+List<U32>>, +v3c: +List<+List<U32>>, +half: U32) -> T.TP<(+List<U32> & AC)>:
   match r1:
     case (o1, (nk1, nv1)):
       do T.TP<(+List<U32> & AC)>:
         r2 : +List<U32> & (+List<+List<U32>> & +List<+List<U32>>) <- attn.head(q2, k2, v2, k2c, v2c, half)
         blk.r2(r2, o0, o1, nk0, nv0, nk1, nv1, q2, k2, v2, k2c, v2c, q3, k3, v3, k3c, v3c, half)

 def blk.r0(r0: +List<U32> & (+List<+List<U32>> & +List<+List<U32>>), +q1: +List<U32>, +k1: +List<U32>, +v1: +List<U32>, +k1c: +List<+List<U32>>, +v1c: +List<+List<U32>>, +q2: +List<U32>, +k2: +List<U32>, +v2: +List<U32>, +k2c: +List<+List<U32>>, +v2c: +List<+List<U32>>, +q3: +List<U32>, +k3: +List<U32>, +v3: +List<U32>, +k3c: +List<+List<U32>>, +v3c: +List<+List<U32>>, +half: U32) -> T.TP<(+List<U32> & AC)>:
   match r0:
     case (o0, (nk0, nv0)):
       do T.TP<(+List<U32> & AC)>:
         r1 : +List<U32> & (+List<+List<U32>> & +List<+List<U32>>) <- attn.head(q1, k1, v1, k1c, v1c, half)
         blk.r1(r1, o0, nk0, nv0, q2, k2, v2, k2c, v2c, q3, k3, v3, k3c, v3c, half)


 def blk.top(+q0: +List<U32>, +q1: +List<U32>, +k0: +List<U32>, +k1: +List<U32>, +v0: +List<U32>, +v1: +List<U32>, +q2: +List<U32>, +q3: +List<U32>, +k2: +List<U32>, +k3: +List<U32>, +v2: +List<U32>, +v3: +List<U32>, ac: AC, +half: U32) -> T.TP<(+List<U32> & AC)>:
   match ac:
     case AC{k0c, v0c, k1c, v1c, k2c, v2c, k3c, v3c}:
       do T.TP<(+List<U32> & AC)>:
         r0 : +List<U32> & (+List<+List<U32>> & +List<+List<U32>>) <- attn.head(q0, k0, v0, k0c, v0c, half)
         blk.r0(r0, q1, k1, v1, k1c, v1c, q2, k2, v2, k2c, v2c, q3, k3, v3, k3c, v3c, half)
 def blk.heads(qs: (+List<U32> & +List<U32>) & (+List<U32> & +List<U32>), ks: (+List<U32> & +List<U32>) & (+List<U32> & +List<U32>), vs: (+List<U32> & +List<U32>) & (+List<U32> & +List<U32>), ac: AC, +half: U32) -> T.TP<(+List<U32> & AC)>:
   ((q0, q1), (q2, q3)) = qs
   ((k0, k1), (k2, k3)) = ks
   ((v0, v1), (v2, v3)) = vs
   blk.top(q0, q1, k0, k1, v0, v1, q2, q3, k2, k3, v2, v3, ac, half)
 def base.wpe(+V: U32) -> U32:
   (V * 16 : U32)

 def base.lm(+V: U32) -> U32:
   ((V * 16 : U32) + 256 : U32)

 def base.wq(+V: U32) -> U32:
   (base.lm(V) + (V * 16 : U32) : U32)

 def base.wk(+V: U32) -> U32:
   ((base.wq(V) + 256 : U32))

 def base.wv(+V: U32) -> U32:
   ((base.wk(V) + 256 : U32))

 def base.wo(+V: U32) -> U32:
   ((base.wv(V) + 256 : U32))

 def base.fc1(+V: U32) -> U32:
   ((base.wo(V) + 256 : U32))

 def base.fc2(+V: U32) -> U32:
   ((base.fc1(V) + 1024 : U32))

 def base.np(+V: U32) -> U32:
   ((base.fc2(V) + 1024 : U32))

 type Step is Data:
   S{logits: +List<U32>, ac: AC}

 def step.out(+x2: +List<U32>, +V: U32, ac: AC) -> T.TP<Step>:
   do T.TP<Step>:
     lg : +List<U32> <- vlin(base.lm(V), U32.to_nat(V), 16, x2)
     return S{lg, ac}

 def step.mlp5(+m: +List<U32>, +x1: +List<U32>, +V: U32, ac: AC) -> T.TP<Step>:
   do T.TP<Step>:
     x2 : +List<U32> <- vadd.zip(x1, m)
     step.out(x2, V, ac)

 def step.mlp4(+r: +List<U32>, +x1: +List<U32>, +V: U32, ac: AC) -> T.TP<Step>:
   do T.TP<Step>:
     m : +List<U32> <- vlin(base.fc2(V), 16n, 64, r)
     step.mlp5(m, x1, V, ac)

 def step.mlp3(+h: +List<U32>, +x1: +List<U32>, +V: U32, ac: AC) -> T.TP<Step>:
   do T.TP<Step>:
     r : +List<U32> <- vrelu.map(h)
     step.mlp4(r, x1, V, ac)

 def step.mlp2(+xm: +List<U32>, +x1: +List<U32>, +V: U32, ac: AC) -> T.TP<Step>:
   do T.TP<Step>:
     h : +List<U32> <- vlin(base.fc1(V), 64n, 16, xm)
     step.mlp3(h, x1, V, ac)

 def step.mlp(+x1: +List<U32>, +V: U32, ac: AC) -> T.TP<Step>:
   do T.TP<Step>:
     xm : +List<U32> <- vrms(x1)
     step.mlp2(xm, x1, V, ac)

 def step.res(+p: +List<U32>, +x0: +List<U32>, +V: U32, ac: AC) -> T.TP<Step>:
   do T.TP<Step>:
     x1 : +List<U32> <- vadd.zip(x0, p)
     step.mlp(x1, V, ac)

 def step.proj2(+o: +List<U32>, ac: AC, +x0: +List<U32>, +V: U32) -> T.TP<Step>:
   do T.TP<Step>:
     p : +List<U32> <- vlin(base.wo(V), 16n, 16, o)
     step.res(p, x0, V, ac)

 def step.proj(po: +List<U32> & AC, +x0: +List<U32>, +V: U32) -> T.TP<Step>:
   (o, ac) = po
   step.proj2(o, ac, x0, V)


 def step.attn3(+q: +List<U32>, +k: +List<U32>, +v: +List<U32>, +x0: +List<U32>, +V: U32, ac: AC, +half: U32) -> T.TP<Step>:
   do T.TP<Step>:
     po : +List<U32> & AC <- blk.heads(split4(q), split4(k), split4(v), ac, half)
     step.proj(po, x0, V)

 def step.attn2(qkv: (+List<U32> & +List<U32>) & +List<U32>, +x0: +List<U32>, +V: U32, ac: AC, +half: U32) -> T.TP<Step>:
   ((q, k), v) = qkv
   step.attn3(q, k, v, x0, V, ac, half)

 def step.qkv3(+q: +List<U32>, +k: +List<U32>, +xn: +List<U32>, +r: +List<U32>, +V: U32, ac: AC, +half: U32) -> T.TP<Step>:
   do T.TP<Step>:
     v : +List<U32> <- vlin(base.wv(V), 16n, 16, xn)
     step.attn2(((q, k), v), r, V, ac, half)

 def step.qkv2(+q: +List<U32>, +xn: +List<U32>, +r: +List<U32>, +V: U32, ac: AC, +half: U32) -> T.TP<Step>:
   do T.TP<Step>:
     k : +List<U32> <- vlin(base.wk(V), 16n, 16, xn)
     step.qkv3(q, k, xn, r, V, ac, half)

 def step.qkv1(+xn: +List<U32>, +r: +List<U32>, +V: U32, ac: AC, +half: U32) -> T.TP<Step>:
   do T.TP<Step>:
     q : +List<U32> <- vlin(base.wq(V), 16n, 16, xn)
     step.qkv2(q, xn, r, V, ac, half)

 def step.attn(+r: +List<U32>, +V: U32, ac: AC) -> T.TP<Step>:
   do T.TP<Step>:
     half : U32 <- T.vconst({0.5 : F32})
     xn : +List<U32> <- vrms(r)
     step.qkv1(xn, r, V, ac, half)

 def step.emb2(+te: +List<U32>, +pe: +List<U32>, +V: U32, ac: AC) -> T.TP<Step>:
   do T.TP<Step>:
     e : +List<U32> <- vadd.zip(te, pe)
     x0 : +List<U32> <- vrms(e)
     step.attn(x0, V, ac)

 def step.emb(+te: +List<U32>, +pos: Nat, +V: U32, ac: AC) -> T.TP<Step>:
   do T.TP<Step>:
     pe : +List<U32> <- embs(erow(U32.from_nat(pos), base.wpe(V)))
     step.emb2(te, pe, V, ac)

 def step(+tok: U32, pos: Nat, +V: U32, ac: AC) -> T.TP<Step>:
   do T.TP<Step>:
     te : +List<U32> <- embs(erow(tok, 0))
     step.emb(te, pos, V, ac)

 def vnth(+xs: +List<U32>, n: Nat) -> T.TP<U32>:
   match xs n:
     case Nil{} _:
       do T.TP<U32>:
         z : U32 <- T.vconst({0.0 : F32})
         return z
     case Con{x, xt} 0n:
       do T.TP<U32>:
         return x
     case Con{x, xt} 1n+p:
       vnth(xt, p)

 def loss.term(+logits: +List<U32>, +tgt: U32) -> T.TP<U32>:
   do T.TP<U32>:
     ps : +List<U32> <- vsoftmax(logits)
     p : U32 <- vnth(ps, U32.to_nat(tgt))
     l : U32 <- T.vlog(p)
     T.vneg(l)

 def step.lg(st: Step) -> T.TP<+List<U32>>:
   match st:
     case S{logits, cache2}:
       do T.TP<+List<U32>>:
         return logits

 def step.cc(st: Step) -> T.TP<AC>:
   match st:
     case S{logits, cache2}:
       do T.TP<AC>:
         return cache2

 def step.logits(+tok: U32, +pos: Nat, +V: U32, +cache: AC) -> T.TP<+List<U32>>:
   do T.TP<+List<U32>>:
     s : Step <- step(tok, pos, V, cache)
     step.lg(s)

 def step.cache(+tok: U32, +pos: Nat, +V: U32, +cache: AC) -> T.TP<AC>:
   do T.TP<AC>:
     s : Step <- step(tok, pos, V, cache)
     step.cc(s)

 def loss.go(+toks: +List<U32>, +pos: Nat, +V: U32, +cache: AC, acc: U32, n: Nat) -> T.TP<(U32 & (Nat & AC))>:
   match toks:
     case Nil{}:
       do T.TP<(U32 & (Nat & AC))>:
         return (acc, ((n, cache)))
     case Con{tok, rest}:
       match rest:
         case Nil{}:
           do T.TP<(U32 & (Nat & AC))>:
             return (acc, ((n, cache)))
         case Con{+tgt, tail}:
           do T.TP<(U32 & (Nat & AC))>:
             lg : +List<U32> <- step.logits(tok, pos, V, cache)
             c2 : AC <- step.cache(tok, pos, V, cache)
             l : U32 <- loss.term(lg, tgt)
             s2 : U32 <- T.vadd(acc, l)
             loss.go(rest, 1n + pos, V, c2, s2, 1n + n)
