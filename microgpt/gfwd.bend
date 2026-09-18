import Base
import ./gten.bend as G

# Pure single-doc transformer forward, mirroring model.bend op-for-op
# (same order => bit-identical results) but with no tape and no monad:
# everything is nested calls over +List<F32>, so later milestones can
# mark data-parallel calls with `!`. The loss loop double-calls
# (logits-path + cache-path) exactly like the scalar engine.
# Callee-before-caller throughout.

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

def sl0(+xs: +List<F32>) -> +List<F32>:
  List.take(&2, F32, xs, 4n)

def sl1(+xs: +List<F32>) -> +List<F32>:
  List.take(&2, F32, List.drop(&2, F32, xs, 4n), 4n)

def sl2(+xs: +List<F32>) -> +List<F32>:
  List.take(&2, F32, List.drop(&2, F32, xs, 8n), 4n)

def sl3(+xs: +List<F32>) -> +List<F32>:
  List.take(&2, F32, List.drop(&2, F32, xs, 12n), 4n)

def gscores(+ks: +List<+List<F32>>, +q: +List<F32>) -> +List<F32>:
  match ks:
    case Nil{}:
      Nil{}
    case Con{k, kt}:
      Con{(G.dot(q, k) * 0.5 : F32), gscores(kt, q)}

def gw.one(+w: F32, +v: +List<F32>, +acc: +List<F32>) -> +List<F32>:
  match v acc:
    case Nil{} Nil{}:
      Nil{}
    case Con{vj, vt} Con{aj, at}:
      Con{((w * vj : F32) + aj : F32), gw.one(w, vt, at)}
    case _ _:
      Nil{}

def gw.go(+ws: +List<F32>, +vs: +List<+List<F32>>, +acc: +List<F32>) -> +List<F32>:
  match ws vs:
    case Nil{} Nil{}:
      acc
    case Con{w, wt} Con{v, vt}:
      gw.go(wt, vt, gw.one(w, v, acc))
    case _ _:
      acc

def gwsum(+ws: +List<F32>, +vs: +List<+List<F32>>) -> +List<F32>:
  gw.go(ws, vs, [{0.0 : F32}, {0.0 : F32}, {0.0 : F32}, {0.0 : F32}])

 def grelu.map(+xs: +List<F32>) -> +List<F32>:
   match xs:
     case Nil{}:
       Nil{}
     case Con{x, xt}:
       Con{F32.max(x, {0.0 : F32}), grelu.map(xt)}

def cat4(+a: +List<F32>, +b: +List<F32>, +c: +List<F32>, +d: +List<F32>) -> +List<F32>:
  List.append(&2, F32, a, List.append(&2, F32, b, List.append(&2, F32, c, d)))

type PC is Data:
  PC{k0: +List<+List<F32>>, v0: +List<+List<F32>>, k1: +List<+List<F32>>, v1: +List<+List<F32>>, k2: +List<+List<F32>>, v2: +List<+List<F32>>, k3: +List<+List<F32>>, v3: +List<+List<F32>>}

def pc0() -> PC:
  PC{Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}}

 def ghead(+q: +List<F32>, +k: +List<F32>, +v: +List<F32>, +ks: +List<+List<F32>>, +vs: +List<+List<F32>>) -> +List<F32> & (+List<+List<F32>> & +List<+List<F32>>):
   (gwsum(G.softmax(gscores(Con{k, ks}, q)), Con{v, vs}), ((Con{k, ks}, Con{v, vs})))

 def gheads4(h3: +List<F32> & (+List<+List<F32>> & +List<+List<F32>>), +o0: +List<F32>, +o1: +List<F32>, +o2: +List<F32>, +nk0: +List<+List<F32>>, +nv0: +List<+List<F32>>, +nk1: +List<+List<F32>>, +nv1: +List<+List<F32>>, +nk2: +List<+List<F32>>, +nv2: +List<+List<F32>>) -> +List<F32> & PC:
   match h3:
     case (o3, (nk3, nv3)):
       (cat4(o0, o1, o2, o3), PC{nk0, nv0, nk1, nv1, nk2, nv2, nk3, nv3})

 def gheads3(r2: +List<F32> & (+List<+List<F32>> & +List<+List<F32>>), +o0: +List<F32>, +o1: +List<F32>, +nk0: +List<+List<F32>>, +nv0: +List<+List<F32>>, +nk1: +List<+List<F32>>, +nv1: +List<+List<F32>>, +q3: +List<F32>, +k3: +List<F32>, +v3: +List<F32>, +k3c: +List<+List<F32>>, +v3c: +List<+List<F32>>) -> +List<F32> & PC:
   match r2:
     case (o2, (nk2, nv2)):
       gheads4(ghead(q3, k3, v3, k3c, v3c), o0, o1, o2, nk0, nv0, nk1, nv1, nk2, nv2)

 def gheads2(r1: +List<F32> & (+List<+List<F32>> & +List<+List<F32>>), +o0: +List<F32>, +nk0: +List<+List<F32>>, +nv0: +List<+List<F32>>, +q2: +List<F32>, +k2: +List<F32>, +v2: +List<F32>, +k2c: +List<+List<F32>>, +v2c: +List<+List<F32>>, +q3: +List<F32>, +k3: +List<F32>, +v3: +List<F32>, +k3c: +List<+List<F32>>, +v3c: +List<+List<F32>>) -> +List<F32> & PC:
   match r1:
     case (o1, (nk1, nv1)):
       gheads3(ghead(q2, k2, v2, k2c, v2c), o0, o1, nk0, nv0, nk1, nv1, q3, k3, v3, k3c, v3c)

 def gheads1(r0: +List<F32> & (+List<+List<F32>> & +List<+List<F32>>), +q1: +List<F32>, +k1: +List<F32>, +v1: +List<F32>, +k1c: +List<+List<F32>>, +v1c: +List<+List<F32>>, +q2: +List<F32>, +k2: +List<F32>, +v2: +List<F32>, +k2c: +List<+List<F32>>, +v2c: +List<+List<F32>>, +q3: +List<F32>, +k3: +List<F32>, +v3: +List<F32>, +k3c: +List<+List<F32>>, +v3c: +List<+List<F32>>) -> +List<F32> & PC:
   match r0:
     case (o0, (nk0, nv0)):
       gheads2(ghead(q1, k1, v1, k1c, v1c), o0, nk0, nv0, q2, k2, v2, k2c, v2c, q3, k3, v3, k3c, v3c)

 def gheads(q0: +List<F32>, q1: +List<F32>, q2: +List<F32>, q3: +List<F32>, k0: +List<F32>, k1: +List<F32>, k2: +List<F32>, k3: +List<F32>, v0: +List<F32>, v1: +List<F32>, v2: +List<F32>, v3: +List<F32>, ac: PC) -> +List<F32> & PC:
   match ac:
     case PC{k0c, v0c, k1c, v1c, k2c, v2c, k3c, v3c}:
       gheads1(ghead(q0, k0, v0, k0c, v0c), q1, k1, v1, k1c, v1c, q2, k2, v2, k2c, v2c, q3, k3, v3, k3c, v3c)

def gstep.out(+x2: +List<F32>, +V: U32, +params: +List<F32>, cache: PC) -> +List<F32> & PC:
  (G.tmatvec(params, base.lm(V), U32.to_nat(V), 16n, x2), cache)

def gstep.mlp5(+m: +List<F32>, +x1: +List<F32>, +V: U32, +params: +List<F32>, cache: PC) -> +List<F32> & PC:
  gstep.out(G.tadd.zip(x1, m), V, params, cache)

def gstep.mlp4(+r: +List<F32>, +x1: +List<F32>, +V: U32, +params: +List<F32>, cache: PC) -> +List<F32> & PC:
  gstep.mlp5(G.tmatvec(params, base.fc2(V), 16n, 64n, r), x1, V, params, cache)

def gstep.mlp3(+h: +List<F32>, +x1: +List<F32>, +V: U32, +params: +List<F32>, cache: PC) -> +List<F32> & PC:
  gstep.mlp4(grelu.map(h), x1, V, params, cache)

def gstep.mlp2(+xm: +List<F32>, +x1: +List<F32>, +V: U32, +params: +List<F32>, cache: PC) -> +List<F32> & PC:
  gstep.mlp3(G.tmatvec(params, base.fc1(V), 64n, 16n, xm), x1, V, params, cache)

def gstep.mlp(+x1: +List<F32>, +V: U32, +params: +List<F32>, cache: PC) -> +List<F32> & PC:
  gstep.mlp2(G.rmsnorm(x1), x1, V, params, cache)

def gstep.res(+p: +List<F32>, +x0: +List<F32>, +V: U32, +params: +List<F32>, cache: PC) -> +List<F32> & PC:
  gstep.mlp(G.tadd.zip(x0, p), V, params, cache)
def gstep.proj2(+o: +List<F32>, ac: PC, +x0: +List<F32>, +V: U32, +params: +List<F32>) -> +List<F32> & PC:
  gstep.res(G.tmatvec(params, base.wo(V), 16n, 16n, o), x0, V, params, ac)

def gstep.proj(po: +List<F32> & PC, +x0: +List<F32>, +V: U32, +params: +List<F32>) -> +List<F32> & PC:
  match po:
    case (o, ac):
      gstep.proj2(o, ac, x0, V, params)


def gstep.attn3(+q: +List<F32>, +k: +List<F32>, +v: +List<F32>, +x0: +List<F32>, +V: U32, +params: +List<F32>, +cache: PC) -> +List<F32> & PC:
  gstep.proj(gheads(sl0(q), sl1(q), sl2(q), sl3(q), sl0(k), sl1(k), sl2(k), sl3(k), sl0(v), sl1(v), sl2(v), sl3(v), cache), x0, V, params)

 def gstep.attn2(+q: +List<F32>, +xn: +List<F32>, +r: +List<F32>, +V: U32, +params: +List<F32>, +cache: PC) -> +List<F32> & PC:
   gstep.attn3(q, G.tmatvec(params, base.wk(V), 16n, 16n, xn), G.tmatvec(params, base.wv(V), 16n, 16n, xn), r, V, params, cache)

def gstep.attn1(+xn: +List<F32>, +r: +List<F32>, +V: U32, +params: +List<F32>, +cache: PC) -> +List<F32> & PC:
  gstep.attn2(G.tmatvec(params, base.wq(V), 16n, 16n, xn), xn, r, V, params, cache)

def gstep.attn(+r: +List<F32>, +V: U32, +params: +List<F32>, +cache: PC) -> +List<F32> & PC:
  gstep.attn1(G.rmsnorm(r), r, V, params, cache)

def gstep.emb2(+te: +List<F32>, +pe: +List<F32>, +V: U32, +params: +List<F32>, +cache: PC) -> +List<F32> & PC:
  gstep.attn(G.tadd.zip(te, pe), V, params, cache)

def gstep.emb(+te: +List<F32>, +pos: Nat, +V: U32, +params: +List<F32>, +cache: PC) -> +List<F32> & PC:
  gstep.emb2(te, G.tgather(params, base.wpe(V), pos, 16n), V, params, cache)

def gstep(+tok: U32, pos: Nat, +V: U32, +params: +List<F32>, +cache: PC) -> +List<F32> & PC:
  gstep.emb(G.tgather(params, 0, U32.to_nat(tok), 16n), pos, V, params, cache)

def gstep.lg(st: +List<F32> & PC) -> +List<F32>:
  match st:
    case (logits, cache):
      logits

def gstep.cc(st: +List<F32> & PC) -> PC:
  match st:
    case (logits, cache):
      cache

def gstep.logits(+tok: U32, +pos: Nat, +V: U32, +params: +List<F32>, +cache: PC) -> +List<F32>:
  gstep.lg(gstep(tok, pos, V, params, cache))

def gstep.cache(+tok: U32, +pos: Nat, +V: U32, +params: +List<F32>, +cache: PC) -> PC:
  gstep.cc(gstep(tok, pos, V, params, cache))

def gstep.term(+lg: +List<F32>, +tgt: U32) -> F32:
  F32.neg(F32.log(G.tnth(G.softmax(lg), U32.to_nat(tgt))))

def gloss.go(+toks: +List<U32>, +pos: Nat, +V: U32, +params: +List<F32>, +cache: PC, acc: F32, n: Nat) -> F32 & Nat:
  match toks:
    case Nil{}:
      (acc, n)
    case Con{tok, rest}:
      match rest:
        case Nil{}:
          (acc, n)
        case Con{tgt, tail}:
          gloss.go(rest, 1n + pos, V, params, gstep.cache(tok, pos, V, params, cache), (acc + gstep.term(gstep.logits(tok, pos, V, params, cache), tgt) : F32), 1n + n)
