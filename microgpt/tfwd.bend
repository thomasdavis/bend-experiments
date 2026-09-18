import Base
import ./gten.bend as G
import ./gfwd.bend as F

# Tensor-level recording forward (M2'b2 final shape). Values computed by
# the SAME gten fns as M1 (=> bit-identical loss); each vector op appends
# ONE tensor node. Every forward fn takes the cache IN and returns HB
# (out-FV3 + updated cache), so caches advance with zero splits in loops.
# The loss loop (fill/iter/go) mirrors the TP topology with GS3 states.

type TN is Data:
  TN{op: U32, a: U32, b: U32, c: U32, xs: +List<F32>, ys: +List<F32>, ids: +List<U32>}

type FV3 is Data:
  FV3{xs: +List<F32>, tp: +List<TN>, nx: U32, id: U32}

type KV is Data:
  KV{v: +List<F32>, id: U32}

type PC3 is Data:
  PC3{k0: +List<KV>, v0: +List<KV>, k1: +List<KV>, v1: +List<KV>, k2: +List<KV>, v2: +List<KV>, k3: +List<KV>, v3: +List<KV>}

type FH is Data:
  FH{xs: +List<F32>, tp: +List<TN>, nx: U32, id: U32, kc: +List<KV>, vc: +List<KV>}

type HB is Data:
  HB{out: FV3, ac: PC3}

type GS3 is Data:
  GS3{toks: +List<U32>, pos: Nat, cache: PC3, acc: F32, n: Nat, tp: +List<TN>, nx: U32, V: U32, params: +List<F32>}

def tadd.rec(+xs: +List<F32>, +ys: +List<F32>, +xaid: U32, +ybid: U32, +tp: +List<TN>, +nx: U32) -> FV3:
  FV3{G.tadd.zip(xs, ys), Con{TN{1, xaid, ybid, 0, xs, ys, Nil{}}, tp}, (nx + 1 : U32), nx}

def tmul.rec(+xs: +List<F32>, +ys: +List<F32>, +xaid: U32, +ybid: U32, +tp: +List<TN>, +nx: U32) -> FV3:
  FV3{G.tmul.zip(xs, ys), Con{TN{2, xaid, ybid, 0, xs, ys, Nil{}}, tp}, (nx + 1 : U32), nx}

def tlin.rec(+tab: +List<F32>, +base: U32, +rows: Nat, +cols: Nat, +x: +List<F32>, +xaid: U32, +tp: +List<TN>, +nx: U32) -> FV3:
  FV3{G.tmatvec(tab, base, rows, cols, x), Con{TN{3, base, U32.from_nat(rows), U32.from_nat(cols), x, Nil{}, Nil{}}, tp}, (nx + 1 : U32), nx}

def trms.rec(+xs: +List<F32>, +xaid: U32, +tp: +List<TN>, +nx: U32) -> FV3:
  FV3{G.rmsnorm(xs), Con{TN{4, xaid, 0, 0, xs, Nil{}, Nil{}}, tp}, (nx + 1 : U32), nx}

def tsm.rec(+xs: +List<F32>, +xaid: U32, +tp: +List<TN>, +nx: U32) -> FV3:
  FV3{G.softmax(xs), Con{TN{5, xaid, 0, 0, xs, Nil{}, Nil{}}, tp}, (nx + 1 : U32), nx}

def trelu.rec(+xs: +List<F32>, +xaid: U32, +tp: +List<TN>, +nx: U32) -> FV3:
  FV3{F.grelu.map(xs), Con{TN{6, xaid, 0, 0, xs, Nil{}, Nil{}}, tp}, (nx + 1 : U32), nx}

def tgat.rec(+tab: +List<F32>, +base: U32, +r: Nat, +C: Nat, +tp: +List<TN>, +nx: U32) -> FV3:
  FV3{G.tgather(tab, base, r, C), Con{TN{8, base, U32.from_nat(r), U32.from_nat(C), Nil{}, Nil{}, Nil{}}, tp}, (nx + 1 : U32), nx}

def tconst.rec(+c: F32, +tp: +List<TN>, +nx: U32) -> FV3:
  FV3{Con{c, Nil{}}, Con{TN{7, 0, 0, 0, Nil{}, Nil{}, Nil{}}, tp}, (nx + 1 : U32), nx}

def pscores.go(+ks: +List<KV>, +q: +List<F32>) -> +List<F32>:
  match ks:
    case Nil{}:
      Nil{}
    case Con{k, kt}:
      match k:
        case KV{v, id}:
          Con{(G.dot(q, v) * 0.5 : F32), pscores.go(kt, q)}

def kvecs(+ks: +List<KV>) -> +List<+List<F32>>:
  match ks:
    case Nil{}:
      Nil{}
    case Con{k, kt}:
      match k:
        case KV{v, id}:
          Con{v, kvecs(kt)}

def kids(+ks: +List<KV>) -> +List<U32>:
  match ks:
    case Nil{}:
      Nil{}
    case Con{k, kt}:
      match k:
        case KV{v, id}:
          Con{id, kids(kt)}

def concatv(+vss: +List<+List<F32>>) -> +List<F32>:
  match vss:
    case Nil{}:
      Nil{}
    case Con{v, t}:
      List.append(&2, F32, v, concatv(t))

def klen(+ks: +List<KV>) -> U32:
  U32.from_nat(List.length(&2, KV, ks))

 def pscores(+ks: +List<KV>, +q: +List<F32>) -> +List<F32>:
   pscores.go(ks, q)

def tscores.rec(+q: +List<F32>, +qid: U32, +ks: +List<KV>, +tp: +List<TN>, +nx: U32) -> FV3:
  FV3{pscores(ks, q), Con{TN{9, qid, klen(ks), 0, q, concatv(kvecs(ks)), kids(ks)}, tp}, (nx + 1 : U32), nx}

def pwsum.go(+ws: +List<F32>, +vs: +List<KV>, +acc: +List<F32>) -> +List<F32>:
  match ws vs:
    case Nil{} Nil{}:
      acc
    case Con{w, wt} Con{v, vt}:
      match v:
        case KV{vv, id}:
          pwsum.go(wt, vt, G.tadd.zip(acc, G.tscale.map(vv, w)))
    case _ _:
      acc

def pwsum(+ws: +List<F32>, +vs: +List<KV>) -> +List<F32>:
  pwsum.go(ws, vs, [{0.0 : F32}, {0.0 : F32}, {0.0 : F32}, {0.0 : F32}])

def twsum.rec(+w: +List<F32>, +wid: U32, +vs: +List<KV>, +tp: +List<TN>, +nx: U32) -> FV3:
  FV3{pwsum(w, vs), Con{TN{10, wid, klen(vs), 0, w, concatv(kvecs(vs)), kids(vs)}, tp}, (nx + 1 : U32), nx}

def pc30() -> PC3:
  PC3{Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}}

def thead.wx(ws: FV3, +k: +List<F32>, +kid: U32, +ks: +List<KV>, +v: +List<F32>, +vid: U32, +vs: +List<KV>) -> FH:
  match ws:
    case FV3{o, tp4, nx4, oid}:
      FH{o, tp4, nx4, oid, Con{KV{k, kid}, ks}, Con{KV{v, vid}, vs}}

def thead.ws(sm: FV3, +v: +List<F32>, +vid: U32, +vs: +List<KV>, +k: +List<F32>, +kid: U32, +ks: +List<KV>) -> FH:
  match sm:
    case FV3{w, tp3, nx3, wid}:
      thead.wx(twsum.rec(w, wid, Con{KV{v, vid}, vs}, tp3, nx3), k, kid, ks, v, vid, vs)

def thead.sm(sc: FV3, +v: +List<F32>, +vid: U32, +vs: +List<KV>, +k: +List<F32>, +kid: U32, +ks: +List<KV>) -> FH:
  match sc:
    case FV3{scv, tp2, nx2, scid}:
      thead.ws(tsm.rec(scv, scid, tp2, nx2), v, vid, vs, k, kid, ks)

def thead.sw(sc: FV3, +v: +List<F32>, +vid: U32, +vs: +List<KV>, +k: +List<F32>, +kid: U32, +ks: +List<KV>) -> FH:
  match sc:
    case FV3{scv, tp2, nx2, scid}:
      thead.sm(tsm.rec(scv, scid, tp2, nx2), v, vid, vs, k, kid, ks)
def thead(+q: +List<F32>, +qid: U32, +k: +List<F32>, +kid: U32, +v: +List<F32>, +vid: U32, +ks: +List<KV>, +vs: +List<KV>, +tp: +List<TN>, +nx: U32) -> FH:
  thead.sw(tscores.rec(q, qid, Con{KV{k, kid}, ks}, tp, nx), v, vid, vs, k, kid, ks)


def theads4(h3: FH, +o0: +List<F32>, +o1: +List<F32>, +o2: +List<F32>, +c0k: +List<KV>, +c0v: +List<KV>, +c1k: +List<KV>, +c1v: +List<KV>, +c2k: +List<KV>, +c2v: +List<KV>) -> HB:
  match h3:
    case FH{o3, tp4, nx4, oid, k3c, v3c}:
      HB{FV3{F.cat4(o0, o1, o2, o3), tp4, nx4, oid}, PC3{c0k, c0v, c1k, c1v, c2k, c2v, k3c, v3c}}

def theads3(r2: FH, +o0: +List<F32>, +o1: +List<F32>, +c0k: +List<KV>, +c0v: +List<KV>, +c1k: +List<KV>, +c1v: +List<KV>, +q3: +List<F32>, +qid3: U32, +k3: +List<F32>, +kid3: U32, +v3: +List<F32>, +vid3: U32, +k3c: +List<KV>, +v3c: +List<KV>) -> HB:
  match r2:
    case FH{o2, tp2, nx2, oid2, k2c, v2c}:
      theads4(thead(q3, qid3, k3, kid3, v3, vid3, k3c, v3c, tp2, nx2), o0, o1, o2, c0k, c0v, c1k, c1v, k2c, v2c)

def theads2(r1: FH, +o0: +List<F32>, +c0k: +List<KV>, +c0v: +List<KV>, +q2: +List<F32>, +qid2: U32, +k2: +List<F32>, +kid2: U32, +v2: +List<F32>, +vid2: U32, +k2c: +List<KV>, +v2c: +List<KV>, +q3: +List<F32>, +qid3: U32, +k3: +List<F32>, +kid3: U32, +v3: +List<F32>, +vid3: U32, +k3c: +List<KV>, +v3c: +List<KV>) -> HB:
  match r1:
    case FH{o1, tp1, nx1, oid1, k1c, v1c}:
      theads3(thead(q2, qid2, k2, kid2, v2, vid2, k2c, v2c, tp1, nx1), o0, o1, c0k, c0v, k1c, v1c, q3, qid3, k3, kid3, v3, vid3, k3c, v3c)

def theads1(r0: FH, +q1: +List<F32>, +qid1: U32, +k1: +List<F32>, +kid1: U32, +v1: +List<F32>, +vid1: U32, +k1c: +List<KV>, +v1c: +List<KV>, +q2: +List<F32>, +qid2: U32, +k2: +List<F32>, +kid2: U32, +v2: +List<F32>, +vid2: U32, +k2c: +List<KV>, +v2c: +List<KV>, +q3: +List<F32>, +qid3: U32, +k3: +List<F32>, +kid3: U32, +v3: +List<F32>, +vid3: U32, +k3c: +List<KV>, +v3c: +List<KV>) -> HB:
  match r0:
    case FH{o0, tp0, nx0, oid0, k0c, v0c}:
      theads2(thead(q1, qid1, k1, kid1, v1, vid1, k1c, v1c, tp0, nx0), o0, k0c, v0c, q2, qid2, k2, kid2, v2, vid2, k2c, v2c, q3, qid3, k3, kid3, v3, vid3, k3c, v3c)

 def theads(+q0: +List<F32>, +qid0: U32, +k0: +List<F32>, +kid0: U32, +v0: +List<F32>, +vid0: U32, +c0k: +List<KV>, +c0v: +List<KV>, +q1: +List<F32>, +qid1: U32, +k1: +List<F32>, +kid1: U32, +v1: +List<F32>, +vid1: U32, +c1k: +List<KV>, +c1v: +List<KV>, +q2: +List<F32>, +qid2: U32, +k2: +List<F32>, +kid2: U32, +v2: +List<F32>, +vid2: U32, +c2k: +List<KV>, +c2v: +List<KV>, +q3: +List<F32>, +qid3: U32, +k3: +List<F32>, +kid3: U32, +v3: +List<F32>, +vid3: U32, +c3k: +List<KV>, +c3v: +List<KV>, +tp: +List<TN>, +nx: U32) -> HB:
   theads1(thead(q0, qid0, k0, kid0, v0, vid0, c0k, c0v, tp, nx), q1, qid1, k1, kid1, v1, vid1, c1k, c1v, q2, qid2, k2, kid2, v2, vid2, c2k, c2v, q3, qid3, k3, kid3, v3, vid3, c3k, c3v)
 def gstep.out(x2: FV3, +V: U32, +params: +List<F32>, cache: PC3) -> HB:
   match x2:
     case FV3{+x2v, tp, nx, +x2id}:
       HB{tlin.rec(params, F.base.lm(V), U32.to_nat(V), 16n, x2v, x2id, tp, nx), cache}

 def gstep.mlp5(m: FV3, +x1v: +List<F32>, +x1id: U32, +V: U32, +params: +List<F32>, cache: PC3) -> HB:
   match m:
     case FV3{+mv, tp, nx, +mid}:
       gstep.out(tadd.rec(x1v, mv, x1id, mid, tp, nx), V, params, cache)

 def gstep.mlp4(r: FV3, +x1v: +List<F32>, +x1id: U32, +V: U32, +params: +List<F32>, cache: PC3) -> HB:
   match r:
     case FV3{+rv, tp, nx, +rid}:
       gstep.mlp5(tlin.rec(params, F.base.fc2(V), 16n, 64n, rv, rid, tp, nx), x1v, x1id, V, params, cache)

 def gstep.mlp3(h: FV3, +x1v: +List<F32>, +x1id: U32, +V: U32, +params: +List<F32>, cache: PC3) -> HB:
   match h:
     case FV3{+hv, tp, nx, +hid}:
       gstep.mlp4(trelu.rec(hv, hid, tp, nx), x1v, x1id, V, params, cache)

 def gstep.mlp2(xm: FV3, +x1v: +List<F32>, +x1id: U32, +V: U32, +params: +List<F32>, cache: PC3) -> HB:
   match xm:
     case FV3{+xmv, tp, nx, +xmid}:
       gstep.mlp3(tlin.rec(params, F.base.fc1(V), 64n, 16n, xmv, xmid, tp, nx), x1v, x1id, V, params, cache)

 def gstep.mlp(x1: FV3, +V: U32, +params: +List<F32>, cache: PC3) -> HB:
   match x1:
     case FV3{+x1v, tp, nx, +x1id}:
       gstep.mlp2(trms.rec(x1v, x1id, tp, nx), x1v, x1id, V, params, cache)

 def gstep.res(p: FV3, +x0v: +List<F32>, +x0id: U32, +V: U32, +params: +List<F32>, cache: PC3) -> HB:
   match p:
     case FV3{+pv, tp, nx, +pid}:
       gstep.mlp(tadd.rec(x0v, pv, x0id, pid, tp, nx), V, params, cache)

 def gstep.proj2(+o: +List<F32>, +oid: U32, ac: PC3, +x0v: +List<F32>, +x0id: U32, +V: U32, +params: +List<F32>, +tp: +List<TN>, +nx: U32) -> HB:
   gstep.res(tlin.rec(params, F.base.wo(V), 16n, 16n, o, oid, tp, nx), x0v, x0id, V, params, ac)

 def gstep.projA(out: FV3, ac: PC3, +x0v: +List<F32>, +x0id: U32, +V: U32, +params: +List<F32>) -> HB:
   match out:
     case FV3{+ov, tp, nx, +oid}:
       gstep.proj2(ov, oid, ac, x0v, x0id, V, params, tp, nx)


 def gstep.proj(po: HB, +x0v: +List<F32>, +x0id: U32, +V: U32, +params: +List<F32>) -> HB:
   match po:
     case HB{out, ac}:
       gstep.projA(out, ac, x0v, x0id, V, params)


 def F.sl2(+xs: +List<F32>) -> +List<F32>:
   List.take(&2, F32, List.drop(&2, F32, xs, 8n), 4n)

 def F.sl3(+xs: +List<F32>) -> +List<F32>:
   List.take(&2, F32, List.drop(&2, F32, xs, 12n), 4n)

 def gstep.attn3(+q: +List<F32>, +qid: U32, +k: +List<F32>, +kid: U32, +v: +List<F32>, +vid: U32, +x0v: +List<F32>, +x0id: U32, +V: U32, +params: +List<F32>, cache: PC3, +tp: +List<TN>, +nx: U32) -> HB:
   match cache:
     case PC3{k0c, v0c, k1c, v1c, k2c, v2c, k3c, v3c}:
       gstep.proj(theads(F.sl0(q), qid, F.sl0(k), kid, F.sl0(v), vid, k0c, v0c, F.sl1(q), qid, F.sl1(k), kid, F.sl1(v), vid, k1c, v1c, F.sl2(q), qid, F.sl2(k), kid, F.sl2(v), vid, k2c, v2c, F.sl3(q), qid, F.sl3(k), kid, F.sl3(v), vid, k3c, v3c, tp, nx), x0v, x0id, V, params)

 def gstep.attn1d(v: FV3, +qv: +List<F32>, +qid: U32, +kv: +List<F32>, +kid: U32, +xv: +List<F32>, +xid: U32, +V: U32, +params: +List<F32>, +cache: PC3) -> HB:
   match v:
     case FV3{+vv, tp, nx, +vid}:
       gstep.attn3(qv, qid, kv, kid, vv, vid, xv, xid, V, params, cache, tp, nx)

 def gstep.attn1c(k: FV3, +qv: +List<F32>, +qid: U32, +xnv: +List<F32>, +xnid: U32, +xv: +List<F32>, +xid: U32, +V: U32, +params: +List<F32>, +cache: PC3) -> HB:
   match k:
     case FV3{+kv, tp, nx, +kid}:
       gstep.attn1d(tlin.rec(params, F.base.wv(V), 16n, 16n, xnv, xnid, tp, nx), qv, qid, kv, kid, xv, xid, V, params, cache)

 def gstep.attn1b(q: FV3, +xnv: +List<F32>, +xnid: U32, +xv: +List<F32>, +xid: U32, +V: U32, +params: +List<F32>, +cache: PC3) -> HB:
   match q:
     case FV3{+qv, tp, nx, +qid}:
       gstep.attn1c(tlin.rec(params, F.base.wk(V), 16n, 16n, xnv, xnid, tp, nx), qv, qid, xnv, xnid, xv, xid, V, params, cache)

 def gstep.attn1(st: FV3, +xv: +List<F32>, +xid: U32, +V: U32, +params: +List<F32>, +cache: PC3) -> HB:
   match st:
     case FV3{+xnv, tp, nx, +xnid}:
       gstep.attn1b(tlin.rec(params, F.base.wq(V), 16n, 16n, xnv, xnid, tp, nx), xnv, xnid, xv, xid, V, params, cache)

 def gstep.attn(x: FV3, +V: U32, +params: +List<F32>, +cache: PC3) -> HB:
   match x:
     case FV3{+xv, tp, nx, +xid}:
       gstep.attn1(trms.rec(xv, xid, tp, nx), xv, xid, V, params, cache)

 def gstep.emb2(pe: FV3, +tev: +List<F32>, +teid: U32, +V: U32, +params: +List<F32>, +cache: PC3) -> HB:
   match pe:
     case FV3{+pev, tp, nx, +peid}:
       gstep.attn(tadd.rec(tev, pev, teid, peid, tp, nx), V, params, cache)

 def gstep.emb(te: FV3, +pos: Nat, +V: U32, +params: +List<F32>, +cache: PC3) -> HB:
   match te:
     case FV3{+tev, tp, nx, +teid}:
       gstep.emb2(tgat.rec(params, F.base.wpe(V), pos, 16n, tp, nx), tev, teid, V, params, cache)

 def gstep(+tok: U32, +pos: Nat, +V: U32, +params: +List<F32>, +cache: PC3, +tp: +List<TN>, +nx: U32) -> HB:
   gstep.emb(tgat.rec(params, 0, U32.to_nat(tok), 16n, tp, nx), pos, V, params, cache)

 def tterm(+lg: +List<F32>, +tgt: U32) -> F32:
   F32.neg(F32.log(G.tnth(G.softmax(lg), U32.to_nat(tgt))))

 def tloss.go(+rest: +List<U32>, hb: HB, +tgt: U32, +pos: Nat, +V: U32, +params: +List<F32>, +acc: F32, +n: Nat) -> GS3:
   match rest:
     case Nil{}:
       match hb:
         case HB{out, ac}:
           match out:
             case FV3{lg, tp2, nx2, id}:
               GS3{rest, pos, ac, (acc + tterm(lg, tgt) : F32), n, tp2, nx2, V, params}
     case Con{tok2, rest2}:
       match rest2:
         case Nil{}:
           match hb:
             case HB{out, ac}:
               match out:
                 case FV3{lg, tp2, nx2, id}:
                   GS3{rest, pos, ac, (acc + tterm(lg, tgt) : F32), n, tp2, nx2, V, params}
         case Con{tgt2, tail2}:
           match hb:
             case HB{out, ac}:
               match out:
                 case FV3{lg, tp2, nx2, id}:
                   tloss.go(rest2, gstep(tok2, 1n + pos, V, params, ac, tp2, nx2), tgt2, 1n + pos, V, params, (acc + tterm(lg, tgt) : F32), 1n + n)

 def tloss.entry(+toks: +List<U32>, +V: U32, +params: +List<F32>) -> GS3:
   match toks:
     case Nil{}:
       GS3{toks, 0n, pc30(), {0.0 : F32}, 0n, Nil{}, 0, V, params}
     case Con{tok, rest}:
       match rest:
         case Nil{}:
           GS3{toks, 0n, pc30(), {0.0 : F32}, 0n, Nil{}, 0, V, params}
         case Con{tgt, tail}:
           tloss.go(rest, gstep(tok, 0n, V, params, pc30(), Nil{}, 0), tgt, 0n, V, params, {0.0 : F32}, 0n)
# touch
