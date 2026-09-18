import Base

# M2'a: explicit-state pure reverse sweep (no monad, no arrays).
# Tape is a +List<PN> (newest-first); node grads travel in a parallel
# +List<F32>; param grads are two F32s. Everything bundles in the SW
# record (pairs cannot hold &2-kind variables). Loops match shrinking
# lists and thread SW whole; splits happen only in acyclic helpers.

type PN is Data:
  PN{op: U32, a: U32, b: U32, k: F32, data: F32}

type SW is Data:
  SW{gs: +List<F32>, p0: F32, p1: F32}

def pdata.nd(nd: PN) -> F32:
  match nd:
    case PN{op, a, b, k, d}:
      d

def pdata.nth(+tp: +List<PN>, id: Nat) -> F32:
  match tp id:
    case Nil{} _:
      {0.0 : F32}
    case Con{nd, t} 0n:
      pdata.nd(nd)
    case Con{nd, t} 1n+p:
      pdata.nth(t, p)

def pdata(+tp: +List<PN>, id: Nat) -> F32:
  pdata.nth(List.reverse(&2, PN, tp), id)

def paddc.go(+gs: +List<F32>, id: Nat, +x: F32) -> +List<F32>:
  match gs id:
    case Nil{} _:
      Nil{}
    case Con{g, t} 0n:
      Con{(g + x : F32), t}
    case Con{g, t} 1n+p:
      Con{g, paddc.go(t, p, x)}

def paddc(+gs: +List<F32>, id: Nat, +x: F32) -> +List<F32>:
  List.reverse(&2, F32, paddc.go(List.reverse(&2, F32, gs), id, x))

def sweep.add(+gs: +List<F32>, +a: U32, +b: U32, +g: F32, +p0: F32, +p1: F32) -> SW:
  SW{paddc(paddc(gs, U32.to_nat(a), g), U32.to_nat(b), g), p0, p1}

def sweep.mul1(+db: F32, +da: F32, +a: U32, +b: U32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
  SW{paddc(paddc(gs, U32.to_nat(a), (db * g : F32)), U32.to_nat(b), (da * g : F32)), p0, p1}

def sweep.mulB(+db: F32, +da: F32, +a: U32, +b: U32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
  sweep.mul1(db, da, a, b, g, gs, p0, p1)

def sweep.mulA(+da: F32, +tp: +List<PN>, +a: U32, +b: U32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
  sweep.mulB(pdata(tp, U32.to_nat(b)), da, a, b, g, gs, p0, p1)

def sweep.mul(+tp: +List<PN>, +a: U32, +b: U32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
  sweep.mulA(pdata(tp, U32.to_nat(a)), tp, a, b, g, gs, p0, p1)

def sweep.pow1(+da: F32, +a: U32, +k: F32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
  SW{paddc(gs, U32.to_nat(a), (k * F32.pow(da, (k - 1.0 : F32)) * g : F32)), p0, p1}

def sweep.pow(+tp: +List<PN>, +a: U32, +k: F32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
  sweep.pow1(pdata(tp, U32.to_nat(a)), a, k, g, gs, p0, p1)

def sweep.log1(+da: F32, +a: U32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
  SW{paddc(gs, U32.to_nat(a), (g / da : F32)), p0, p1}

def sweep.log(+tp: +List<PN>, +a: U32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
  sweep.log1(pdata(tp, U32.to_nat(a)), a, g, gs, p0, p1)

def sweep.exp(+d: F32, +a: U32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
  SW{paddc(gs, U32.to_nat(a), (d * g : F32)), p0, p1}

def sweep.relu(+d: F32, +a: U32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
  SW{paddc(gs, U32.to_nat(a), Bool.pick(F32, F32.is_gt(d, {0.0 : F32}), g, {0.0 : F32})), p0, p1}

 def sweep.leaf.go(z: Bool, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
   match z:
     case True{}:
       SW{gs, (p0 + g : F32), p1}
     case False{}:
       SW{gs, p0, (p1 + g : F32)}

 def sweep.leaf(+a: U32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
   sweep.leaf.go(U32.is_eq(a, 0), g, gs, p0, p1)

 def sweep.noop(+gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
   SW{gs, p0, p1}

 def sweep.is7(z: Bool, +a: U32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
   match z:
     case True{}:
       sweep.noop(gs, p0, p1)
     case False{}:
       sweep.leaf(a, g, gs, p0, p1)

def sweep.is6(z: Bool, +op: U32, +a: U32, +b: U32, +k: F32, +d: F32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32) -> SW:
  match z:
    case True{}:
      sweep.relu(d, a, g, gs, p0, p1)
    case False{}:
      sweep.is7(U32.is_eq(op, 7), a, g, gs, p0, p1)

def sweep.is5(z: Bool, +op: U32, +a: U32, +b: U32, +k: F32, +d: F32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32, +tp: +List<PN>) -> SW:
  match z:
    case True{}:
      sweep.exp(d, a, g, gs, p0, p1)
    case False{}:
      sweep.is6(U32.is_eq(op, 6), op, a, b, k, d, g, gs, p0, p1)

def sweep.is4(z: Bool, +op: U32, +a: U32, +b: U32, +k: F32, +d: F32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32, +tp: +List<PN>) -> SW:
  match z:
    case True{}:
      sweep.log(tp, a, g, gs, p0, p1)
    case False{}:
      sweep.is5(U32.is_eq(op, 5), op, a, b, k, d, g, gs, p0, p1, tp)

def sweep.is3(z: Bool, +op: U32, +a: U32, +b: U32, +k: F32, +d: F32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32, +tp: +List<PN>) -> SW:
  match z:
    case True{}:
      sweep.pow(tp, a, k, g, gs, p0, p1)
    case False{}:
      sweep.is4(U32.is_eq(op, 4), op, a, b, k, d, g, gs, p0, p1, tp)

def sweep.is2(z: Bool, +op: U32, +a: U32, +b: U32, +k: F32, +d: F32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32, +tp: +List<PN>) -> SW:
  match z:
    case True{}:
      sweep.mul(tp, a, b, g, gs, p0, p1)
    case False{}:
      sweep.is3(U32.is_eq(op, 3), op, a, b, k, d, g, gs, p0, p1, tp)

def sweep.is1(z: Bool, +op: U32, +a: U32, +b: U32, +k: F32, +d: F32, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32, +tp: +List<PN>) -> SW:
  match z:
    case True{}:
      sweep.add(gs, a, b, g, p0, p1)
    case False{}:
      sweep.is2(U32.is_eq(op, 2), op, a, b, k, d, g, gs, p0, p1, tp)

def sweep.one(nd: PN, +g: F32, +gs: +List<F32>, +p0: F32, +p1: F32, +tp: +List<PN>) -> SW:
  match nd:
    case PN{+op, +a, +b, +k, +d}:
      sweep.is1(U32.is_eq(op, 1), op, a, b, k, d, g, gs, p0, p1, tp)

def sweep.go(+tp: +List<PN>, acc: SW, +tpall: +List<PN>) -> SW:
  match tp:
    case Nil{}:
      acc
    case Con{nd, t}:
      match acc:
        case SW{gs, p0, p1}:
          match gs:
            case Nil{}:
              acc
            case Con{g, gt}:
              sweep.go(t, sweep.one(nd, g, gt, p0, p1, tpall), tpall)
 type ES is Type:
   ES{tp: +List<PN>, n: U32, dat: Array<F32>, grd: Array<F32>, pgd: Array<F32>}

 type FR is Type:
   FR{st: ES, id: U32}

 def esnew(+tp: +List<PN>, +n: U32, dat: Array<F32>, grd: Array<F32>, pgd: Array<F32>) -> ES:
   ES{tp, n, dat, grd, pgd}
 def pleaf(+pi: U32, +pd: F32, +tp: +List<PN>, +n: U32, dat: Array<F32>, grd: Array<F32>, pgd: Array<F32>) -> FR:
   FR{esnew(Con{PN{0, pi, 0, {0.0 : F32}, pd}, tp}, (n + 1 : U32), Array.set(F32, dat, n, pd), Array.set(F32, grd, n, {0.0 : F32}), pgd), n}

 def pconst(+c: F32, +tp: +List<PN>, +n: U32, dat: Array<F32>, grd: Array<F32>, pgd: Array<F32>) -> FR:
   FR{esnew(Con{PN{7, 0, 0, {0.0 : F32}, c}, tp}, (n + 1 : U32), Array.set(F32, dat, n, c), Array.set(F32, grd, n, {0.0 : F32}), pgd), n}

 def padd(+x: U32, +y: U32, +xd: F32, +yd: F32, +tp: +List<PN>, +n: U32, dat: Array<F32>, grd: Array<F32>, pgd: Array<F32>) -> FR:
   FR{esnew(Con{PN{1, x, y, {0.0 : F32}, (xd + yd : F32)}, tp}, (n + 1 : U32), Array.set(F32, dat, n, (xd + yd : F32)), Array.set(F32, grd, n, {0.0 : F32}), pgd), n}

 def pmul(+x: U32, +y: U32, +xd: F32, +yd: F32, +tp: +List<PN>, +n: U32, dat: Array<F32>, grd: Array<F32>, pgd: Array<F32>) -> FR:
   FR{esnew(Con{PN{2, x, y, {0.0 : F32}, (xd * yd : F32)}, tp}, (n + 1 : U32), Array.set(F32, dat, n, (xd * yd : F32)), Array.set(F32, grd, n, {0.0 : F32}), pgd), n}

 def ppow(+x: U32, +xd: F32, +k: F32, +tp: +List<PN>, +n: U32, dat: Array<F32>, grd: Array<F32>, pgd: Array<F32>) -> FR:
   FR{esnew(Con{PN{3, x, 0, k, F32.pow(xd, k)}, tp}, (n + 1 : U32), Array.set(F32, dat, n, F32.pow(xd, k)), Array.set(F32, grd, n, {0.0 : F32}), pgd), n}

 def plog(+x: U32, +xd: F32, +tp: +List<PN>, +n: U32, dat: Array<F32>, grd: Array<F32>, pgd: Array<F32>) -> FR:
   FR{esnew(Con{PN{4, x, 0, {0.0 : F32}, F32.log(xd)}, tp}, (n + 1 : U32), Array.set(F32, dat, n, F32.log(xd)), Array.set(F32, grd, n, {0.0 : F32}), pgd), n}

 def pexp(+x: U32, +xd: F32, +tp: +List<PN>, +n: U32, dat: Array<F32>, grd: Array<F32>, pgd: Array<F32>) -> FR:
   FR{esnew(Con{PN{5, x, 0, {0.0 : F32}, F32.exp(xd)}, tp}, (n + 1 : U32), Array.set(F32, dat, n, F32.exp(xd)), Array.set(F32, grd, n, {0.0 : F32}), pgd), n}

 def prelu(+x: U32, +xd: F32, +tp: +List<PN>, +n: U32, dat: Array<F32>, grd: Array<F32>, pgd: Array<F32>) -> FR:
   FR{esnew(Con{PN{6, x, 0, {0.0 : F32}, F32.max(xd, {0.0 : F32})}, tp}, (n + 1 : U32), Array.set(F32, dat, n, F32.max(xd, {0.0 : F32})), Array.set(F32, grd, n, {0.0 : F32}), pgd), n}

 def dget(a: Array<F32>, +i: U32) -> Array<F32> & F32:
   Array.get(F32, a, i)
 def smoke.prt(p: Array<F32> & F32) -> F32:
   match p:
     case (dat, v):
       v

 def smoke.rd(fr: FR) -> F32:
   match fr:
     case FR{st, f}:
       match st:
         case ES{tp, n, dat, grd, pgd}:
           smoke.prt(dget(dat, 3))

 def smoke.ad(fr: FR) -> F32:
   match fr:
     case FR{st, m}:
       match st:
         case ES{tp, n, dat, grd, pgd}:
           smoke.rd(padd(2, 0, {6.0 : F32}, {2.0 : F32}, tp, n, dat, grd, pgd))

 def smoke.mu(fr: FR) -> F32:
   match fr:
     case FR{st, y}:
       match st:
         case ES{tp, n, dat, grd, pgd}:
           smoke.ad(pmul(0, 1, {2.0 : F32}, {3.0 : F32}, tp, n, dat, grd, pgd))

 def smoke.ly(fr: FR) -> F32:
   match fr:
     case FR{st, x}:
       match st:
         case ES{tp, n, dat, grd, pgd}:
           smoke.mu(pleaf(1, {3.0 : F32}, tp, n, dat, grd, pgd))

 def smoke() -> F32:
   smoke.ly(pleaf(0, {2.0 : F32}, Nil{}, 0, Array.new(F32, 2n, {0.0 : F32}), Array.new(F32, 2n, {0.0 : F32}), Array.new(F32, 1n, {0.0 : F32})))
