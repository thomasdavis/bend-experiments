import Base

# Pure tensor/list ops for the GPU-track engine (M1: numeric parity, no tape).
# Core is nn.bend verbatim (proven); extras below it. Everything pure and
# structural so later milestones can mark data-parallel calls with `!`.

def dot.go(+xs: +List<F32>, +ys: +List<F32>, acc: F32) -> F32:
  match xs ys:
    case Nil{} Nil{}:
      acc
    case Con{hx, tx} Con{hy, ty}:
      dot.go(tx, ty, (acc + hx * hy : F32))
    case _ _:
      acc

def dot(+xs: +List<F32>, +ys: +List<F32>) -> F32:
  dot.go(xs, ys, {0.0 : F32})

def sumsq.go(+xs: +List<F32>, n: Nat, acc: F32) -> Nat & F32:
  match xs:
    case Nil{}:
      (n, acc)
    case Con{h, t}:
      sumsq.go(t, 1n + n, (acc + h * h : F32))

def sumsq(+xs: +List<F32>) -> Nat & F32:
  sumsq.go(xs, 0n, {0.0 : F32})

def rmsnorm.map(+xs: +List<F32>, +s: F32) -> +List<F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{(h * s : F32), rmsnorm.map(t, s)}


 def rmsnorm.sc(+ss: F32, +n: Nat) -> F32:
   F32.pow((F32.div(ss, F32.from_nat(n)) + 0.00001 : F32), ((0.0 - 0.5 : F32)))
 def rmsnorm.f(p: +List<F32> & (Nat & F32)) -> +List<F32>:
   (xs, (n, ss)) = p
   rmsnorm.map(xs, rmsnorm.sc(ss, n))
def rmsnorm(+xs: +List<F32>) -> +List<F32>:
  rmsnorm.f((xs, sumsq(xs)))

def listmax.go(+xs: +List<F32>, +best: F32) -> F32:
  match xs:
    case Nil{}:
      best
    case Con{h, t}:
      listmax.go(t, F32.max(h, best))

def listmax.h(+xs: +List<F32>) -> F32:
  match xs:
    case Nil{}:
      {0.0 : F32}
    case Con{h, t}:
      listmax.go(t, h)

def listmax(+xs: +List<F32>) -> F32:
  listmax.h(xs)

def softmax.tot.go(+xs: +List<F32>, +m: F32, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      softmax.tot.go(t, m, (acc + F32.exp((h - m : F32)) : F32))

def softmax.e(+xs: +List<F32>, +m: F32, +tot: F32) -> +List<F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{(F32.exp((h - m : F32)) / tot : F32), softmax.e(t, m, tot)}

def softmax.m(p: +List<F32> & F32) -> +List<F32>:
  (+xs, +m) = p
  softmax.e(xs, m, softmax.tot.go(xs, m, {0.0 : F32}))

def softmax(+xs: +List<F32>) -> +List<F32>:
  softmax.m((xs, listmax(xs)))

# --- tensor extras (all pure, callee-first) ---

def tadd.zip(+xs: +List<F32>, +ys: +List<F32>) -> +List<F32>:
  match xs ys:
    case Nil{} Nil{}:
      Nil{}
    case Con{x, xt} Con{y, yt}:
      Con{(x + y : F32), tadd.zip(xt, yt)}
    case _ _:
      Nil{}

def tscale.map(+xs: +List<F32>, +s: F32) -> +List<F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{x, xt}:
      Con{(x * s : F32), tscale.map(xt, s)}

def tnth(+xs: +List<F32>, n: Nat) -> F32:
  match xs n:
    case Nil{} _:
      {0.0 : F32}
    case Con{x, xt} 0n:
      x
    case Con{x, xt} 1n+p:
      tnth(xt, p)

def tgather(+tab: +List<F32>, +base: U32, +r: Nat, +C: Nat) -> +List<F32>:
  List.take(&2, F32, List.drop(&2, F32, tab, U32.to_nat((base + U32.mul(U32.from_nat(r), U32.from_nat(C)) : U32))), C)

def tmatvec.go(+rs: +List<Nat>, +tab: +List<F32>, +base: U32, +C: Nat, +x: +List<F32>, +acc: +List<F32>) -> +List<F32>:
  match rs:
    case Nil{}:
      List.reverse(&2, F32, acc)
    case Con{h, t}:
      tmatvec.go(t, tab, base, C, x, Con{dot(tgather(tab, base, h, C), x), acc})

def tmatvec(+tab: +List<F32>, +base: U32, rows: Nat, +C: Nat, +x: +List<F32>) -> +List<F32>:
  tmatvec.go(List.range(rows), tab, base, C, x, Nil{})

def phash.c(+u: F32) -> F32:
  ((u - 0.08 : F32))

def phash(+i: U32) -> F32:
  phash.c((U32.to_f32(U32.mul(i, 2654435761)) / 4294967296.0 : F32))

def pinit.go(n: Nat, +off: U32) -> +List<F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+m:
      Con{phash(off), pinit.go(m, (off + 1 : U32))}

def zeros(n: Nat) -> +List<F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+m:
      Con{{0.0 : F32}, zeros(m)}
