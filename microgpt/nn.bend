import Base

# NN primitives for the microgpt port, over reusable +List<F32>.
# (Plain affine lists cannot be shared across matrix rows or two passes,
# so all numeric data is +List<F32>; + bindings may be used more than once.
# Pair-passing helpers are still required wherever a value comes from a
# computed call — Bend only scrutinizes parameters.)

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

def linear.go(+rows: +List<+List<F32>>, +x: +List<F32>, +acc: +List<F32>) -> +List<F32>:
  match rows:
    case Nil{}:
      List.reverse(&2, F32, acc)
    case Con{row, rest}:
      linear.go(rest, x, Con{dot(row, x), acc})

def linear(+rows: +List<+List<F32>>, +x: +List<F32>) -> +List<F32>:
  linear.go(rows, x, Nil{})

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

def rmsnorm.f(p: +List<F32> & (Nat & F32)) -> +List<F32>:
  (xs, (n, ss)) = p
  mean = (ss / F32.from_nat(n) : F32)
  scale = F32.div({1.0 : F32}, F32.sqrt((mean + 0.00001 : F32)))
  rmsnorm.map(xs, scale)

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
