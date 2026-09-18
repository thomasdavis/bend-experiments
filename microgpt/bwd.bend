import Base
import ./tape.bend as T

# Reverse-mode sweep over the tape. Dispatch is a Bool chain (Bend cannot
# match on a computed U32); every level takes the test result as a parameter.

def bwd.add(a: U32, b: U32, +g: F32) -> T.TP<Unit>:
  do T.TP<Unit>:
    u : Unit <- T.gadd(a, g)
    T.gadd(b, g)

def bwd.mul3(bn: T.Node, +da: F32, +a: U32, +b: U32, +g: F32) -> T.TP<Unit>:
  match bn:
    case T.N{oy, ay, by, ky, db}:
      do T.TP<Unit>:
        u : Unit <- T.gadd(a, (db * g : F32))
        T.gadd(b, (da * g : F32))

def bwd.mul2(da: F32, +a: U32, +b: U32, +g: F32) -> T.TP<Unit>:
  do T.TP<Unit>:
    bn : T.Node <- T.tget(b)
    bwd.mul3(bn, da, a, b, g)

def bwd.mul1(an: T.Node, +a: U32, +b: U32, +g: F32) -> T.TP<Unit>:
  match an:
    case T.N{ox, ax, bx, kx, da}:
      bwd.mul2(da, a, b, g)

def bwd.mul(+a: U32, +b: U32, +g: F32) -> T.TP<Unit>:
  do T.TP<Unit>:
    an : T.Node <- T.tget(a)
    bwd.mul1(an, a, b, g)

def bwd.pow1(an: T.Node, +a: U32, +k: F32, +g: F32) -> T.TP<Unit>:
  match an:
    case T.N{ox, ax, bx, kx, da}:
      T.gadd(a, (k * F32.pow(da, (k - 1.0 : F32)) * g : F32))

def bwd.pow(+a: U32, +k: F32, +g: F32) -> T.TP<Unit>:
  do T.TP<Unit>:
    an : T.Node <- T.tget(a)
    bwd.pow1(an, a, k, g)

def bwd.log1(an: T.Node, +a: U32, +g: F32) -> T.TP<Unit>:
  match an:
    case T.N{ox, ax, bx, kx, da}:
      T.gadd(a, (g / da : F32))

def bwd.log(+a: U32, +g: F32) -> T.TP<Unit>:
  do T.TP<Unit>:
    an : T.Node <- T.tget(a)
    bwd.log1(an, a, g)

def bwd.exp(+a: U32, +d: F32, +g: F32) -> T.TP<Unit>:
  T.gadd(a, (d * g : F32))

def bwd.relu(+a: U32, +d: F32, +g: F32) -> T.TP<Unit>:
  T.gadd(a, Bool.pick(F32, F32.is_gt(d, {0.0 : F32}), g, {0.0 : F32}))

def bwd.leaf(a: U32, +g: F32) -> T.TP<Unit>:
  T.pgadd(a, g)

def bwd.noop() -> T.TP<Unit>:
  do T.TP<Unit>:
    return Unit{}
 def bwd.is7(z: Bool, a: U32, b: U32, k: F32, d: F32, +g: F32) -> T.TP<Unit>:
   match z:
     case True{}:
       bwd.noop()
     case False{}:
       bwd.leaf(a, g)

def bwd.is6(z: Bool, +op: U32, a: U32, b: U32, +k: F32, +d: F32, +g: F32) -> T.TP<Unit>:
  match z:
    case True{}:
      bwd.relu(a, d, g)
    case False{}:
      bwd.is7(U32.is_eq(op, 7), a, b, k, d, g)

def bwd.is5(z: Bool, +op: U32, a: U32, b: U32, +k: F32, +d: F32, +g: F32) -> T.TP<Unit>:
  match z:
    case True{}:
      bwd.exp(a, d, g)
    case False{}:
      bwd.is6(U32.is_eq(op, 6), op, a, b, k, d, g)

def bwd.is4(z: Bool, +op: U32, a: U32, b: U32, +k: F32, +d: F32, +g: F32) -> T.TP<Unit>:
  match z:
    case True{}:
      bwd.log(a, g)
    case False{}:
      bwd.is5(U32.is_eq(op, 5), op, a, b, k, d, g)

def bwd.is3(z: Bool, +op: U32, a: U32, b: U32, +k: F32, +d: F32, +g: F32) -> T.TP<Unit>:
  match z:
    case True{}:
      bwd.pow(a, k, g)
    case False{}:
      bwd.is4(U32.is_eq(op, 4), op, a, b, k, d, g)

def bwd.is2(z: Bool, +op: U32, a: U32, b: U32, +k: F32, +d: F32, +g: F32) -> T.TP<Unit>:
  match z:
    case True{}:
      bwd.mul(a, b, g)
    case False{}:
      bwd.is3(U32.is_eq(op, 3), op, a, b, k, d, g)

def bwd.is1(z: Bool, +op: U32, a: U32, b: U32, +k: F32, +d: F32, +g: F32) -> T.TP<Unit>:
  match z:
    case True{}:
      bwd.add(a, b, g)
    case False{}:
      bwd.is2(U32.is_eq(op, 2), op, a, b, k, d, g)

def bwd.op(nd: T.Node, +g: F32) -> T.TP<Unit>:
  match nd:
    case T.N{+op, a, b, +k, +d}:
      bwd.is1(U32.is_eq(op, 1), op, a, b, k, d, g)

def bwd.put(nd: T.Node, g: F32) -> T.TP<Unit>:
  bwd.op(nd, g)

def bwd.one(+idx: U32) -> T.TP<Unit>:
  do T.TP<Unit>:
    nd : T.Node <- T.tget(idx)
    g : F32 <- T.gget(idx)
    bwd.put(nd, g)

 def bwd.n2u.go(+xs: +List<Nat>) -> +List<U32>:
   match xs:
     case Nil{}:
       Nil{}
     case Con{h, t}:
       Con{U32.from_nat(h), bwd.n2u.go(t)}

 def bwd.ids(n: Nat) -> +List<U32>:
   bwd.n2u.go(List.reverse(&2, Nat, List.range(n)))

 def bwd.go(+ids: +List<U32>) -> T.TP<Unit>:
   match ids:
     case Nil{}:
       do T.TP<Unit>:
         return Unit{}
     case Con{h, t}:
       do T.TP<Unit>:
         u : Unit <- bwd.one(h)
         bwd.go(t)
