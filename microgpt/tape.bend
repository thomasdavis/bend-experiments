import Base

# Scalar-tape autograd for the microgpt port.
# microgpt's Value graph mutates shared nodes in place; Bend is affine, so the
# tape is explicit: forward appends Node records, backward sweeps top-down
# accumulating into grad arrays. Activations travel as (id & data) pairs, so
# forward never reads the tape back.
#
# op tags: 0 leaf(param idx in a) 1 add 2 mul 3 pow(k) 4 log 5 exp 6 relu 7 const

type Node is Data:
  N{op: U32, a: U32, b: U32, k: F32, data: F32}

def node0() -> Node:
  N{0, 0, 0, {0.0 : F32}, {0.0 : F32}}

# Whole training state: tape + next id, params + grads, Adam m/v buffers.
type TS is Type:
  TS{t: Array<Node>, n: U32, p: Array<F32>, g: Array<F32>, pg: Array<F32>, m: Array<F32>, v: Array<F32>}

def ts_new() -> TS:
  TS{Array.new(Node, 17n, node0()), 0, Array.new(F32, 13n, {0.0 : F32}), Array.new(F32, 17n, {0.0 : F32}), Array.new(F32, 13n, {0.0 : F32}), Array.new(F32, 13n, {0.0 : F32}), Array.new(F32, 13n, {0.0 : F32})}

# State monad: TP<A> is a tape-state-passing computation (same shape as IO).
type TP<-A: Type> is Type:
  TP{run: TS -> TS & A}

def TP.open(-A: Type, m: TP<A>, st: TS) -> TS & A:
  match m:
    case TP{run}:
      run(st)

def TP.then.go(-A: Type, -B: Type, f: A -> TP<B>, st2: TS, a: A) -> TS & B:
  TP.open(B, f(a), st2)

def TP.then(-A: Type, -B: Type, f: A -> TP<B>, r: TS & A) -> TS & B:
  match r:
    case (st2, a):
      TP.then.go(A, B, f, st2, a)

def TP.pure(-A: Type, x: A) -> TP<A>:
  TP{st => (st, x)}

def TP.bind(-A: Type, -B: Type, m: TP<A>, f: A -> TP<B>) -> TP<B>:
  TP{st => TP.then(A, B, f, TP.open(A, m, st))}

def TP.run(-A: Type, m: TP<A>, st: TS) -> TS & A:
  TP.open(A, m, st)

# --- array accessors (each puts the array back; pairs go through helpers) ---

def tappend.go(st: TS, op: U32, a: U32, b: U32, k: F32, d: F32) -> TS & U32:
  match st:
    case TS{t, +n, p, g, pg, m, v}:
      (TS{Array.set(Node, t, n, N{op, a, b, k, d}), (n + 1 : U32), p, g, pg, m, v}, n)

def tappend(op: U32, a: U32, b: U32, k: F32, d: F32) -> TP<U32>:
  TP{st => tappend.go(st, op, a, b, k, d)}

 def tget.put(q: Array<Node> & Node, n: U32, p: Array<F32>, g: Array<F32>, pg: Array<F32>, m: Array<F32>, v: Array<F32>) -> TS & Node:
   (t2, nd) = q
   (TS{t2, n, p, g, pg, m, v}, nd)

 def tget.go(st: TS, i: U32) -> TS & Node:
   match st:
     case TS{t, n, p, g, pg, m, v}:
       tget.put(Array.get(Node, t, i), n, p, g, pg, m, v)

 def tget(i: U32) -> TP<Node>:
   TP{st => tget.go(st, i)}

 def gget.put(q: Array<F32> & F32, t: Array<Node>, n: U32, p: Array<F32>, pg: Array<F32>, m: Array<F32>, v: Array<F32>) -> TS & F32:
   (g2, x) = q
   (TS{t, n, p, g2, pg, m, v}, x)

 def gget.go(st: TS, i: U32) -> TS & F32:
   match st:
     case TS{t, n, p, g, pg, m, v}:
       gget.put(Array.get(F32, g, i), t, n, p, pg, m, v)

 def gget(i: U32) -> TP<F32>:
   TP{st => gget.go(st, i)}

 def gset.go(st: TS, i: U32, x: F32) -> TS & Unit:
   match st:
     case TS{t, n, p, g, pg, m, v}:
       (TS{t, n, p, Array.set(F32, g, i, x), pg, m, v}, Unit{})

 def gset(i: U32, x: F32) -> TP<Unit>:
   TP{st => gset.go(st, i, x)}

 def gadd(+i: U32, x: F32) -> TP<Unit>:
   do TP<Unit>:
     old : F32 <- gget(i)
     gset(i, (old + x : F32))

 def pget.put(q: Array<F32> & F32, t: Array<Node>, n: U32, g: Array<F32>, pg: Array<F32>, m: Array<F32>, v: Array<F32>) -> TS & F32:
   (p2, x) = q
   (TS{t, n, p2, g, pg, m, v}, x)

 def pget.go(st: TS, i: U32) -> TS & F32:
   match st:
     case TS{t, n, p, g, pg, m, v}:
       pget.put(Array.get(F32, p, i), t, n, g, pg, m, v)

 def pget(i: U32) -> TP<F32>:
   TP{st => pget.go(st, i)}

 def pset.go(st: TS, i: U32, x: F32) -> TS & Unit:
   match st:
     case TS{t, n, p, g, pg, m, v}:
       (TS{t, n, Array.set(F32, p, i, x), g, pg, m, v}, Unit{})

 def pset(i: U32, x: F32) -> TP<Unit>:
   TP{st => pset.go(st, i, x)}

 def pgget.put(q: Array<F32> & F32, t: Array<Node>, n: U32, p: Array<F32>, g: Array<F32>, m: Array<F32>, v: Array<F32>) -> TS & F32:
   (pg2, x) = q
   (TS{t, n, p, g, pg2, m, v}, x)

 def pgget.go(st: TS, i: U32) -> TS & F32:
   match st:
     case TS{t, n, p, g, pg, m, v}:
       pgget.put(Array.get(F32, pg, i), t, n, p, g, m, v)

 def pgget(i: U32) -> TP<F32>:
   TP{st => pgget.go(st, i)}

 def pgset.go(st: TS, i: U32, x: F32) -> TS & Unit:
   match st:
     case TS{t, n, p, g, pg, m, v}:
       (TS{t, n, p, g, Array.set(F32, pg, i, x), m, v}, Unit{})

 def pgset(i: U32, x: F32) -> TP<Unit>:
   TP{st => pgset.go(st, i, x)}

 def pgadd(+i: U32, x: F32) -> TP<Unit>:
   do TP<Unit>:
     old : F32 <- pgget(i)
     pgset(i, (old + x : F32))

 def mget.put(q: Array<F32> & F32, t: Array<Node>, n: U32, p: Array<F32>, g: Array<F32>, pg: Array<F32>, v: Array<F32>) -> TS & F32:
   (m2, x) = q
   (TS{t, n, p, g, pg, m2, v}, x)

 def mget.go(st: TS, i: U32) -> TS & F32:
   match st:
     case TS{t, n, p, g, pg, m, v}:
       mget.put(Array.get(F32, m, i), t, n, p, g, pg, v)

 def mget(i: U32) -> TP<F32>:
   TP{st => mget.go(st, i)}

 def mset.go(st: TS, i: U32, x: F32) -> TS & Unit:
   match st:
     case TS{t, n, p, g, pg, m, v}:
       (TS{t, n, p, g, pg, Array.set(F32, m, i, x), v}, Unit{})

 def mset(i: U32, x: F32) -> TP<Unit>:
   TP{st => mset.go(st, i, x)}

 def vget.put(q: Array<F32> & F32, t: Array<Node>, n: U32, p: Array<F32>, g: Array<F32>, pg: Array<F32>, m: Array<F32>) -> TS & F32:
   (v2, x) = q
   (TS{t, n, p, g, pg, m, v2}, x)

 def vget.go(st: TS, i: U32) -> TS & F32:
   match st:
     case TS{t, n, p, g, pg, m, v}:
       vget.put(Array.get(F32, v, i), t, n, p, g, pg, m)

 def vget(i: U32) -> TP<F32>:
   TP{st => vget.go(st, i)}

 def vset.go(st: TS, i: U32, x: F32) -> TS & Unit:
   match st:
     case TS{t, n, p, g, pg, m, v}:
       (TS{t, n, p, g, pg, m, Array.set(F32, v, i, x)}, Unit{})

 def vset(i: U32, x: F32) -> TP<Unit>:
   TP{st => vset.go(st, i, x)}

 # --- scalar ops over bare U32 ids. Data is never carried: every op re-reads
 # its inputs with tget and returns only the new id. (Carrying (id & data)
 # pairs needs `+` in let-destructuring, which Bend rejects.)

 def vadd.h3(xn: Node, yd: F32, +x: U32, +y: U32) -> TP<U32>:
   match xn:
     case N{ox, ax, bx, kx, xd}:
       tappend(1, x, y, {0.0 : F32}, (xd + yd : F32))

 def vadd.h2(yd: F32, +x: U32, +y: U32) -> TP<U32>:
   do TP<U32>:
     xn : Node <- tget(x)
     vadd.h3(xn, yd, x, y)

 def vadd.h1(yn: Node, +x: U32, +y: U32) -> TP<U32>:
   match yn:
     case N{oy, ay, by, ky, yd}:
       vadd.h2(yd, x, y)

 def vadd(+x: U32, +y: U32) -> TP<U32>:
   do TP<U32>:
     yn : Node <- tget(y)
     vadd.h1(yn, x, y)

 def vmul.h3(xn: Node, yd: F32, +x: U32, +y: U32) -> TP<U32>:
   match xn:
     case N{ox, ax, bx, kx, xd}:
       tappend(2, x, y, {0.0 : F32}, (xd * yd : F32))

 def vmul.h2(yd: F32, +x: U32, +y: U32) -> TP<U32>:
   do TP<U32>:
     xn : Node <- tget(x)
     vmul.h3(xn, yd, x, y)

 def vmul.h1(yn: Node, +x: U32, +y: U32) -> TP<U32>:
   match yn:
     case N{oy, ay, by, ky, yd}:
       vmul.h2(yd, x, y)

 def vmul(+x: U32, +y: U32) -> TP<U32>:
   do TP<U32>:
     yn : Node <- tget(y)
     vmul.h1(yn, x, y)

 def vpow.h(xn: Node, +x: U32, +k: F32) -> TP<U32>:
   match xn:
     case N{o, a, b, kk, xd}:
       tappend(3, x, 0, k, F32.pow(xd, k))

 def vpow(+x: U32, +k: F32) -> TP<U32>:
   do TP<U32>:
     xn : Node <- tget(x)
     vpow.h(xn, x, k)

 def vlog.h(xn: Node, +x: U32) -> TP<U32>:
   match xn:
     case N{o, a, b, k, xd}:
       tappend(4, x, 0, {0.0 : F32}, F32.log(xd))

 def vlog(+x: U32) -> TP<U32>:
   do TP<U32>:
     xn : Node <- tget(x)
     vlog.h(xn, x)

 def vexp.h(xn: Node, +x: U32) -> TP<U32>:
   match xn:
     case N{o, a, b, k, xd}:
       tappend(5, x, 0, {0.0 : F32}, F32.exp(xd))

 def vexp(+x: U32) -> TP<U32>:
   do TP<U32>:
     xn : Node <- tget(x)
     vexp.h(xn, x)

 def vrelu.h(xn: Node, +x: U32) -> TP<U32>:
   match xn:
     case N{o, a, b, k, xd}:
       tappend(6, x, 0, {0.0 : F32}, F32.max(xd, {0.0 : F32}))

 def vrelu(+x: U32) -> TP<U32>:
   do TP<U32>:
     xn : Node <- tget(x)
     vrelu.h(xn, x)

 def vconst(+c: F32) -> TP<U32>:
   tappend(7, 0, 0, {0.0 : F32}, c)

 def vleaf.go(pd: F32, +pi: U32) -> TP<U32>:
   tappend(0, pi, 0, {0.0 : F32}, pd)

 def vleaf.put(p: F32 & U32) -> TP<U32>:
   (pd, pi) = p
   vleaf.go(pd, pi)

 def vleaf(+pi: U32) -> TP<U32>:
   do TP<U32>:
     pd : F32 <- pget(pi)
     vleaf.put((pd, pi))

 def vneg(+x: U32) -> TP<U32>:
   do TP<U32>:
     m1 : U32 <- vconst(((0.0 - 1.0 : F32)))
     vmul(m1, x)

 def vsub(+x: U32, +y: U32) -> TP<U32>:
   do TP<U32>:
     ny : U32 <- vneg(y)
     vadd(x, ny)

 def vdiv(+x: U32, +y: U32) -> TP<U32>:
   do TP<U32>:
     iy : U32 <- vpow(y, ((0.0 - 1.0 : F32)))
     vmul(x, iy)
