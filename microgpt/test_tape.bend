import Base
import ./tape.bend as T
import ./bwd.bend as B

# f = x*y + x at x=2, y=3: expect f=8, dx=4, dy=2.
 def finish(+f: U32) -> T.TP<(F32 & F32)>:
   do T.TP<(F32 & F32)>:
     u2 : Unit <- T.gset(f, {1.0 : F32})
     u3 : Unit <- B.bwd.go(B.bwd.ids(1n + U32.to_nat(f)))
     gx : F32 <- T.pgget(0)
     gy : F32 <- T.pgget(1)
     return (gx, gy)

 def prog.m(m: U32, +x: U32) -> T.TP<(F32 & F32)>:
   do T.TP<(F32 & F32)>:
     f : U32 <- T.vadd(m, x)
     finish(f)

 def prog.xy(+x: U32, +y: U32) -> T.TP<(F32 & F32)>:
   do T.TP<(F32 & F32)>:
     m : U32 <- T.vmul(x, y)
     prog.m(m, x)

 def prog.x(+x: U32) -> T.TP<(F32 & F32)>:
   do T.TP<(F32 & F32)>:
     y : U32 <- T.vleaf(1)
     prog.xy(x, y)

 def prog() -> T.TP<(F32 & F32)>:
   do T.TP<(F32 & F32)>:
     u0 : Unit <- T.pset(0, {2.0 : F32})
     u1 : Unit <- T.pset(1, {3.0 : F32})
     x : U32 <- T.vleaf(0)
     prog.x(x)

def go(p: T.TS & (F32 & F32)) -> IO(Unit):
  (st, (gx, gy)) = p
  do IO<Unit>:
    IO.print(F32.show(gx) ++ " " ++ F32.show(gy))

def main() -> IO(Unit):
  go(T.TP.run((F32 & F32), prog(), T.ts_new()))
