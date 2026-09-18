import Base
import ./gten.bend as G
import ./gfwd.bend as F

 def zeros(n: Nat) -> +List<F32>:
   match n:
     case 0n:
       Nil{}
     case 1n+m:
       Con{{0.0 : F32}, zeros(m)}

 def showvec(+xs: +List<F32>) -> String:
   match xs:
     case Nil{}:
       ""
     case Con{h, t}:
       F32.show(h) ++ " " ++ showvec(t)

 def showstep(st: +List<F32> & F.PC) -> IO(Unit):
   match st:
     case (lg, ac):
       do IO<Unit>:
         IO.print(showvec(lg))

 def showfin(p: F32 & Nat) -> IO(Unit):
   match p:
     case (s, n):
       do IO<Unit>:
         IO.print(F32.show(s))

def main() -> IO(Unit):
  do IO<Unit>:
    u0 : Unit <- IO.print(F32.show(G.tnth(G.softmax([{1.0 : F32}, {2.0 : F32}, {3.0 : F32}]), 2n)))
    u1 : Unit <- IO.print(F32.show(G.tnth(G.tgather(Con{{1.0 : F32}, Con{{2.0 : F32}, Con{{3.0 : F32}, Con{{4.0 : F32}, Nil{}}}}}, 0, 1n, 2n), 1n)))
    u2 : Unit <- IO.print(F32.show(G.dot([{1.0 : F32}, {2.0 : F32}], [{4.0 : F32}, {5.0 : F32}])))
     u3 : Unit <- showstep(F.gstep(3, 0n, 4, zeros(3456n), F.pc0()))
     showfin(F.gloss.go([{3 : U32}, {0 : U32}, {1 : U32}, {3 : U32}], 0n, 4, zeros(3456n), F.pc0(), {0.0 : F32}, 0n))
