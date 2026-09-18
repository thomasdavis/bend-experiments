import Base
import ./gten.bend as G
import ./batch.bend as B

# GPU measurement entry: 64-doc batch x 3 steps through the `!` tree driver.

def syndocs(n: Nat) -> +List<+List<U32>>:
  match n:
    case 0n:
      Nil{}
    case 1n+m:
      Con{Con{3, Con{0, Nil{}}}, syndocs(m)}

def main() -> IO(Unit):
  B.out(B.bdriver(3n, B.DS{G.zeros(U32.to_nat(3456)), G.zeros(U32.to_nat(3456)), G.zeros(U32.to_nat(3456)), 0n, syndocs(64n), 4, U32.to_nat(3456), {0.01 : F32}, Nil{}}))
