import Base
import ./LAWS.bend as Laws

# PROOF: add_zero holds by induction on x.
def Laws.add_zero(x):
  match x:
    case 0n:
      {==}
    case 1n+p:
      %Laws.add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}
