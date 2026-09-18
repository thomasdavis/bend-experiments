import Base

# LAW: adding zero on the right is identity (starter gate).
law add_zero:
  for x: Nat
  {Nat.add(x, 0n) == x : Nat}
