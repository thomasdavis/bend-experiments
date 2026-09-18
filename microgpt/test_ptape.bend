import Base
import ./ptape.bend as P

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(F32.show(P.smoke()))
