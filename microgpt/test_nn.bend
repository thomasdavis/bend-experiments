import Base
import ./nn.bend as N

def show_list(+xs: +List<F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      " " ++ F32.show(h) ++ show_list(t)

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print("dot:" ++ F32.show(N.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])))
    IO.print("rmsnorm:" ++ show_list(N.rmsnorm([1.0, 2.0, 3.0, 4.0])))
    IO.print("softmax:" ++ show_list(N.softmax([1.0, 2.0, 3.0])))
