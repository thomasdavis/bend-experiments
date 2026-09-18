import Base
import ./data.bend as D

def test.go3(+us: +List<Char>, +ls: +List<String>) -> IO(Unit):
  do IO<Unit>:
    IO.print(U32.show(D.vof(us)) ++ " " ++ U32.show(U32.from_nat(List.length(&2, String, ls))))

def test.go2(+ls: +List<String>) -> IO(Unit):
  do IO<Unit>:
    test.go3(D.vocab(ls), ls)

def test.go(s: String) -> IO(Unit):
  test.go2(D.lines(s))

def main() -> IO(Unit):
  do IO<Unit>:
    s : String <- D.readInput("/home/ajax/tmp/bend-experiments/microgpt/input.txt")
    test.go(s)
