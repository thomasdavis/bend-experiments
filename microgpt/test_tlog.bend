import Base
import ./gten.bend as G
import ./gfwd.bend as F
import ./tfwd.bend as T

def showvec(+xs: +List<F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      F32.show(h) ++ " " ++ showvec(t)

def showhb(hb: T.HB) -> IO(Unit):
  match hb:
    case T.HB{out, ac}:
      match out:
        case T.FV3{lg, tp, nx, id}:
          do IO<Unit>:
            IO.print(showvec(lg))

def showgs(gs: T.GS3) -> IO(Unit):
  match gs:
    case T.GS3{toks, pos, cache, acc, n, tp, nx, V, params}:
      do IO<Unit>:
        IO.print(F32.show(acc))

def main() -> IO(Unit):
  do IO<Unit>:
    u0 : Unit <- showhb(T.gstep(3, 0n, 4, G.zeros(3456n), T.pc30(), Nil{}, 0))
    showgs(T.tloss.entry([{3 : U32}, {0 : U32}], 4, G.zeros(3456n)))
