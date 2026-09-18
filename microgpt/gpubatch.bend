import Base
import ./gten.bend as G
import ./tfwd.bend as F
import ./tsweep.bend as S
import ./batch.bend as B
import ./data.bend as D
import ./model.bend as M
import ./train.bend as TR
import ./gfwd.bend as P

# M4: minibatch GPU training driver. Corpus blocks picked per step,
# B.bstep does batch forward+sweep+Adam; LGPC sampler is pure.
# Callee-before-caller throughout.

 def fuel() -> Nat:
   200n

 def nbs() -> Nat:
   256n

 def lr0() -> F32:
   {0.01 : F32}

type DS2 is Data:
  DS2{params: +List<F32>, m: +List<F32>, v: +List<F32>, step: Nat, blocks: +List<+List<+List<U32>>>, V: U32, pn: Nat, losses: +List<F32>}


def chunk.go(fuel: Nat, +xs: +List<Char>) -> +List<+List<Char>>:
  match fuel xs:
    case 0n _:
      Nil{}
    case 1n+f Nil{}:
      Nil{}
    case 1n+f Con{d, t}:
      Con{List.take(&2, Char, xs, 15n), chunk.go(f, List.drop(&2, Char, xs, 15n))}

def chunk15(+xs: +List<Char>) -> +List<+List<Char>>:
  chunk.go(List.length(&2, Char, xs), xs)

def endoc(+cs: +List<Char>, +us: +List<Char>, +bos: U32) -> +List<U32>:
   D.enc.wrap(D.enc.ids(cs, us), bos)

 def endocs(+css: +List<+List<Char>>, +us: +List<Char>, +bos: U32) -> +List<+List<U32>>:
   match css:
     case Nil{}:
       Nil{}
     case Con{c, t}:
       Con{endoc(c, us, bos), endocs(t, us, bos)}

 def linechunks(+lns: +List<String>) -> +List<+List<Char>>:
   match lns:
     case Nil{}:
       Nil{}
     case Con{l, t}:
       List.append(&2, +List<Char>, chunk15(D.str.chars(l)), linechunks(t))

 def blocks.go(fuel: Nat, +xs: +List<+List<U32>>, +n: Nat) -> +List<+List<+List<U32>>>:
   match fuel xs:
     case 0n _:
       Nil{}
     case 1n+f Nil{}:
       Nil{}
     case 1n+f Con{d, t}:
       Con{List.take(&2, +List<U32>, xs, n), blocks.go(f, List.drop(&2, +List<U32>, xs, n), n)}

 def blocks(+xs: +List<+List<U32>>, n: Nat) -> +List<+List<+List<U32>>>:
   blocks.go(List.length(&2, +List<U32>, xs), xs, n)

 def gpick(+bs: +List<+List<+List<U32>>>, n: Nat) -> +List<+List<U32>>:
   match bs n:
     case Nil{} _:
       Nil{}
     case Con{b, t} 0n:
       b
     case Con{b, t} 1n+p:
       gpick(t, p)

 def gdriver.nx(st: B.DS, +blocks: +List<+List<+List<U32>>>, +acc: +List<F32>) -> DS2:
   match st:
     case B.DS{params, m, v, step, docs, V2, pn2, lr2, losses}:
       match losses:
         case Nil{}:
           DS2{params, m, v, step, blocks, V2, pn2, acc}
         case Con{bl, t}:
           DS2{params, m, v, step, blocks, V2, pn2, Con{bl, acc}}

 def gdriver(fuel: Nat, st: DS2) -> DS2:
   match fuel:
     case 0n:
       st
     case 1n+f:
       match st:
         case DS2{params, m, v, step, blocks, V, pn, losses}:
           match blocks:
             case Nil{}:
               st
             case Con{+blk, rest}:
               gdriver(f, gdriver.nx(B.bstep(blk, V, params, m, v, step, lr0(), pn, Nil{}), List.append(&2, +List<+List<U32>>, rest, Con{blk, Nil{}}), losses))
 type SI is Data:
   SI{id: U32, val: F32}

 type LGPC is Data:
   LGPC{lg: +List<F32>, cache: P.PC}

 def pkup.go(+xs: +List<F32>, +i: U32) -> +List<SI>:
   match xs:
     case Nil{}:
       Nil{}
     case Con{x, t}:
       Con{SI{i, x}, pkup.go(t, (i + 1 : U32))}

 def pkup(+xs: +List<F32>) -> +List<SI>:
   pkup.go(xs, 0)

 def sargmax.go(+xs: +List<SI>, +bi: U32, +bd: F32) -> U32:
   match xs:
     case Nil{}:
       bi
     case Con{pr, t}:
       match pr:
         case SI{i, d}:
           sargmax.go(t, Bool.pick(U32, F32.is_gt(d, bd), i, bi), F32.max(d, bd))

 def sargmax(+pr: +List<SI>) -> U32:
   sargmax.go(pr, 0, ((0.0 - 1000000.0 : F32)))

 def gpert2.go(+xs: +List<F32>, +smp: U32, +pos: Nat, +i: U32) -> +List<SI>:
   match xs:
     case Nil{}:
       Nil{}
     case Con{x, t}:
       Con{SI{i, F32.add(F32.log(x), TR.ggum(TR.ghash(smp, pos, i)))}, gpert2.go(t, smp, pos, (i + 1 : U32))}

 def gpert2(+xs: +List<F32>, +smp: U32, +pos: Nat) -> +List<SI>:
   gpert2.go(xs, smp, pos, 0)

 def fwd1b(st: +List<F32> & P.PC) -> LGPC:
   match st:
     case (lg, ac):
       LGPC{lg, ac}

 def fwd1(+tok: U32, +pos: Nat, +V: U32, +params: +List<F32>, +cache: P.PC) -> LGPC:
   fwd1b(P.gstep(tok, pos, V, params, cache))

 def gsamp.go(fuel: Nat, +pos: Nat, +V: U32, +params: +List<F32>, st: LGPC, +acc: +List<U32>, +smp: U32) -> +List<U32>:
   match fuel:
     case 0n:
       acc
     case 1n+f:
       match st:
         case LGPC{+lg, cache}:
           gsamp.go(f, 1n + pos, V, params, fwd1(sargmax(gpert2(lg, smp, pos)), 1n + pos, V, params, cache), Con{sargmax(gpert2(lg, smp, pos)), acc}, smp)

 def pc0p() -> P.PC:
   P.PC{Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}}

 def gsamps.go(k: Nat, +V: U32, +params: +List<F32>, +j: U32) -> +List<+List<U32>>:
   match k:
     case 0n:
       Nil{}
     case 1n+f:
       Con{gsamp.go(16n, 0n, V, params, fwd1((V - 1 : U32), 0n, V, params, pc0p()), Con{(V - 1 : U32), Nil{}}, j), gsamps.go(f, V, params, (j + 1 : U32))}

 def gsamps(+params: +List<F32>, +V: U32) -> +List<+List<U32>>:
   gsamps.go(10n, V, params, 0)

 def out.samp(+params: +List<F32>, +V: U32, +us: +List<Char>, +losses: +List<F32>) -> IO(Unit):
   do IO<Unit>:
     u : Unit <- IO.print(TR.showL(losses))
     TR.printAll(TR.dec.all(gsamps(params, V), us))

 def out(st: DS2, +us: +List<Char>) -> IO(Unit):
   match st:
     case DS2{params, m, v, step, blocks, V, pn, losses}:
       out.samp(params, V, us, losses)

 def prep.docs(+docs: +List<+List<U32>>, +V: U32, +pn: Nat) -> DS2:
   gdriver(fuel(), DS2{G.pinit.go(U32.to_nat(M.base.np(V)), 0), G.zeros(pn), G.zeros(pn), 0n, blocks(docs, nbs()), V, pn, Nil{}})

 type PR is Data:
   PR{st: DS2, us: +List<Char>}

 def prep.go(+ls: +List<String>) -> PR:
   PR{prep.docs(endocs(linechunks(ls), D.vocab(ls), (D.vof(D.vocab(ls)) - 1 : U32)), D.vof(D.vocab(ls)), U32.to_nat(M.base.np(D.vof(D.vocab(ls))))), D.vocab(ls)}

 def prep.nx(p: PR) -> IO(Unit):
   match p:
     case PR{st, us}:
       out(st, us)

 def prep(s: String) -> IO(Unit):
   prep.nx(prep.go(List.take(&2, String, D.lines(s), U32.to_nat(50000))))

 def main() -> IO(Unit):
   do IO<Unit>:
     p : String <- IO.try(String, IO.get_env("CORPUS"))
     s : String <- D.readInput(p)
     prep(s)
