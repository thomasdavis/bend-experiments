import Base
import ./tape.bend as T

# Data prep for the microgpt port: file IO, line splitting, vocab
# (sorted unique chars), char->id encoding, BOS-wrapped capped docs.
# All loops are single self-recursive defs over shrinking structure.

def chle2(z: Cmp) -> Bool:
  match z:
    case LT{}:
      True{}
    case EQ{}:
      True{}
    case GT{}:
      False{}

 def ccmp.p(p: (Char & Char) & Cmp) -> Cmp:
   ((a2, b2), c) = p
   c

 def ccmp(a: Char, b: Char) -> Cmp:
   ccmp.p(Char.cmp(a, b))
 def chle.p(p: (Char & Char) & Cmp) -> Bool:
   ((a2, b2), c) = p
   chle2(c)

 def chle(a: Char, b: Char) -> Bool:
   chle.p(Char.cmp(a, b))

def str.chars(s: String) -> +List<Char>:
  match s:
    case SNil{}:
      Nil{}
    case SCon{h, t}:
      Con{h, str.chars(t)}

def mkstr.go(+cs: +List<Char>, acc: String) -> String:
  match cs:
    case Nil{}:
      acc
    case Con{h, t}:
      mkstr.go(t, SCon{h, acc})

def mkstr(+cs: +List<Char>) -> String:
  mkstr.go(List.reverse(&2, Char, cs), SNil{})

def lines.go(s: String, +cur: +List<Char>, +acc: +List<String>) -> +List<String>:
  match s:
    case SNil{}:
      List.reverse(&2, String, Con{mkstr(cur), acc})
    case SCon{Chr{10}, t}:
      lines.go(t, Nil{}, Con{mkstr(cur), acc})
    case SCon{h, t}:
      lines.go(t, Con{h, cur}, acc)

 def lines(s: String) -> +List<String>:
   lines.go(s, Nil{}, Nil{})

 def dedup.go(+t: +List<Char>, +prev: Char, +cur: Char, z: Cmp) -> +List<Char>:
   match t:
     case Nil{}:
       match z:
         case EQ{}:
           Nil{}
         case LT{}:
           Con{cur, Nil{}}
         case GT{}:
           Con{cur, Nil{}}
     case Con{+h3, +t3}:
       match z:
         case EQ{}:
           dedup.go(t3, prev, h3, ccmp(h3, prev))
         case LT{}:
           Con{cur, dedup.go(t3, cur, h3, ccmp(h3, cur))}
         case GT{}:
           Con{cur, dedup.go(t3, cur, h3, ccmp(h3, cur))}

 def dedup(+cs: +List<Char>) -> +List<Char>:
   match cs:
     case Nil{}:
       Nil{}
     case Con{h, t}:
       match t:
         case Nil{}:
           Con{h, Nil{}}
         case Con{h2, t2}:
           Con{h, dedup.go(t2, h, h2, ccmp(h2, h))}

def vocab.chars(+docs: +List<String>) -> +List<Char>:
  match docs:
    case Nil{}:
      Nil{}
    case Con{d, t}:
      List.append(&2, Char, str.chars(d), vocab.chars(t))

def vocab(+docs: +List<String>) -> +List<Char>:
  dedup(List.sort(~Char, ~chle, vocab.chars(docs)))

 def vof(+us: +List<Char>) -> U32:
   (U32.from_nat(List.length(&2, Char, us)) + 1 : U32)

 def enc.go(+t: +List<Char>, +c: Char, +n: Nat, z: Cmp) -> U32:
   match t:
     case Nil{}:
       match z:
         case EQ{}:
           U32.from_nat(n)
         case LT{}:
           0
         case GT{}:
           0
     case Con{h2, t2}:
       match z:
         case EQ{}:
           U32.from_nat(n)
         case LT{}:
           enc.go(t2, c, 1n + n, ccmp(c, h2))
         case GT{}:
           enc.go(t2, c, 1n + n, ccmp(c, h2))

 def enc.ch(+c: Char, +us: +List<Char>) -> U32:
   match us:
     case Nil{}:
       0
     case Con{u, t}:
       enc.go(t, c, 0n, ccmp(c, u))
 def enc.ids(+cs: +List<Char>, +us: +List<Char>) -> +List<U32>:
   match cs:
     case Nil{}:
       Nil{}
     case Con{c, t}:
       Con{enc.ch(c, us), enc.ids(t, us)}

def enc.wrap(+ids: +List<U32>, +bos: U32) -> +List<U32>:
  List.take(&2, U32, Con{bos, List.append(&2, U32, ids, Con{bos, Nil{}})}, 17n)

def enc.doc(s: String, +us: +List<Char>, +bos: U32) -> +List<U32>:
  enc.wrap(enc.ids(str.chars(s), us), bos)

def enc.docs(+lns: +List<String>, +us: +List<Char>, +bos: U32) -> +List<+List<U32>>:
  match lns:
    case Nil{}:
      Nil{}
    case Con{l, t}:
      Con{enc.doc(l, us, bos), enc.docs(t, us, bos)}

def read.res(f: File, r: Result<&1, &1, U32 & String, String>) -> IO((File & String)):
  match r:
    case Done{value}:
      do IO<(File & String)>:
        return (f, value)
    case Fail{(code, message)}:
      do IO<(File & String)>:
        IO.die((File & String), code, message)

def read.got(p: File & Result<&1, &1, U32 & String, String>) -> IO((File & String)):
  match p:
    case (f, r):
      read.res(f, r)

def read.closed(p: Unit & String) -> IO(String):
  match p:
    case (u, s):
      do IO<String>:
        return s

def read.body(p: File & String) -> IO(String):
  match p:
    case (f, s):
      do IO<String>:
        u : Unit <- File.close(f)
        read.closed((u, s))
def read.file(f: File) -> IO(String):
  do IO<String>:
    fr : (File & Result<&1, &1, U32 & String, String>) <- File.read(f, 300000)
    fs : (File & String) <- read.got(fr)
    read.body(fs)


def read.opened(r: Result<&1, &1, U32 & String, File>) -> IO(String):
  match r:
    case Done{value}:
      read.file(value)
    case Fail{(code, message)}:
      do IO<String>:
        IO.die(String, code, message)

def readInput(path: String) -> IO(String):
  do IO<String>:
    r : Result<&1, &1, U32 & String, File> <- File.open(path, "r")
    read.opened(r)
