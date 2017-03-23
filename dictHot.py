import numpy 

def one_hot(i):
    a = numpy.zeros(21, 'uint8')
    a[i-1] = 1
    return a


def oneHot(char):
    try:
        return indict[char]
    except:
        return one_hot(21)

indict={'A':one_hot(1),
              'C':one_hot(2),
              'D':one_hot(3),
              'E':one_hot(4),
              'F':one_hot(5),
              'G':one_hot(6),
              'H':one_hot(7),
              'I':one_hot(8),
              'K':one_hot(9),
              'L':one_hot(10),
              'M':one_hot(11),
              'N':one_hot(12),
              'P':one_hot(13),
              'Q':one_hot(14),
              'R':one_hot(15),
              'S':one_hot(16),
              'T':one_hot(17),
              'V':one_hot(18),
              'W':one_hot(19),
              'Y':one_hot(20),
              '0':one_hot(21)}
