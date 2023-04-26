import matrix as ma
import numpy as np

# pour lancer le test : py -m pytest test.py -v

# MATRIX TEST
# SHAPE TEST
def test_shape():
    
    ms = randomMatrix.addMatrix(np.random.randint(2, 10), False, True)
    
    for tab in ms.values():
        assert ma.Matrix(tab).shape == tab.shape

# TRANSPOSE TEST
def test_T():
    
    mt = randomMatrix.addMatrix(np.random.randint(2, 10), False, True)

    for tab in mt.values():
        assert print(ma.Matrix(tab).T()) == print(np.array(tab).T)

# ADD(+) TEST
def test_add():

    madd = randomMatrix.addMatrix(np.random.randint(2, 10))

    for tab in madd.values():
        assert print(ma.Matrix(tab[0]) + ma.Matrix(tab[1])) == print(np.array(tab[0]) + np.array(tab[1]))

# RADD(+) TEST
def test_radd():

    mradd = randomMatrix.addMatrix(np.random.randint(2, 10))

    for tab in mradd.values():
        assert print(ma.Matrix(tab[0]) + ma.Matrix(tab[1])) == print(np.array(tab[0]) + np.array(tab[1]))

# SUB(-) TEST
def test_sub():

    msub = randomMatrix.addMatrix(np.random.randint(2, 10))

    for tab in msub.values():
        assert print(ma.Matrix(tab[0]) - ma.Matrix(tab[1])) == print(np.array(tab[0]) - np.array(tab[1]))

# RSUB(-) TEST
def test_rsub():

    mrsub = randomMatrix.addMatrix(np.random.randint(2, 10))

    for tab in mrsub.values():
        assert print(ma.Matrix(tab[0]) - ma.Matrix(tab[1])) == print(np.array(tab[0]) - np.array(tab[1]))

# DIV(-) TEST
def test_truediv():

    mdiv = randomMatrix.addMatrix(np.random.randint(2, 10))

    for tab in mdiv.values():
        assert print(ma.Matrix(tab[0]) / ma.Matrix(tab[1])) == print(np.array(tab[0]) / np.array(tab[1]))

# RDIV(-) TEST
def test_rtruediv():

    mrdiv = randomMatrix.addMatrix(np.random.randint(2, 10))

    for tab in mrdiv.values():
        assert print(ma.Matrix(tab[0]) / ma.Matrix(tab[1])) == print(np.array(tab[0]) / np.array(tab[1]))

# MUL(@) TEST
def test_mul():

    mmul = randomMatrix.addMatrix(np.random.randint(2, 10), True)

    for tab in mmul.values():
        assert print(ma.Matrix(tab[0]) * ma.Matrix(tab[1])) == print(np.array(tab[0]) @ np.array(tab[1]))

# RMUL(@) TEST
def test_rmul():

    mrmul = randomMatrix.addMatrix(np.random.randint(2, 10), True)

    for tab in mrmul.values():
        assert print(ma.Matrix(tab[0]) * ma.Matrix(tab[1])) == print(np.array(tab[0]) @ np.array(tab[1]))

###############################################################################################################

# VECTOR TEST
# SHAPE VECTOR TEST
def test_Vector_shape():

    ms = randomVector.addVector(np.random.randint(2, 10), False, True)

    for tab in ms.values():
        assert ma.Vector(tab).shape == tab.shape

# TRANSPOSE VECTOR TEST
def test_Vector_T():

    vt = randomVector.addVector(np.random.randint(2, 10), False, True)

    for tab in vt.values():
        assert print(ma.Vector(tab).T()) == print(np.array(tab).T)

# DOT(@) VECTOR TEST
def test_Vector_dot():

    vmul = randomVector.addVector(np.random.randint(2, 10), True)

    for tab in vmul.values():
        assert print(ma.Vector(tab[0]) * ma.Vector(tab[1])) == print(np.array(tab[0]) @ np.array(tab[1]))

# MUL(@) VECTOR TEST
def test_Vector_mul():

    vmul = randomVector.addVector(np.random.randint(2, 10), True)

    for tab in vmul.values():
        assert print(ma.Vector(tab[0]) * ma.Vector(tab[1])) == print(np.array(tab[0]) @ np.array(tab[1]))

# ADD(+) VECTOR TEST
def test_Vector_add():

    vadd = randomVector.addVector(np.random.randint(2, 10))

    for tab in vadd.values():
        assert print(ma.Vector(tab[0]) + ma.Vector(tab[1])) == print(np.array(tab[0]) + np.array(tab[1]))

# RADD(+) VECTOR TEST
def test_Vector_radd():

    vradd = randomVector.addVector(np.random.randint(2, 10))

    for tab in vradd.values():
        assert print(ma.Vector(tab[0]) + ma.Vector(tab[1])) == print(np.array(tab[0]) + np.array(tab[1]))

# SUB(-) VECTOR TEST
def test_Vector_sub():

    vsub = randomVector.addVector(np.random.randint(2, 10))

    for tab in vsub.values():
        assert print(ma.Vector(tab[0]) - ma.Vector(tab[1])) == print(np.array(tab[0]) - np.array(tab[1]))

# RSUB(-) VECTOR TEST
def test_Vector_rsub():

    vrsub = randomVector.addVector(np.random.randint(2, 10))

    for tab in vrsub.values():
        assert print(ma.Vector(tab[0]) - ma.Vector(tab[1])) == print(np.array(tab[0]) - np.array(tab[1]))

# DIV(-) VECTOR TEST
def test_Vector_truediv():

    vdiv = randomVector.addVector(np.random.randint(2, 10))

    for tab in vdiv.values():
        assert print(ma.Vector(tab[0]) / ma.Vector(tab[1])) == print(np.array(tab[0]) / np.array(tab[1]))

# RDIV(-) VECTOR TEST
def test_Vector_rtruediv():

    vrdiv = randomVector.addVector(np.random.randint(2, 10))

    for tab in vrdiv.values():
        assert print(ma.Vector(tab[0]) / ma.Vector(tab[1])) == print(np.array(tab[0]) / np.array(tab[1]))


# CLASS RANDOMMATRIX: RANDOM MATRIX CREATION
class randomMatrix:
  def addMatrix(tab, mulOn = False, singleTab = False):

    matrix = {}
    i = 0
    while i <= tab :
      row = np.random.randint(2, 10)
      col = np.random.randint(2, 10)
      mul = np.random.randint(2, 10)
      matrix[i] = [] if not singleTab else 0

      if singleTab:
        matrix[i] = np.random.uniform(-100, 100, size=(row, col))
     
      else :
        matrix[i].append(np.random.uniform(-100, 100, size=((row, col) if not mulOn else (row, mul))))
        matrix[i].append(np.random.uniform(-100, 100, size=((row, col) if not mulOn else (mul, col))))
      
      i += 1
    return matrix

class randomVector:
  def addVector(tab, mulOn = False, singleTab = False):

    vector = {}
    i = 0
    while i <= tab :
      row = 1 if (np.random.choice([True, False])) else np.random.randint(2, 10)
      col = 1 if row > 1 else np.random.randint(2, 10)
      mul = np.random.randint(2, 10)
      vector[i] = [] if not singleTab else 0

      if singleTab:
        vector[i] = np.random.uniform(-100, 100, size=(row, col))
      
      else :
        vector[i].append(np.random.uniform(-100, 100, size=((row, col) if not mulOn else (1, mul))))
        vector[i].append(np.random.uniform(-100, 100, size=((row, col) if not mulOn else (mul, 1))))
      
      i += 1
    return vector