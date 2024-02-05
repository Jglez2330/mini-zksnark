import numpy as np
import galois
from numpy.lib import poly
from scipy.interpolate import lagrange
#x^4 -5y^2*x^2
def main():
    """L_r1 = np.array([
        [0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,-5,0,0,0,0],
        [0,0,0,0,0,1,0],
    ])"""
    L_r1 = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0], 
        [5, 0, 0, 0, 0, 1]
    ])

    R_r1 = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0]
    ])

    O_r1 = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0]
    ])
    # R_r1 = np.array([
    #     [0,1,0,0,0,0,0],
    #     [0,0,0,1,0,0,0],
    #     [0,0,1,0,0,0,0],
    #     [0,0,0,1,0,0,0]
    # ])
    # 
    # O_r1 = np.array([
    #     [0,0,0,1,0,0,0],
    #     [0,0,0,0,1,0,0],
    #     [0,0,0,0,0,1,0],
    #     [0,0,0,0,-1,0,1]
    # ])

    x = 4
    y = -2
    v1 = x * x
    v2 = v1 * v1         # x^4
    v3 = -5*y * y
    out = v3*v1 + v2    # -5y^2 * x^2

    #wwitness = np.array([1, x, y, v1, v2, v3, out])
    witness = np.array([1, 3, 35, 9, 27, 30])


    assert all(np.equal(np.matmul(L_r1, witness) * np.matmul(R_r1, witness), np.matmul(O_r1, witness))), "not equal"
    print(all(np.equal(np.matmul(L_r1, witness) * np.matmul(R_r1, witness), np.matmul(O_r1, witness))))


    x = np.array([1, 2, 3, 4])
    A = calculate_poly(x, L_r1)

    print(A)
    print(A[0](1))
    print(A[0](2))
    #
    #
    #
    # 
    #
    #
    #
    #
    # prime = 7
    # #Map to a prime finite field
    # GF = galois.GF(prime)
    #
    # L_galois = GF(convert_negtives(L_r1, prime))
    # R_galois = GF(convert_negtives(R_r1, prime))
    # O_galois = GF(convert_negtives(O_r1, prime))
    #
    # x = GF(4)
    # y = GF(prime-2) # we are using 79 as the field size, so 79 - 2 is -2
    # v1 = x * x
    # v2 = v1 * v1         # x^4
    # v3 = GF(prime-5)*y * y
    # out = v3*v1 + v2    # -5y^2 * x^2
    #
    # witness = GF(np.array([1, x, y, v1, v2, v3, out]))
    #
    # assert all(np.equal(np.matmul(L_galois, witness) * np.matmul(R_galois, witness), np.matmul(O_galois, witness))), "not equal"
    #
def convert_negtives(matrix, prime):
    matrix[matrix < 0] += prime
    return matrix
def calculate_poly(x, y_matrix):
    y_matrix = y_matrix.T
    poly = []
    for y in y_matrix:
        poly.append(lagrange(x,y))
    return poly
 
main()
