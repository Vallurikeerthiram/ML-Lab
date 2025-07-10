# Write a program that accepts a square matrix A and a positive integer m as arguments and returns A^m.
def power(sq,m):
    n=len(sq)
    result=[[1 if i==j else 0 for j in range(n)] for i in range(n)] #identity matrix

    def mul(m1,m2):     # for matrix multiplication
        res =[[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    res[i][j]+=m1[i][k]*m2[k][j]
        return res
    while m>0:          # exponentiation by squaring
        if m%2==1:
            result=mul(result,sq)
        sq=mul(sq,sq)
        m//=2
    return result

def inMatrix(n):
    mat = []
    for i in range(n):
        arr= list(map(int, input("enter numbers seperated with spaces for row "+str(i+1)+": ").split()))
        if (len(arr)!=n):
            raise ValueError("Each row must have exactly " + str(n) + " elements.")
        mat.append(arr)
    return mat

def prnt(mat):
    for i in mat:
        print(i)

n = int(input("enter the size of square matrix"))
sq=inMatrix(n)
m = int(input("enter the power"))
prnt(power(sq,m))