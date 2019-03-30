import numpy as np

#高斯消元的lu算法
def my_lu(A):
    U = np.array(A)
    rows,columns=U.shape
    assert rows==columns
    L=np.eye(columns)
    for j in range(columns-1):
        L[j+1:,j]=U[j+1:,j]/U[j,j]
        U[j+1:,j:]-=L[j+1:,j].reshape(-1,1)*U[j,j:]
    return L,U

# plu算法
def my_plu(A):
    U = np.array(A,dtype='float')
    rows, columns = U.shape
    if rows != columns:
        raise Exception('Square matrix is accepted')
    P = np.eye(columns)
    L = np.eye(columns)
    for j in range(columns - 1):
        idx=max(range(j,columns),key=lambda i:abs(U[i][j]))
        if idx!=j:
            P[[j, idx], :] = P[[idx, j], :]
            U[[j,idx],:]=U[[idx,j],:]
            L[[j,idx],:j]=L[[idx,j],:j]
        if abs(U[j,j])<=np.spacing(0):
            raise Exception('Nonsingular matrix is accepted')
        L[j + 1:, j] = U[j + 1:, j] / U[j, j]
        U[j + 1:, j:] -= L[j + 1:, j].reshape(-1, 1) * U[j, j:]
    return P, L, U

def lu_solve(A,b):
    P,L,U=my_plu(A)
    Pb=P.dot(b)
    b_len=len(b)
    x=np.empty(b_len)
    # y=np.empty(b_len)
    # for i in range(b_len):
    #     y[i]=Pb[i]-np.sum(L[i,:i]*y[:i])
    # for i in range(b_len-1,-1,-1):
    #     x[i]=(y[i]-np.sum(U[i,i+1:]*x[i+1:]))/U[i,i]

    # forward substitution and back substitution
    for i in range(b_len):
        x[i]=Pb[i]-np.sum(L[i,:i]*x[:i])
    for i in range(b_len-1,-1,-1):
        x[i]=(x[i]-np.sum(U[i,i+1:]*x[i+1:]))/U[i,i]
    return x

def test_lu():
    M=np.random.sample((4,4))
    p,l,u=my_plu(M)
    print(l)
    print(u)
    print(l.dot(u))
    print(p.dot(M))

def test_lu_solve():
    A = [[2, 3, -5],
         [1, -2, 1],
         [3, 1, 3]]
    b = [3, 0, 7]
    # x=[10/7,1,4/7]
    print(lu_solve(A, b))

if __name__ =='__main__':
    print('test lu:')
    test_lu()
    print('test lu solve')
    test_lu_solve()