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


def test_lu():
    M=np.random.sample((4,4))
    p,l,u=my_plu(M)
    print(l)
    print(u)
    print(l.dot(u))
    print(p.dot(M))

if __name__ =='__main__':
    test_lu()