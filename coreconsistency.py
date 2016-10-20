'''
Alessandro Bessi
Information Sciences Institute
University of Southern California
bessi@isi.edu

Efficient algorithm for computing the core consistency diagnostic
(CORCONDIA) for the PARAFAC tensor decomposition.

References:
Buis, Paul E., and Wayne R. Dyksen. 
"Efficient vector and parallel manipulation of tensor products." 
ACM Transactions on Mathematical Software (TOMS) 22.1 (1996): 18-23.

Papalexakis, Evangelos E., and Christos Faloutsos. 
"Fast efficient and scalable core consistency diagnostic 
for the parafac decomposition for big sparse tensors." 
2015 IEEE International Conference on Acoustics, 
Speech and Signal Processing (ICASSP). IEEE, 2015.

Bro, Rasmus, and Henk AL Kiers. 
"A new efficient method for determining the number of components in PARAFAC models."
Journal of chemometrics 17.5 (2003): 274-286.

'''


from sktensor import dtensor, cp_als
import numpy as np

def kronecker(matrices, tensor):
    K = len(matrices)
    x = tensor
    for k in range(K):
        M = matrices[k]
        x = x.ttm(M, k)
    return x

def corcondia(tensor, components = 1):
    k = components
    X = tensor
    X_approx_ks, fit, itr, exectimes = cp_als(X, k, init='random')
    
    A = X_approx_ks.U[0]
    B = X_approx_ks.U[1]
    C = X_approx_ks.U[2]

    x = X_approx_ks.totensor()

    Ua , Sa , Va = np.linalg.svd(A)
    Ub , Sb , Vb = np.linalg.svd(B)
    Uc , Sc , Vc = np.linalg.svd(C)

    SaI = np.zeros((Ua.shape[0],Va.shape[0]), float)
    np.fill_diagonal(SaI, Sa)

    SbI = np.zeros((Ub.shape[0],Vb.shape[0]), float)
    np.fill_diagonal(SbI, Sb)

    ScI = np.zeros((Uc.shape[0],Vc.shape[0]), float)
    np.fill_diagonal(ScI, Sc)

    SaI = np.linalg.pinv(SaI)
    SbI = np.linalg.pinv(SbI)
    ScI = np.linalg.pinv(ScI)

    y = kronecker([Ua.transpose(), Ub.transpose(), Uc.transpose()], x)
    z = kronecker([SaI, SbI, ScI], y)
    G = kronecker([Va.transpose(), Vb.transpose(), Vc.transpose()], z)
    
    # print G
    
    C = np.full((k, k, k), 0)
    for i in range(k):
        for j in range(k):
            for l in range(k):
                if i == j == l:
                    C[i][j][l] = 1
    
    c = 0
    for i in range(k):
        for j in range(k):
            for l in range(k):
                c += float(G[i][j][l] - C[i][j][l]) ** 2.0
    
    cc = 100 * (1 - (c / float(k)))
    
    return round(cc)
