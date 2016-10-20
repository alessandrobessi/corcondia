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

    # CORCONDIA
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
