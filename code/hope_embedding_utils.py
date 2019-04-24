from sklearn.decomposition import TruncatedSVD
import numpy as np

def embed_layer_hope(layer, embedding_dim):
    beta = 0.01
    svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
    M_g = np.eye(layer.shape[0]) - beta * layer
    M_l = beta * layer
    S = np.dot(np.linalg.inv(M_g), M_l)
    u = svd.fit_transform(S)
    vt = svd.components_
    s = svd.singular_values_
    X1 = np.dot(u, np.diag(np.sqrt(s)))
    X2 = np.dot(vt.T, np.diag(np.sqrt(s)))

    return X1, X2

def reconstruct_layer(u, v, alpha=0.5):
    r = np.dot(u, v.T)
    r[r >= alpha] = 1
    r[r < alpha] = 0
    return r