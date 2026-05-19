import numpy as np

def read_data(filepath):
    Q = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.split()
            Q.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(Q)

def chord_length_parameterization(Q):
    m = len(Q)
    L = 0
    for i in range(1, m):
        L += np.linalg.norm(Q[i] - Q[i-1])
    
    ubar = np.zeros(m)
    ubar[0] = 0.0
    ubar[-1] = 1.0
    for i in range(1, m-1):
        ubar[i] = ubar[i-1] + np.linalg.norm(Q[i] - Q[i-1]) / L
    return ubar

def generate_knot_vector(n, p):
    m_knots = n + p + 1
    U = np.zeros(m_knots)
    for i in range(p + 1):
        U[i] = 0.0
        U[m_knots - 1 - i] = 1.0
    for i in range(p + 1, n):
        U[i] = (i - p) / (n - p)
    return U

def Nip(i, p, u, U):
    if p == 0:
        if U[i] <= u < U[i+1]: return 1.0
        if u == U[i+1] and u == 1.0: return 1.0
        return 0.0
    
    left = 0.0
    if (U[i+p] - U[i]) != 0:
        left = ((u - U[i]) / (U[i+p] - U[i])) * Nip(i, p-1, u, U)
        
    right = 0.0
    if (U[i+p+1] - U[i+1]) != 0:
        right = ((U[i+p+1] - u) / (U[i+p+1] - U[i+1])) * Nip(i+1, p-1, u, U)
        
    return left + right

def setup_LST_matrices(filepath, p, n):
    Q = read_data(filepath)
    m = len(Q)
    
    if m <= n:
        raise ValueError("Loi: So luong diem du lieu (m) phai lon hon so dinh dieu khien (n).")

    ubar = chord_length_parameterization(Q)
    U = generate_knot_vector(n, p)
    
    N = np.zeros((m, n))
    for k in range(m):
        for i in range(n):
            N[k, i] = Nip(i, p, ubar[k], U)
            
    NT_N = np.dot(N.T, N)
    NT_Q = np.dot(N.T, Q)
    
    return NT_N, NT_Q, U, Q

if __name__ == "__main__":
    filepath = 'data/diempixel.dat'
    p = 3  
    n = 40 
    
    print("=== MODULE 2: TOAN HOC B-SPLINE  ===")
    print("-> Dang thiet lap cac ma tran LST...")
    NT_N, NT_Q, U, Q = setup_LST_matrices(filepath, p, n)
    
    print(f"-> So diem du lieu m = {len(Q)}")
    print(f"-> So dinh dieu khien n = {n}")
    print(f"-> Ma tran N^T * N co kich thuoc: {NT_N.shape}")
    print(f"-> Ma tran N^T * Q co kich thuoc: {NT_Q.shape}")
