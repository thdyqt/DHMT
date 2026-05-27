import os
import numpy as np


def read_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Khong tim thay file: {filepath}")

    Q = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.split()

        if len(parts) >= 3:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            Q.append([x, y, z])

    if len(Q) == 0:
        raise ValueError("File diempixel.dat khong co du lieu diem.")

    return np.array(Q)


def chord_length_parameterization(Q):
    m = len(Q)

    distances = np.zeros(m)
    total_length = 0.0

    for i in range(1, m):
        distances[i] = np.linalg.norm(Q[i] - Q[i - 1])
        total_length += distances[i]

    if total_length == 0:
        raise ValueError("Tong chieu dai day cung bang 0.")

    ubar = np.zeros(m)
    ubar[0] = 0.0
    ubar[-1] = 1.0

    for i in range(1, m - 1):
        ubar[i] = ubar[i - 1] + distances[i] / total_length

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
        if U[i] <= u < U[i + 1]:
            return 1.0
        if u == 1.0 and U[i + 1] == 1.0:
            return 1.0
        return 0.0

    left = 0.0
    denom_left = U[i + p] - U[i]

    if denom_left != 0:
        left = ((u - U[i]) / denom_left) * Nip(i, p - 1, u, U)

    right = 0.0
    denom_right = U[i + p + 1] - U[i + 1]

    if denom_right != 0:
        right = ((U[i + p + 1] - u) / denom_right) * Nip(i + 1, p - 1, u, U)

    return left + right


def setup_LST_matrices(filepath, p, n):
    Q = read_data(filepath)
    m = len(Q)

    if m <= n:
        raise ValueError(
            f"Loi: so diem du lieu m = {m} phai lon hon so dinh dieu khien n = {n}."
        )

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
    filepath = "data/diempixel.dat"
    p = 3
    n = 40

    print("=== MODULE 2: TOAN HOC B-SPLINE ===")
    print(f"Thu muc hien tai: {os.getcwd()}")
    print(f"Dang doc file: {filepath}")

    NT_N, NT_Q, U, Q = setup_LST_matrices(filepath, p, n)

    print("-> Dang thiet lap cac ma tran LST...")
    print(f"-> So diem du lieu m = {len(Q)}")
    print(f"-> So dinh dieu khien n = {n}")
    print(f"-> Bac duong cong p = {p}")
    print(f"-> So knot = {len(U)}")
    print(f"-> Ma tran N^T * N co kich thuoc: {NT_N.shape}")
    print(f"-> Ma tran N^T * Q co kich thuoc: {NT_Q.shape}")
    print("-> Module Hieu chay thanh cong.")