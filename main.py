import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp2d
from tqdm import tqdm
from matplotlib.patches import FancyArrowPatch


if __name__ == "__main__":
    
    m, n = 20, 20
    step_size_m = 1/m
    step_size_n = 1/n

    T0 = 500
    Tinf = 300

    HdXK = 2.5

    t0 = time.time()

    A = np.zeros([(m * n), (m*n)])
    C = np.zeros((m*n))

    for i in range(0, m):
        for j in range(0, n):
            node = i*n + j
            neigh = [(i-1)*n + j, (i+1)*n + j, i*n + (j-1), i*n + (j+1)]

            if i < m-1:
                A[node, node] = -4
                for neigh_node in neigh:
                    if neigh_node >= 0 and neigh_node < m*n:
                        A[node, neigh_node] = 1
                    else:
                        C[node] -= T0
            else:
                A[node, node] = -2*(HdXK + 2)

                neigh[1] = neigh[0]

                for neigh_node in neigh:
                    if neigh_node >= 0 and neigh_node < m*n:
                        A[node, neigh_node] += 1
                    else:
                        C[node] -= T0

                C[node] -= 2*HdXK*Tinf

    T = np.linalg.solve(A, C)
    # T = np.linalg.inv(A) @ C
    t1 = time.time()
    print(f"Time taken: {t1-t0}sec")

    print("A:")
    print(A)
    
    print("C:")
    print(C)

    T_mesh = T.reshape((m, n))
    print("T:")
    # print(T)
    print(T_mesh)

    T_mesh = np.pad(T_mesh, 1, 'constant', constant_values=T0)
    T_mesh[-1,:] = Tinf
    m, n = T_mesh.shape

    plot = False
    max_iters = 1000
    mean_error = 0
    for it in tqdm(range(max_iters)):
        T_new = T_mesh.copy()
        for i in range(1, T_mesh.shape[0]-1):
            for j in range(1, T_mesh.shape[1]-1):
                T_new[i,j] = (T_mesh[i-1,j] + T_mesh[i+1,j] + T_mesh[i,j-1] + T_mesh[i,j+1])/4

        mean_error = np.mean(np.abs(T_new - T_mesh))
        T_mesh = T_new.copy()

    
        if plot:
            plt.clf()
            # scipy interp. cubic
            y = np.linspace(0, 1, m)
            x = np.linspace(0, 1, n)
            f = interp2d(x, y, T_mesh, kind='cubic')
            xnew =  np.linspace(0, 1, 400)
            ynew =  np.linspace(0, 1, 400)
            data1 = f(xnew,ynew)
            Xn, Yn = np.meshgrid(xnew, ynew)
            # plt.pcolormesh(Xn, Yn, data1, cmap='coolwarm')

            yd, xd = np.gradient(T_mesh)
            def func_to_vectorize(_x, _y, dx, dy, scaling=0.01):
                grad = np.array([dx, dy])
                grad /= np.linalg.norm(grad) + 1e-10
                dx, dy = grad*scaling
                plt.arrow(_x, _y, -dx, -dy, fc="k", ec="k", head_width=0.01, head_length=0.01, alpha=0.5)
            vectorized_arrow_drawing = np.vectorize(func_to_vectorize)

            plt.contourf(Xn, Yn, data1, cmap='coolwarm')
            orig_x, orig_y = np.meshgrid(x, y)
            vectorized_arrow_drawing(orig_x, orig_y, xd, yd, scaling=step_size_m/5)

            plt.suptitle(f"Temperature distribution after {it+1} iterations. Using {m}x{n} grid")
            plt.title(f"Mean error: {mean_error:.4f}")
            plt.colorbar()
            plt.tight_layout()
            plt.pause(0.05)
            plt.savefig(f"tmp/img{it:>04d}.png")

        if mean_error < 0.01:
            break

    plt.show()


    plt.clf()
    # scipy interp. cubic
    y = np.linspace(0, 1, m)
    x = np.linspace(0, 1, n)
    f = interp2d(x, y, T_mesh, kind='cubic')
    xnew =  np.linspace(0, 1, 400)
    ynew =  np.linspace(0, 1, 400)
    data1 = f(xnew,ynew)
    Xn, Yn = np.meshgrid(xnew, ynew)
    # plt.pcolormesh(Xn, Yn, data1, cmap='coolwarm')

    yd, xd = np.gradient(T_mesh)
    def func_to_vectorize(_x, _y, dx, dy, scaling=0.01):
        grad = np.array([dx, dy])
        grad /= np.linalg.norm(grad) + 1e-10
        dx, dy = grad*scaling
        plt.arrow(_x, _y, -dx, -dy, fc="k", ec="k", head_width=0.01, head_length=0.01, alpha=0.5)
    vectorized_arrow_drawing = np.vectorize(func_to_vectorize)

    plt.contourf(Xn, Yn, data1, cmap='coolwarm')
    orig_x, orig_y = np.meshgrid(x, y)
    vectorized_arrow_drawing(orig_x, orig_y, xd, yd, scaling=step_size_m/5)

    plt.suptitle(f"Temperature distribution after {it+1} iterations. Using {m}x{n} grid")
    plt.title(f"Mean error: {mean_error:.4f}ºC")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"Final.png")
