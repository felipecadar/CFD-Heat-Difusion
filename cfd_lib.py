import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp2d

def interp(mesh, new_shape):

    y = np.linspace(0, 1, mesh.shape[0])
    x = np.linspace(0, 1, mesh.shape[1])
    f = interp2d(x, y, mesh, kind='cubic')
    xnew =  np.linspace(0, 1, new_shape[0])
    ynew =  np.linspace(0, 1, new_shape[1])
    data1 = f(xnew,ynew)
    Xn, Yn = np.meshgrid(xnew, ynew)

    return data1, Xn, Yn

def PlotMesh(T_mesh, save_fig=False, animate=False, it=-1, mean_error=-1):
    plt.clf()

    m,n = T_mesh.shape

    data1, Xn, Yn = interp(T_mesh, [100,100])
    plt.contourf(Xn, Yn, data1, levels=10, cmap='coolwarm')

    def func_to_vectorize(_x, _y, dx, dy, scaling=0.01):
        grad = np.array([dx, dy])
        grad /= np.linalg.norm(grad) + 1e-10
        dx, dy = grad*scaling
        plt.arrow(_x, _y, -dx, -dy, fc="k", ec="k", head_width=0.01, head_length=0.01, alpha=0.5)

    vectorized_arrow_drawing = np.vectorize(func_to_vectorize)

    step_size = 1/m

    yd, xd = np.gradient(T_mesh)
    orig_x, orig_y = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, n))
    vectorized_arrow_drawing(orig_x, orig_y, xd, yd, scaling=step_size/5)

    if it >= 0:
        plt.suptitle(f"Temperature distribution after {it+1} iterations. Using {m-2}x{n-2} grid")
    if mean_error >= 0:
        plt.title(f"Mean error: {mean_error:.4f}")
    
    plt.colorbar()
    plt.tight_layout()

    if animate:
        plt.pause(0.005)

    if save_fig:
        plt.savefig(f"tmp/img{it:>04d}.png")

def CFD(grid_size=[20,20], T0=500, Tinf=300, HdXK=2.5, error_th=0.01, max_iters=1000, save_plot=True, animate=False):

    m, n = grid_size

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

    t1 = time.time()

    T_mesh = T.reshape((m, n))

    T_mesh = np.pad(T_mesh, 1, 'constant', constant_values=T0)
    T_mesh[-1,:] = Tinf


    mean_error = 0
    for it in range(max_iters):
        T_new = T_mesh.copy()
        for i in range(1, T_mesh.shape[0]-1):
            for j in range(1, T_mesh.shape[1]-1):
                T_new[i,j] = (T_mesh[i-1,j] + T_mesh[i+1,j] + T_mesh[i,j-1] + T_mesh[i,j+1])/4

        mean_error = np.mean(np.abs(T_new - T_mesh))
        T_mesh = T_new.copy()
    
        if animate:
            PlotMesh(T_mesh, save_fig=False, animate=True, it=it, mean_error=mean_error)

        if mean_error < error_th:
            break

    if animate:
        plt.show()

    PlotMesh(T_mesh, save_fig=False, animate=False, it=it, mean_error=mean_error)
    if save_plot:
        plt.savefig(f"Final.png")

    return plt.gcf()

if __name__ == "__main__":
    CFD()