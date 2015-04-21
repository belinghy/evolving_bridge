import numpy as np
import matplotlib
from matplotlib.pyplot import plot, axis, arrow, xkcd, savefig, grid, clf

# Yield strength of steel
Fy = 344 * pow(10,6)
# Elastic modulus of steel
E = 210 * pow(10, 9)
# Outer diameters of some optional sizes, in meters
OUTER_DIAM = [(x+1.0)/100 for x in range(10)]
# Thickness of the wall sections of optional sizes, in meters
THICK = [d/15 for d in OUTER_DIAM]
# Cross sectional area in m^2
AREA_SEC = [np.pi*pow(d/2, 2) - np.pi*pow(d/2-d/15, 2) for d in OUTER_DIAM]
# Moment of inertia, in m^4
I_SEC = [np.pi*(pow(d, 4) - pow((d - 2*d/15), 4))/64 for d in OUTER_DIAM]
# Weight per length, kg/m
WEIGHT = [a*7870 for a in AREA_SEC]
# Factor of safety
FOS_MIN = 1.25

# Predefined Joints
Pre = {}
Pre["Joints"] = np.array([[0,0,0], [2,0,0], [4,0,0], [6,0,0], [8,0,0], [10,0,0]])
# Predefined Load
Pre_force = -20000.0
Pre["Load"] = np.array([[0,0,0], [0,Pre_force,0], [0,Pre_force,0], [0,Pre_force,0], [0,Pre_force,0], [0,0,0]]).T



def force_eval(D):
    """
    This function takes as input a dictionary D that defines the following variables:
        "Re":    encodes the permitted motion of each joint. Each joint is represented by 
                 a vector of three binary variables, indicating whether or not it is supported
                 in the, x, y and z direction.
        "Joints": This indicates the x, y, and z coordinates of each joint.
        "Load":  This indicates the x, y and z loadings placed on each joint.
        "Beams":   This indicates how the joints are connected
        "E":     This indicates the elastic modullus of each connecting member described above.
        "A":     This indicates the area of each connecting member described above
        
    The function returns three variables.  
        F:       The forces present in each member
        U:       Node displacements
        R:       Node reactions
    """
    Tj = np.zeros([3, np.size(D["Beams"], axis=1)])
    w = np.array([np.size(D["Re"], axis=0), np.size(D["Re"], axis=1)])
    SS = np.zeros([3*w[1], 3*w[1]])
    U = 1.0 - D["Re"]
    
    # This identifies joints that are unsupported, and can therefore be loaded
    ff = np.where(U.T.flat == 1)[0]
    
    # Step through the each member in the truss, and build the global stiffness matrix
    for i in range(np.size(D["Beams"], axis=1)):
        H = D["Beams"][:, i]
        C = D["Joints"][:, H[1]] - D["Joints"][:, H[0]]
        Le = np.linalg.norm(C)
        T = C/Le
        s = np.outer(T, T)
        G = D["E"][i]*D["A"][i]/Le
        ss = G*np.concatenate((np.concatenate((s, -s), axis=1), np.concatenate((-s, s), axis=1)), axis=0)
        Tj[:, i] = G*T
        e = range((3*H[0]), (3*H[0] + 3)) + range((3*H[1]), (3*H[1] + 3))
        for ii in range(6):
            for j in range(6):
                SS[e[ii], e[j]] += ss[ii, j]

    SSff = np.zeros([len(ff), len(ff)])
    for i in range(len(ff)):
        for j in range(len(ff)):
            SSff[i,j] = SS[ff[i], ff[j]]
                
    Loadff = D["Load"].T.flat[ff]
    Uff = np.linalg.solve(SSff, Loadff)
    
    ff = np.where(U.T==1)
    for i in range(len(ff[0])):
        U[ff[1][i], ff[0][i]] = Uff[i]
    
    F = np.sum(np.multiply(Tj, U[:, D["Beams"][1,:]] - U[:, D["Beams"][0,:]]), axis=0)
    R = np.sum(SS*U.T.flat[:], axis=1).reshape([w[1], w[0]]).T
    
    return F, U, R
    
    
def fos_eval(truss):
    D = {}
    
    M = len(truss["Beams"].T)
    N = len(truss["Joints"].T)
    
    # "1" means it is fixed, "0" means it is able to move, (x,y,z)
    D["Re"] = np.array([[1,1,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1], [1,1,1]]).T
    for _ in range(N-len(Pre["Joints"])):
        D["Re"] = np.column_stack([D["Re"], [0,0,1]])
        
    # Add loads, applying load to second and fourth beam in downward direction
    D["Load"] = Pre["Load"]
    for _ in range(N-len(Pre["Joints"])):
        D["Load"] = np.column_stack([D["Load"], [0,0,0]])

    # Add area information from truss
    D["A"] = []
    for member_size in truss["Sizes"]:
        D["A"].append(AREA_SEC[int(member_size)])
    D["Joints"] = truss["Joints"]
    D["Beams"] = truss["Beams"]
    D["E"] = E*np.ones(M)
    
    # Do force analysis
    F, U, R = force_eval(D)
    
    # Calculate lengths
    L = np.zeros(M)
    for i in range(M):
        L[i] = np.linalg.norm(D["Joints"][:, D["Beams"][0, i]] - D["Joints"][:, D["Beams"][1, i]])
    
    # Calculate FOS's
    FOS = np.zeros(M)
    for i in range(len(F)):
        FOS[i] = D["A"][i] * Fy / F[i]
        if FOS[i] < 0:
            FOS[i] = min(np.pi * np.pi * E * I_SEC[int(truss["Sizes"][i] - 1)] / (L[i]*L[i]) / -F[i], -FOS[i])
    
    return FOS, F, D["Load"]
    
    
def plot_truss(truss, FOS, F, Load):
    # Collect some information
    M = len(truss["Beams"].T)    
    N = len(truss["Joints"].T)

    Hm = []
    # Plot every member
    for i in range(M):
        p1 = truss["Joints"][:, truss["Beams"][0, i]]
        p2 = truss["Joints"][:, truss["Beams"][1, i]]
        if FOS[i] > 1:
            color = 'b'
        else:
            color = 'r'
        if F[i] > 0:
            lst = '--'
        else:
            lst = '-'
        Hm.append(plot([p1[0], p2[0]], [p1[1], p2[1]], color, linewidth=truss["Sizes"][i]+1, linestyle = lst))
        axis('equal')
        
    # Plot supports
    Hs = []
    for i in range(len(Pre["Joints"])):
        Hs.append(plot(truss["Joints"][0, i], truss["Joints"][1, i], 'ks', ms=15))
    
    # Plot loads
    Hl = []
    for i in range(len(Load.T)):
        if (Load[1,i] < 0):
            Hl.append(arrow(truss["Joints"][0, i], truss["Joints"][1, i] + 1.0, 0.0, -0.5, 
                        fc="m", ec="m", head_width=0.3, head_length=0.6, width=0.1, zorder=3))
        
    # Plot every joint
    Hj = []
    for i in range(N-len(Pre["Joints"])):
        Hj.append(plot(truss["Joints"][0, i + len(Pre["Joints"])], truss["Joints"][1, i + len(Pre["Joints"])], 'ko', ms=10))
    
    return Hm, Hj, Hl, Hs    
    
    
def init_truss(N):
    truss = {}
    
    # Predefined joints
    truss["Joints"] = Pre["Joints"]
    
    # Number of joints
    truss["N"] = N
    
    for i in range(N-len(Pre["Joints"])):
        x = np.random.randint(0, 9)
        y = np.random.randint(1, 9)
        truss["Joints"] = np.vstack([truss["Joints"], np.array([x,y,0])])
    
    truss["Joints"] = truss["Joints"].T
    edges = matplotlib.tri.Triangulation(truss["Joints"][0, :], truss["Joints"][1, :]).edges
    truss["Beams"] = edges.T
    truss["Sizes"] = np.ones(len(truss["Beams"].T))*4 # Choose the 4th one in AREA_SEC
    return truss

    
if __name__ == "__main__":
    tr = init_truss(10)
    FOS, F, Load = fos_eval(tr)
    plot_truss(tr, FOS, F, Load)
    savefig("truss.png")