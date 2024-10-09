import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist, pdist2, squareform
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import time

# Load the necessary data files
map_data = sio.loadmat('mapdim20_updatethresh.mat')
beh_dist_data = sio.loadmat('beh_dist_6.mat')

# Extract variables from the loaded data
map = map_data['map']
ind_ic_neq_fdt = map['goodsim88'][0][0]
reps = 500
LPtype = 'hiker'
ics = len(ind_ic_neq_fdt)
simname = ['0', '1', '2', '3', '4', '500t']
isim = 6
probs = map_data['probs']  # Load the 'probs' array

# Initialize arrays
bestbeh = [None] * ics
allbp = np.zeros((ics, 6))
allE = np.zeros(ics)
E = np.zeros((len(probs), ics))
dippf = np.zeros(ics)

# Iterate over each IC
for iic in range(ics):
    ic = ind_ic_neq_fdt[iic][0]
    pointdata = sio.loadmat(f'../sims{simname[isim]}/sim_{LPtype}_ic{ic}_t100.mat')['pointdata']
    final_point = map['findxy'][ic,:][0]
    ipp = map['icsxy'][ic,:][0]
    
    # Calculate distance from find to ipp for weighting
    dippf[iic] = np.linalg.norm(final_point - ipp)
    
    # Energy distance calculation
    A = np.zeros(len(probs))
    B = np.zeros(len(probs))
    nreps = np.zeros(len(probs))
    
    for iprob in range(len(probs)):
        X = np.array([pointdata[iprob, irep][2, 0:2] for irep in range(reps)])
        Y = np.array([final_point for _ in range(reps)])
        
        # Remove NaNs
        inan = np.isnan(X)
        X = X[~inan[:, 0]]
        Y = Y[~inan[:, 0]]
        nreps[iprob] = len(X)
        
        Ap = pdist2(X, Y)  # Pairwise distances
        Aps = np.sum(Ap[:, 0])
        A[iprob] = (1/nreps[iprob]) * Aps
        
        Bp = 2 * np.sum(pdist(X))  # Sum of pairwise distances within X
        B[iprob] = (1 / nreps[iprob]**2) * Bp
        E[iprob, iic] = 2 * A[iprob] - B[iprob]
    
    # Sort by lowest value of E statistic
    E[nreps < 98, iic] = np.nan
    Ev, Iprob = np.sort(E[:, iic]), np.argsort(E[:, iic])
    
    bestbeh[iic] = (Iprob, Ev)
    allbp[iic, :] = probs[Iprob[0], :]
    allE[iic] = Ev[0]

# Save the result
sio.savemat(f'best fit data/allbesthiker_edist{simname[isim]}.mat', {'bestbeh': bestbeh, 'allbp': allbp, 'allE': allE, 'E': E, 'dippf': dippf})

# Compute best behavioral profiles based on energy distance
LE = 0.5
allwt_EL = allE ** (-LE)
sumbeh_EL = np.sum(allbp * allwt_EL[:, None], axis=0)
best_EL = sumbeh_EL / np.linalg.norm(sumbeh_EL, 1)

# Weighted by 1/(E/d)^L
ipps = map['icsxy'][ind_ic_neq_fdt, :]
finds = map['findxy'][ind_ic_neq_fdt, :]
ndippf = np.sqrt(np.sum((ipps - finds)**2, axis=1))
allwt_EdL = (allE / ndippf) ** (-LE)
sumbeh_EdL = np.sum(allbp * allwt_EdL[:, None], axis=0)
best_EdL = sumbeh_EdL / np.linalg.norm(sumbeh_EdL, 1)

# Save results
sio.savemat(f'best fit data/allweightedbesthiker_edist{simname[isim]}.mat', 
            {'ndippf': ndippf, 'bestbeh': bestbeh, 'allbp': allbp, 'allE': allE, 'E': E, 'LE': LE, 'best_EL': best_EL, 'best_EdL': best_EdL, 'allwt_EL': allwt_EL, 'allwt_EdL': allwt_EdL})

# Find closest reps using dsearchn equivalent in Python (scikit-learn)
clpts = [None] * ics
for iic in range(ics):
    ic = ind_ic_neq_fdt[iic][0]
    pointdata = sio.loadmat(f'../sims{simname[isim]}/sim_{LPtype}_ic{ic}_t100.mat')['pointdata']
    findxy = map['findxy'][ic, :][0]
    
    clpts[iic] = []
    for iprob in range(462):
        reppts = np.array([pointdata[iprob, irep][2, 0:2] for irep in range(reps)])
        nbrs = NearestNeighbors(n_neighbors=1).fit(reppts)
        distances, indices = nbrs.kneighbors([findxy])
        clpts[iic].append(reppts[indices[0][0]])

# Save closest points
sio.savemat(f'best fit data/allclosestpoints{simname[isim]}.mat', {'clpts': clpts})

# Plotting the results using matplotlib
for iic in range(ics):
    plt.figure()
    ic = ind_ic_neq_fdt[iic][0]
    bestp = bestbeh[iic][0][0]
    pointdata = sio.loadmat(f'../sims{simname[isim]}/sim_{LPtype}_ic{ic}_t100.mat')['pointdata']
    otherp = [2, 5, 10]
    
    for iiprob in range(3):
        iprob = bestbeh[iic][0][otherp[iiprob]]
        for ii in range(reps):
            cpx, cpy = pointdata[iprob, ii][2, 0], pointdata[iprob, ii][2, 1]
            plt.scatter(cpx, cpy, 50, marker='o', alpha=0.4)
        
        plt.plot(clpts[iic][iprob][0], clpts[iic][iprob][1], 'p', markersize=12, markerfacecolor='k', label=f'Closest Rep {otherp[iiprob]}')
    
    # Final plotting adjustments
    plt.legend(loc='upper right')
    plt.show()
    plt.close()
