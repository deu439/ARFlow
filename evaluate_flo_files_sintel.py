import numpy as np
from path import Path
from utils.flow_utils import load_flow, sp_plot
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

list_path = Path("./lists/MPISintel_train.txt")
gt_path = Path("/home/deu/Datasets/MPI_Sintel/")
est_path = Path("/home/deu/Datasets/MPI_Sintel/uflow_elbo/")
epe = 0
sauc = 0
oauc = 0
splots = []
oplots = []
sp_samples = 25

# Load list
sample_list = []
with open(list_path, "r") as fh:
    while True:
        line = fh.readline().strip()
        if line:
            sample_list.append(line.split(' '))
        else:
            break

for sample in tqdm(sample_list):
    tflow_path = gt_path / sample[2]
    eflow_path = est_path / Path(sample[0]).with_suffix(".flo")
    ent_path = est_path / Path(sample[0]).with_suffix(".npy")

    assert tflow_path.is_file() and eflow_path.is_file() and ent_path.is_file()

    gt = load_flow(tflow_path)
    est = load_flow(eflow_path)
    var = np.load(ent_path)
    #entropy_map = np.sum(np.log(var), axis=-1) / 2.0
    entropy_map = np.sum(var, axis=-1)

    # Calculate endpoint error
    epe_map = np.sqrt(np.sum(np.square(est - gt), axis=2))
    mask = np.ones_like(epe_map)
    epe += np.mean(epe_map)

    # Calculate sparsification plots and AUC
    splot = sp_plot(epe_map, entropy_map, mask, n=sp_samples)
    oplot = sp_plot(epe_map, epe_map, mask, n=sp_samples)  # Oracle
    splots += [splot]
    oplots += [oplot]

    # Cummulate AUC
    frac = np.linspace(0, 1, sp_samples)
    sauc += np.trapz(splot / splot[0], x=frac)
    oauc += np.trapz(oplot / oplot[0], x=frac)

# Print metrics
print(f"EPE: {epe / len(sample_list)}")
print(f"AUC: {sauc / len(sample_list)}")
print(f"AUC diff: {(sauc - oauc) / len(sample_list)}")


# Display sparsification plots
def display_sp_plots(ax, splots, sp_samples=sp_samples):
    splots_mean = np.mean(splots, axis=0)
    #splots_std = np.std(splots, axis=0)

    # the 1 sigma upper and lower analytic population bounds
    frac = np.linspace(0, 1, sp_samples)
    ax.plot(frac, splots_mean)
    #ax.errorbar(frac, splots_mean, splots_std)
    ax.set_xlabel('fraction removed [-]')
    ax.set_ylabel('average endpoint error [px]')


fig, ax = plt.subplots(1, 2)
display_sp_plots(ax[0], splots, sp_samples=sp_samples)
display_sp_plots(ax[0], oplots, sp_samples=sp_samples)
ax[0].legend(['splot', 'oracle'])
display_sp_plots(ax[1], np.array(splots) - np.array(oplots), sp_samples=sp_samples)
ax[1].legend(['diff'])
plt.show()
