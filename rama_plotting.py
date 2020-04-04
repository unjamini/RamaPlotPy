import Bio.PDB
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


molecules_d = {'6LML': 'data/6lml.pdb',
               '6VQ9': 'data/6vq9.pdb',
               '6HGA': 'data/6hga.pdb'}


def get_angles(filename):
    """

    :param filename: name of the file with data
    :return: 2 lists of phi and psi angles values
    """
    phi_angles = []
    psi_angles = []
    for model in Bio.PDB.PDBParser().get_structure(mol_id, filename):
        for chain in model:
            polypeptides = Bio.PDB.CaPPBuilder().build_peptides(chain)
            for polypeptide in polypeptides:
                angles = polypeptide.get_phi_psi_list()
                for residue_idx in range(len(polypeptide)):
                    phi, psi = angles[residue_idx]
                    if phi and psi:
                        phi_angles.append(math.degrees(phi))
                        psi_angles.append(math.degrees(psi))
    return phi_angles, psi_angles


def draw_scatter(phi_angles, psi_angles, m_id):
    """

    :param phi_angles: list of phi angles
    :param psi_angles: list of psi angles
    :param m_id: id (name) of structure
    :return:
    """
    x_g, y_g = np.mgrid[min(phi_angles):max(phi_angles):100j, min(psi_angles):max(psi_angles):100j]

    positions = np.vstack([x_g.ravel(), y_g.ravel()])
    kernel = st.gaussian_kde(np.vstack([phi_angles, psi_angles]))
    f = np.reshape(kernel(points=positions).T, x_g.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title(f'Ramachandran plot for {m_id}')
    ax1.scatter(phi_angles, psi_angles, s=11, c='purple')
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-180, 180)
    ax1.set_xlabel(r'$\phi$')
    ax1.set_ylabel(r'$\psi$')
    ax1.grid()
    ax1.locator_params(axis='x', nbins=7)
    ax1.locator_params(axis='y', nbins=7)

    ax2.set_title(f'Ramachandran plot for {m_id} with Gaussian KDE levels map')
    ax2.scatter(phi_angles, psi_angles, s=11, c='purple')
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-180, 180)
    ax2.contour(x_g, y_g, f)
    ax2.set_xlabel(r'$\phi$')
    ax2.set_ylabel(r'$\psi$')
    ax2.grid()
    ax2.locator_params(axis='x', nbins=7)
    ax2.locator_params(axis='y', nbins=7)
    plt.savefig(f'results/{m_id}.png', dpi=300)
    plt.show()


for mol_id, filename in molecules_d.items():
    phi, psi = get_angles(filename)
    draw_scatter(phi, psi, mol_id)