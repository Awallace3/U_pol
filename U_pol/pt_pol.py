import time
from datetime import datetime
import logging
import torch
from torchmin import minimize  # use torchmin for minimization
import argparse
import os
import sys
import numpy as np

from openmm.app import *
from openmm import *
from simtk.unit import *
import os, sys
sys.path.append('.')

# Global constant to be set in set_constants()
ONE_4PI_EPS0 = None

def set_constants():
    global ONE_4PI_EPS0
    M_PI = 3.14159265358979323846
    E_CHARGE = 1.602176634e-19
    AVOGADRO = 6.02214076e23
    EPSILON0 = (1e-6 * 8.8541878128e-12 / (E_CHARGE * E_CHARGE * AVOGADRO))
    ONE_4PI_EPS0 = 1 / (4 * M_PI * EPSILON0)

def get_displacements(r_core, r_shell):
    """
    Given initial positions of a crystal structure or trajectory file, 
    initialize shell charge site positions and charges.
    """
    # Compute norm along last dimension and create a boolean mask for nonzero shells
    shell_mask = (torch.norm(r_shell, dim=-1) > 0.0)
    d = r_core - r_shell
    # Expand mask to last dimension
    d = torch.where(shell_mask.unsqueeze(-1), d, torch.tensor(0.0, dtype=d.dtype, device=d.device))
    return d

def get_inputs(scf='openmm', **kwargs):
    """
    Generate inputs based on an OpenMM realization (i.e., pdb, ff.xml, and residue.xml as inputs).

    Arguments:
        scf (str): Method for optimizing Drude positions.
        kwargs: Keyword arguments including:
            - dir: Directory path for pdb, xml, and residue.xml files.
            - mol: Name of benchmark molecule.
            - logger: Logger instance.

    Returns:
        Tuple of torch tensors and other parameters needed to compute U_ind.
    """
    set_constants()
    logger = kwargs['logger']
    path = kwargs['dir']
    pt_inputs = os.path.join(path, kwargs['mol'], kwargs['mol'] + ".pt")
    if os.path.exists(pt_inputs):
        pt_data = torch.load(pt_inputs, weights_only=False)
        # Convert numpy arrays to torch tensors
        Rij         = torch.tensor(pt_data["Rij"], dtype=torch.float64)
        Dij         = torch.tensor(pt_data["Dij"], dtype=torch.float64)
        Qi_shell    = torch.tensor(pt_data["Qi_shell"], dtype=torch.float64)
        Qj_shell    = torch.tensor(pt_data["Qj_shell"], dtype=torch.float64)
        Qi_core     = torch.tensor(pt_data["Qi_core"], dtype=torch.float64)
        Qj_core     = torch.tensor(pt_data["Qj_core"], dtype=torch.float64)
        u_scale     = torch.tensor(pt_data["u_scale"], dtype=torch.float64)
        k           = torch.tensor(pt_data["k"], dtype=torch.float64)
        Uind_openmm = pt_data["Uind_openmm"]
        return Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, Uind_openmm
    else:
        if openmm:
            pdb_file     = os.path.join(path, kwargs['mol'], kwargs['mol'] + ".pdb")
            xml_file     = os.path.join(path, kwargs['mol'], kwargs['mol'] + ".xml")
            residue_file = os.path.join(path, kwargs['mol'], kwargs['mol'] + "_residue.xml")

            # Use OpenMM to obtain bond definitions and atom/Drude positions.
            Topology().loadBondDefinitions(residue_file)
            integrator = DrudeSCFIntegrator(0.00001 * picoseconds)
            integrator.setRandomNumberSeed(123)
            pdb = PDBFile(pdb_file)
            modeller = Modeller(pdb.topology, pdb.positions)
            print(pdb.positions)
            forcefield = ForceField(xml_file)
            modeller.addExtraParticles(forcefield)
            system = forcefield.createSystem(modeller.topology, constraints=None, rigidWater=True)
            for i in range(system.getNumForces()):
                f = system.getForce(i)
                f.setForceGroup(i)
            # platform = Platform.getPlatformByName('CUDA')
            platform = None
            simmd = Simulation(modeller.topology, system, integrator, platform)
            simmd.context.setPositions(modeller.positions)

            drude = [f for f in system.getForces() if isinstance(f, DrudeForce)][0]
            nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
            
            positions = simmd.context.getState(getPositions=True).getPositions()
            
            # Optimize Drude positions using OpenMM
            # simmd.step(1)
            state = simmd.context.getState(getEnergy=True, getForces=True, getVelocities=True, getPositions=True)
            Uind_openmm = state.getPotentialEnergy()
            logger.info("=-=-=-=-=-=-=-=-=-=-=-=-OpenMM Output-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            logger.info("total Energy = " + str(Uind_openmm))
            for j in range(system.getNumForces()):
                f = system.getForce(j)
                PE = str(type(f)) + str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy())
                logger.info(PE)
                
            if scf == "openmm":  # if using openmm for drude optimization, update positions
                positions = simmd.context.getState(getPositions=True).getPositions()
            
            numDrudes = drude.getNumParticles()

            drude_indices  = [drude.getParticleParameters(i)[0] for i in range(numDrudes)]
            parent_indices = [drude.getParticleParameters(i)[1] for i in range(numDrudes)]
            
            # Initialize lists for core and shell positions and charges.
            topology = modeller.getTopology()
            r_core = []
            r_shell = []
            q_core = []
            q_shell = []
            alphas = []
            tholes = []
            tholeMatrixMade = False  # flag for whether a Thole matrix was built
            tholeTrue = False
            for res in topology.residues():
                res_core_pos = []
                res_shell_pos = []
                res_charge = []
                res_shell_charge = []
                res_alpha = []
                for atom in res.atoms():
                    if atom.index in drude_indices:
                        continue  # these are handled as part of the parent atoms
                    charge, sigma, epsilon = nonbonded.getParticleParameters(atom.index)
                    charge = charge.value_in_unit(elementary_charge)
                    if atom.index in parent_indices:
                        drude_index = drude_indices[parent_indices.index(atom.index)]
                        drude_pos = list(positions[drude_index])
                        drude_pos = [p.value_in_unit(nanometer) for p in drude_pos]
                        drude_params = drude.getParticleParameters(parent_indices.index(atom.index))
                        drude_charge = drude_params[5].value_in_unit(elementary_charge)
                        alpha = drude_params[6]
                        numScreenedPairs = drude.getNumScreenedPairs()
                        if numScreenedPairs > 0:
                            tholeTrue = True
                            if not tholeMatrixMade:
                                natoms_per_res = int((topology.getNumAtoms() - len(drude_indices)) / topology.getNumResidues())
                                natoms = len(list(res.atoms()))
                                nmol = len(list(topology.residues()))
                                tholeMatrix = np.zeros((nmol, natoms_per_res, natoms_per_res))
                                for sp_i in range(numScreenedPairs):
                                    screened_params = drude.getScreenedPairParameters(sp_i)
                                    prt0_params = drude.getParticleParameters(screened_params[0])
                                    drude0 = prt0_params[0]
                                    core0  = prt0_params[1]
                                    alpha0 = prt0_params[6].value_in_unit(nanometer**3)
                                    imol = int(core0 / natoms)
                                    prt1_params = drude.getParticleParameters(screened_params[1])
                                    drude1 = prt1_params[0]
                                    core1  = prt1_params[1]
                                    alpha1 = prt1_params[6].value_in_unit(nanometer**3)
                                    thole = screened_params[2]
                                    if core0 >= natoms:
                                        core0 = core0 % natoms
                                    if core1 >= natoms:
                                        core1 = core1 % natoms
                                    val = thole / ((alpha0 * alpha1)**(1./6.))
                                    tholeMatrix[imol][core0][core1] = val
                                    tholeMatrix[imol][core1][core0] = val
                                tholeMatrixMade = True
                        elif numScreenedPairs == 0:
                            tholeTrue = False
                            tholeMatrixMade = False
                        res_shell_charge.append(drude_charge)
                    else:
                        res_shell_charge.append(0.0)
                        drude_pos = [0.0, 0.0, 0.0]
                        alpha = 0.0 * nanometer**3
                    pos = list(positions[atom.index])
                    pos = [p.value_in_unit(nanometer) for p in pos]
                    alpha = alpha.value_in_unit(nanometer**3)
                    
                    res_core_pos.append(pos)
                    res_shell_pos.append(drude_pos)
                    res_charge.append(charge)
                    res_alpha.append(alpha)
                r_core.append(res_core_pos)
                r_shell.append(res_shell_pos)
                q_core.append(res_charge)
                q_shell.append(res_shell_charge)
                alphas.append(res_alpha)
        # Convert lists to torch tensors (using double precision)
        r_core  = torch.tensor(r_core, dtype=torch.float64)
        r_shell = torch.tensor(r_shell, dtype=torch.float64)
        q_core  = torch.tensor(q_core, dtype=torch.float64)
        q_shell = torch.tensor(q_shell, dtype=torch.float64)
        alphas  = torch.tensor(alphas, dtype=torch.float64)
        
        # Replace zeros in alphas with infinity to avoid division by zero.
        _alphas = torch.where(alphas == 0.0, torch.tensor(float('inf'), dtype=torch.float64), alphas)
        k = torch.where(alphas == 0.0, torch.tensor(0.0, dtype=torch.float64), 
                        ONE_4PI_EPS0 * q_shell**2 / _alphas)
        
        # Broadcast r_core: (nmol, natoms, 3) --> Rij (nmol, nmol, natoms, natoms, 3)
        Rij = r_core.unsqueeze(0).unsqueeze(2) - r_core.unsqueeze(1).unsqueeze(3)
        
        if tholeMatrixMade:
            tholes = torch.tensor(tholeMatrix, dtype=torch.float64)
            I_eye = torch.eye(Rij.shape[0], dtype=tholes.dtype)
            # Expand dimensions to match tholes: (1, nmol, 1, 1) multiplies tholes elementwise.
            u_scale = tholes.unsqueeze(0) * I_eye.unsqueeze(-1).unsqueeze(-1)
        else:
            u_scale = torch.tensor(0.0, dtype=torch.float64)
        
        # Create Dij from r_core and r_shell
        Dij = get_displacements(r_core, r_shell)
        
        # Expand dimensions of charge arrays to create core/shell interaction matrices.
        Qi_shell = q_shell.unsqueeze(1).unsqueeze(3)
        Qj_shell = q_shell.unsqueeze(0).unsqueeze(2)
        Qi_core  = q_core.unsqueeze(1).unsqueeze(3)
        Qj_core  = q_core.unsqueeze(0).unsqueeze(2)
        
        # (Optionally, one might save these inputs for future use.)
        # np.savez(npz_inputs, Rij=Rij.numpy(), Dij=Dij.numpy(), Qi_shell=Qi_shell.numpy(), 
        #          Qj_shell=Qj_shell.numpy(), Qi_core=Qi_core.numpy(), Qj_core=Qj_core.numpy(),
        #          u_scale=u_scale.numpy(), k=k.numpy(), Uind_openmm=Uind_openmm.value_in_unit(kilojoules_per_mole))
        # np.savez(npz_inputs, Rij=Rij.numpy(), Dij=Dij.numpy(), Qi_shell=Qi_shell.numpy(), 
        #          Qj_shell=Qj_shell.numpy(), Qi_core=Qi_core.numpy(), Qj_core=Qj_core.numpy(),
        #          u_scale=u_scale.numpy(), k=k.numpy(), Uind_openmm=Uind_openmm.value_in_unit(kilojoules_per_mole))
        data = {"Rij": Rij.numpy(), "Dij": Dij.numpy(), "Qi_shell": Qi_shell.numpy(),
                "Qj_shell": Qj_shell.numpy(), "Qi_core": Qi_core.numpy(), "Qj_core": Qj_core.numpy(),
                "u_scale": u_scale.numpy(), "k": k.numpy(), "Uind_openmm": Uind_openmm.value_in_unit(kilojoules_per_mole)}
        torch.save(data, pt_inputs)  # Save as a PyTorch tensor using
        
        return Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, Uind_openmm.value_in_unit(kilojoules_per_mole)

def get_raw_inputs(simmd, system, nonbonded_force, drude_force):
    positions = simmd.context.getState(getPositions=True).getPositions()
    r = []
    q = []
    Drude = []
    
    for i in range(system.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        charge = charge.value_in_unit(elementary_charge)
        pos = list(positions[i])
        pos = [p.value_in_unit(nanometer) for p in pos]
        has_drude = False
        for j in range(drude_force.getNumParticles()):
            params = drude_force.getParticleParameters(j)
            parent_atom_index = params[0]
            polarizability = params[6]
            if parent_atom_index == i:
                has_drude = True
                Drude.append(True)
        if not has_drude:
            Drude.append(False)
        q.append(charge)
        r.append(pos)
    
    logger.debug("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    logger.debug("\nq")
    logger.debug(q)
    logger.debug("\nr")
    logger.debug(r)
    logger.debug("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")

def get_DrudeTypeMap(forcefield):
    """
    For a given OpenMM force field, get the 'DrudeTypeMap' to distinguish the name of Drude atom types.
    """
    from openmm.app.forcefield import DrudeGenerator
    drudeTypeMap = {}
    for force in forcefield._forces:
        if isinstance(force, DrudeGenerator):
            for typ in force.typeMap:
                drudeTypeMap[typ] = force.typeMap[typ][0]

def Upol(Dij, k):
    """
    Calculates polarization energy, 
        U_pol = 1/2 Σ k_i * ||d_i||^2.
    """
    d_mag = torch.norm(Dij, dim=2)
    return 0.5 * torch.sum(k * d_mag**2)

def Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale):
    """
    Computes the Coulomb (electrostatic) contribution to the induction energy.
    """
    # Expand Dij to allow broadcasting: 
    # Di: (nmol, 1, natoms, 1, 3) and Dj: (1, nmol, 1, natoms, 3)
    Di = Dij.unsqueeze(1).unsqueeze(3)  # Dij[:, None, :, None, :]
    Dj = Dij.unsqueeze(0).unsqueeze(2)    # Dij[None, :, None, :, :]
    
    Rij_norm       = torch.norm(Rij, dim=-1)
    Rij_Di_norm    = torch.norm(Rij + Di, dim=-1)
    Rij_Dj_norm    = torch.norm(Rij - Dj, dim=-1)
    Rij_Di_Dj_norm = torch.norm(Rij + Di - Dj, dim=-1)
    
    # Replace zero norms with infinity to avoid division by zero.
    _Rij_norm       = torch.where(Rij_norm == 0.0, torch.tensor(float('inf'), dtype=Rij_norm.dtype), Rij_norm)
    _Rij_Di_norm    = torch.where(Rij_Di_norm == 0.0, torch.tensor(float('inf'), dtype=Rij_Di_norm.dtype), Rij_Di_norm)
    _Rij_Dj_norm    = torch.where(Rij_Dj_norm == 0.0, torch.tensor(float('inf'), dtype=Rij_Dj_norm.dtype), Rij_Dj_norm)
    _Rij_Di_Dj_norm = torch.where(Rij_Di_Dj_norm == 0.0, torch.tensor(float('inf'), dtype=Rij_Di_Dj_norm.dtype), Rij_Di_Dj_norm)
    
    Sij       = 1. - (1. + 0.5 * Rij_norm * u_scale) * torch.exp(-u_scale * Rij_norm)
    Sij_Di    = 1. - (1. + 0.5 * Rij_Di_norm * u_scale) * torch.exp(-u_scale * Rij_Di_norm)
    Sij_Dj    = 1. - (1. + 0.5 * Rij_Dj_norm * u_scale) * torch.exp(-u_scale * Rij_Dj_norm)
    Sij_Di_Dj = 1. - (1. + 0.5 * Rij_Di_Dj_norm * u_scale) * torch.exp(-u_scale * Rij_Di_Dj_norm)
    
    U_coul = (Qi_core  * Qj_core  / _Rij_norm +
              Qi_shell * Qj_core  / _Rij_Di_norm +
              Qi_core  * Qj_shell / _Rij_Dj_norm +
              Qi_shell * Qj_shell / _Rij_Di_Dj_norm)
    
    U_coul_intra = (Sij       * (-Qi_shell) * (-Qj_shell) / _Rij_norm +
                    Sij_Di    * ( Qi_shell) * (-Qj_shell) / _Rij_Di_norm +
                    Sij_Dj    * (-Qi_shell) * ( Qj_shell) / _Rij_Dj_norm +
                    Sij_Di_Dj * ( Qi_shell) * ( Qj_shell) / _Rij_Di_Dj_norm)
    
    # Remove self-interactions by applying appropriate masks.
    I_intra = torch.eye(U_coul_intra.shape[0], dtype=U_coul_intra.dtype, device=U_coul_intra.device)\
                    .unsqueeze(-1).unsqueeze(-1)
    I_self  = torch.eye(U_coul_intra.shape[-1], dtype=U_coul_intra.dtype, device=U_coul_intra.device)\
                    .unsqueeze(0).unsqueeze(0)
    U_coul_intra = (U_coul_intra * I_intra) * (1 - I_self)
    U_coul_intra = 0.5 * torch.sum(torch.where(torch.isfinite(U_coul_intra), U_coul_intra, 
                                                torch.tensor(0.0, dtype=U_coul_intra.dtype, device=U_coul_intra.device)))
    
    I = torch.eye(U_coul.shape[0], dtype=U_coul.dtype, device=U_coul.device)\
            .unsqueeze(-1).unsqueeze(-1)
    U_coul_inter = U_coul * (1 - I)
    U_coul_inter = 0.5 * torch.sum(torch.where(torch.isfinite(U_coul_inter), U_coul_inter, 
                                                torch.tensor(0.0, dtype=U_coul.dtype, device=U_coul.device)))
    return ONE_4PI_EPS0 * (U_coul_inter + U_coul_intra)

def Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape=None):
    """
    Calculates the total induction energy:
        U_ind = U_pol + U_coul.
    """
    if reshape is not None:
        Dij = Dij.view(reshape)
    U_pol  = Upol(Dij, k)
    U_coul_val = Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
    logger.debug(f"U_pol = {U_pol} kJ/mol\nU_coul = {U_coul_val}\n")
    return U_pol + U_coul_val

def drudeOpt(Rij, Dij0, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, methods=["BFGS"], d_ref=None, reshape=None):
    """
    Optimizes the Drude (core/shell) displacements by minimizing the induction energy U_ind.
    """
    def Uind_min(Dij):
        return Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape)
    
    start = time.time()
    # Use torchmin for minimization (here, method is assumed to be 'BFGS')
    res = minimize(Uind_min, x0=Dij0, method="BFGS")
    end = time.time()
    logger.info(f"torchmin.BFGS Minimizer completed in {end - start:.3f} seconds!!")
    d_opt = res["x"]
    if reshape is not None:
        d_opt = d_opt.view(reshape)
    if d_ref is not None:
        diff = torch.norm(d_ref - d_opt)
        logger.info(f"Difference from reference: {diff.item()}")
    return d_opt

# Initialize a logger.
logger = logging.getLogger(__name__)

def main():
    # Configure logging (this will create a file named 'log.out').
    logging.basicConfig(filename='log.out', level=logging.INFO, format='%(message)s')
    logging.info(f"Log started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    set_constants()
    
    parser = argparse.ArgumentParser(description="Calculate U_ind = U_pol + U_es for a selected molecule.")
    parser.add_argument("--mol", type=str, required=True, choices=["water", "acnit", "imidazole", "pyrazole"],
                        help="Molecule type (with OpenMM files).")
    parser.add_argument("--dir", type=str, default="../benchmarks/OpenMM",
                        help="Directory for benchmark input files.")
    parser.add_argument("--scf", type=str, default=None,
                        help="SCF method, can be 'openmm' (for reference Dij) or None.")
    
    args = parser.parse_args()
    dir_path = args.dir
    mol = args.mol
    scf = args.scf
    
    logger.info(f"%%%%%%%%%%% STARTING {mol.upper()} U_IND CALCULATION %%%%%%%%%%%%")
    logger.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    
    # Get inputs (all arrays are now PyTorch tensors)
    Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, Uind_openmm = \
                get_inputs(scf=scf, dir=dir_path, mol=mol, logger=logger)
    # Flatten Dij to a 1D tensor for minimization and later reshape it back.
    Dij_flat = Dij.view(-1)
    Dij_opt = drudeOpt(Rij, Dij_flat, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape=Dij.shape)
    U_ind = Uind(Rij, Dij_opt, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k)
    logger.info(f"OpenMM U_ind = {Uind_openmm:.4f} kJ/mol")
    logger.info(f"PyTorch U_ind = {U_ind:.4f} kJ/mol")
    error_percent = abs((Uind_openmm - U_ind) / U_ind) * 100
    logger.info(f"{error_percent:.2f}% Error")
    logger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")

if __name__ == "__main__":
    main()

