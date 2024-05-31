# JAXIONS MODULE TO TRANSFER JAXIONS CONFIGURATIONS TO AMR CODES (CHOMBO AND SLEDGEHAMR)
import numpy as np
import os, sys, h5py


def transferConfig(
    config='out/m/axion.00000',
    where='.',
    name='config',
    shape_init=(256, 256, 256),
    chombo = False,
    nghosts =3,
    shape_chunk=(64, 64, 64)
):
    """
    transferConfig(config='out/m/axion.00000', where='.', name='config', shape_init=(256, 256, 256), chombo = False, nghosts=3, shape_chunk=(70, 70, 70))

    Reads a jaxions configuration (HDF5 file), reshapes the datasets 'm' and 'v' from 1D arrays to 3D arrays of desired shape,
    then splits them into two new datasets each. Saves these datasets into a separate HDF5 file for sledgehamr and into text files
    for Chombo. Since Chombo needs the ghost cells explicitly, the way to go is to generate the ICs in a bigger grid with shape_init and cut out the respective chunk
    of size of shape_chunk from the middle.
    For Chombo we also save x, y, z values into separate text files in order to map the correct coordinates to the grid.

    config          specifies the path to the jaxions configuration file
    where           where to save the new files
    name            how to name the new files
    shape_init      3D shape of the initial configuration
    chombo          if we need to output the txt files for chombo
    nghosts         number of ghost cells needed by chombo (need to be stored in the chunk)
    shape_chunk     3D shape of the chunk

    """
    # Open the original jaxions config HDF5 file
    with h5py.File(config, "r") as f:
        # Read the datasets 'm' and 'v' (fields + conjugated momenta)
        data_m = f["m"][...]
        data_v = f["v"][...]

        # Reshape the datasets from flattened array to shape of the simulation box (factor of two for real + imag part)
        reshaped_m = np.reshape(data_m, (shape_init[0], shape_init[1], shape_init[2], 2))
        reshaped_v = np.reshape(data_v, (shape_init[0], shape_init[1], shape_init[2], 2))

        # Split the datasets into two parts each
        m1 = reshaped_m[:, :, :, 0]
        m2 = reshaped_m[:, :, :, 1]

        v1 = reshaped_v[:, :, :, 0]
        v2 = reshaped_v[:, :, :, 1]


        # Save the new fields to a new HDF5 file
        with h5py.File(f"{where}/{name}_full.hdf5", "w") as new_f:
            new_f.create_dataset("Psi1", data=m1)
            new_f.create_dataset("Psi2", data=m2)
            new_f.create_dataset("Pi1", data=v1)
            new_f.create_dataset("Pi2", data=v2)

        if chombo:
            #cut out the outer part to get right shape
            start_x = (shape_init[0] - (shape_chunk[0] + 2*nghosts))// 2
            end_x = start_x + shape_chunk[0] + 2*nghosts

            start_y = (shape_init[1] - (shape_chunk[1] + 2*nghosts))// 2
            end_y = start_x + shape_chunk[1] + 2*nghosts

            start_z = (shape_init[2] - (shape_chunk[2]+ 2*nghosts))// 2
            end_z = start_x + shape_chunk[2] + 2*nghosts

            m1_chunk = m1[start_x:end_x, start_y:end_y, start_z:end_z]
            m2_chunk = m2[start_x:end_x, start_y:end_y, start_z:end_z]
            v1_chunk = v1[start_x:end_x, start_y:end_y, start_z:end_z]
            v2_chunk = v2[start_x:end_x, start_y:end_y, start_z:end_z]


            # Save the chunk of the full config to a new HDF5 file
            with h5py.File(f"{where}/{name}_chunk.hdf5", "w") as new_f:
                new_f.create_dataset("Psi1", data=m1_chunk)
                new_f.create_dataset("Psi2", data=m2_chunk)
                new_f.create_dataset("Pi1", data=v1_chunk)
                new_f.create_dataset("Pi2", data=v2_chunk)


            # Save reshaped datasets to text files for Chombo read-in (note phi=psi in chombo conventions)
            np.savetxt(f"{where}/{name}_phi1.txt", m1_chunk.flatten(), delimiter=" ", newline=" ")
            np.savetxt(f"{where}/{name}_phi2.txt", m2_chunk.flatten(), delimiter=" ", newline=" ")
            np.savetxt(f"{where}/{name}_pi1.txt", v1_chunk.flatten(), delimiter=" ", newline=" ")
            np.savetxt(f"{where}/{name}_pi2.txt", v2_chunk.flatten(), delimiter=" ", newline=" ")

            # Create x, y, z linspace arrays
            xl, yl, zl = [], [], []
            for x in range(shape_chunk[0] + 2*nghosts):
                for y in range(shape_chunk[1] + 2*nghosts):
                    for z in range(shape_chunk[2] + 2*nghosts):
                        xl.append(x-nghosts)
                        yl.append(y-nghosts)
                        zl.append(z-nghosts)

            # Save x, y, z to text files
            np.savetxt(f"{where}/{name}_x.txt", xl, delimiter=" ", newline=" ", fmt = "%d")
            np.savetxt(f"{where}/{name}_y.txt", yl, delimiter=" ", newline=" ", fmt = "%d")
            np.savetxt(f"{where}/{name}_z.txt", zl, delimiter=" ", newline=" ", fmt = "%d")

    return None

## Formerly part of nyx_tools.py, maybe outdated but we'll keep it here for now

#General tool to access the simulation parameters directly with python
def readInputs(PATH):
    """
    readInputs(PATH)
    1 - accesses the inputs file in PATH
    2 - reads out all the parameters that have been used and stores them in a dict
    3 - creates a second dict with the aliases to avoid unnecessary prefixes that are relics of Nyx and AMReX syntax (for example you access a variable via "nyx.msa" but also with "msa", cf. help(getValue()))

    PATH is a string specifying the location of the "spectra" folder which is created automatically for every nyx simulation
    """

    parameters_dict = {}
    alias_dict = {}

    # Open the inputs file in PATH and read the parameters
    with open(PATH+"inputs", "r") as file:
        for line in file:
            line = line.strip()
            #Skip commands and empty lines
            if line.startswith("#") or not line:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                #Cut off commands at the end of a line
                value = value.split("#")[0].strip()
                values = value.split()
                #Store values as a list if there are multiple values (for example the three coordinates ot the specifications for the levels)
                if len(values) > 1:
                    # Try converting each value to float, int, or leave it as string
                    converted_values = []
                    for val in values:
                        try:
                            converted_val = int(val)
                        except ValueError:
                            try:
                                converted_val = float(val)
                            except ValueError:
                                converted_val = val
                        converted_values.append(converted_val)
                    full_key = key.strip()
                    parameters_dict[full_key] = converted_values

                     # Check if the key has a prefix, e.g., "amr." or "nyx."
                    prefix, _, short_key = full_key.partition(".")
                    if prefix:
                        alias_dict[short_key] = full_key
                else:
                    full_key = key.strip()
                    parameters_dict[full_key] = values[0]

                    # Check if the key has a prefix, e.g., "amr." or "nyx."
                    prefix, _, short_key = full_key.partition(".")
                    if prefix:
                        alias_dict[short_key] = full_key
    return parameters_dict, alias_dict


#Function to retrieve the value of a simulation parameter using the full key or alias
def getValue(PARAM, parameters_dict, alias_dict):
    """
    getValue(PARAM)
    1 - Accessses the parameter dictionaries created with readInputs() (cf. help(readInputs()))
    2 - returns the respective value(s) if possible

    PARAM is a string of a key in parameters_dict or alias_dict
    """
    if PARAM in parameters_dict:
        return parameters_dict[PARAM]
    elif PARAM in alias_dict:
        return parameters_dict[alias_dict[PARAM]]
    else:
        print('%s not used in inputs. Assume default value or check for typos.'%PARAM)
        return None

#Overview of effective simulation paramters for AMR simulations
def overview(PATH):
    """
    overview(PATH)
    1 - accesses the inputs file in PATH
    2 - takes the relevant parameters and computes the effective  paramters for the finest-level AMR grid
    3 - prints an overview

    PATH is a string specifying the location of the "spectra" folder which is created automatically for every nyx simulation
    """
    parameters_dict, alias_dict = readInputs(PATH)

    #Get relevant parameters from inputs file
    try:
        ell = int(getValue("max_level",parameters_dict, alias_dict))
    except (KeyError, TypeError):
        print('max_level not used in inputs. Assume default value $\ell = 0$.')
        ell = int(0)

    try:
        ref_ratio = int(getValue("ref_ratio",parameters_dict, alias_dict))
    except (KeyError, TypeError):
        print('ref_ratio not used in inputs. Assume default value of 2.')
        ref_ratio = int(2)

    try:
        msa = float(getValue("msa",parameters_dict, alias_dict))
    except (KeyError, TypeError):
        print('msa not used in inputs. Assume default value of $ms_a=1.0$.')
        msa = float(1.0)

    try:
        N = int(getValue("n_cell",parameters_dict, alias_dict)[0])
    except (KeyError, TypeError):
        print('n_cell not used in inputs. Assume default value of $N=128$.')
        N = int(128)

    #Compute effective parameters
    N_eff = N * ref_ratio**ell
    msa_eff = float(msa/(ref_ratio**ell))

    print(r'This AMR simulation effectively resolves with an effective N = {:d} and msa = {:.2f} on the finest level (l = {:d}).'.format(N_eff, msa_eff, ell))



#Access the spectrum data for a Nyx simulation (usually stored in ./spectra of the simulation folder)
def getSpecfiles(PATH = './spectra/'):
    """
    getspecfiles(PATH)
    1 - accesses the spectrum data files in PATH
    2 - renames and sorts them w.r.t to the simulation time
    3 - returns the list of files as a list
    !mpirun $USA -np $RANK -x OMP_NUM_THREADS=$THR vaxion3d $JAX > log.txt

    PATH is a string specifying the location of the "spectra" folder which is created automatically for every nyx simulation
    """
    specfiles = []

    _, _, filenames = next(os.walk(PATH))

    times = []
    for name in filenames:
        times.append(float(name.split("_")[1])) #reduce names to their codetimes and sort them accordingly
    times = np.array(times)
    timeinds = times.argsort()
    filenames = np.array(filenames)[timeinds]
    specfiles.append(filenames)

    return specfiles


#Automate processing of the spectrum data files
class processSpectrum:
    """
    processSpectrum: __init__(self, PATH, FILES, MSA, L, N)
    1 - Reads the spectrum FILES in PATH and associates the data with the relevant parameters
    2 - Computes some other potentially interesting variables for comparison with Jaxions spectra (e.g log(h) etc.)

    PATH is a string specifying the location of the "spectra" folder which is created automatically for every nyx simulation
    FILES is list of data files such as for example generareted with the getspecfiles function defined above in nyx_tools.py (see help(getspecfiles()))
    MSA is the value of msa on the root grid
    L is the simulation volume
    N (or N0) is the number of grid points in one direction in the root grid.
    """
    def __init__(self, PATH, FILES, MSA=2, L=256, N=128):

        #Read and prepare data
        spec=np.transpose(np.loadtxt(PATH+FILES[0]))

        #Assign data to new variables
        self.nm = spec[2]
        self.k = spec[0]/self.nm

        #Cutoff at the saxion mass
        self.k_below = self.k <= MSA* N/L

        self.t = []
        self.log = []
        self.esp = []
        self.P = []
        for file in FILES:
            self.t.append(float(file.split("_")[1]))
            self.log.append(np.log(MSA*self.t[-1]))
            self.esp.append(np.transpose(np.loadtxt(PATH+file))[1])
            self.P.append((self.k**3)*self.esp[-1]/((np.pi**2)*self.nm))

        self.t = np.array(self.t)
        self.log = np.array(self.log)
        self.esp = np.array(self.esp)
        self.P = np.array(self.P)
