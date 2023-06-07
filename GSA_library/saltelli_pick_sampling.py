from __future__ import division

import os

import copy

import numpy as np

from SALib.sample import common_args
from SALib.sample import sobol_sequence
from SALib.util import scale_samples, nonuniform_scale_samples, read_param_file, compute_groups_matrix

from GSA_library.plotting import plot_dataset

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.design import lhd

from Historia.history import hm

def sample_NIMP(problem, N, 
                calc_second_order=True, 
                seed=None, 
                skip_values=1000, 
                sampling_method='sobol',
                wave_file=None,
                wave_gpepath=None,
                wave_features_idx=None,
                wave_idx_param=None):

    """Generates model inputs using Saltelli's extension of the Sobol sequence.

    Returns a NumPy matrix containing the model inputs using Saltelli's sampling
    scheme.  Saltelli's scheme extends the Sobol sequence in a way to reduce
    the error rates in the resulting sensitivity index calculations.  If
    calc_second_order is False, the resulting matrix has N * (D + 2)
    rows, where D is the number of parameters.  If calc_second_order is True,
    the resulting matrix has N * (2D + 2) rows.  These model inputs are
    intended to be used with :func:`SALib.analyze.sobol.analyze`.

    Parameters
    ----------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate
    calc_second_order : bool
        Calculate second-order sensitivities (default True)

    WARNING: This is a modification of the original Saltelli sampling where the 
    sampling for a subset of parameters is restricted to a non-implausible area
    given by a history matching wave ran previously. The GPEs trained during this wave
    are used to exclude non-implausible samples.

    """
    if seed:
        np.random.seed(seed)

    D = problem['num_vars']
    groups = problem.get('groups')

    if not groups:
        Dg = problem['num_vars']
    else:
        Dg = len(set(groups))
        G, group_names = compute_groups_matrix(groups)

    # Create base sequence - could be any type of sampling
    if sampling_method=='sobol':

        base_sequence = sobol_sequence.sample(N + skip_values, 2 * D)

    elif sampling_method=='lhd':
        
        I = np.zeros((D*2,2),dtype=float)
        I[:,1] = 1.
        base_sequence = lhd(I,(N + skip_values))

    elif sampling_method=='lhd_NIMP':

        if (wave_file is None):
            raise Exception('You need to provide a wave file.')
        if (wave_gpepath is None):
            raise Exception('Please provide the path to the GPE you want to use to exclude points outside the NIMP.')
        if (wave_features_idx is None):
            raise Exception('Please provide the index of the output features you considered for the wave.')
        if  (wave_idx_param is None):
            raise Exception('Please provide the index of the parameters that are part of the NIMP.')

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('                   Setting skip_values to 0                ')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        skip_values = 0

        I = np.zeros((D,2),dtype=float)
        I[:,1] = 1.
        lhd_samples = lhd(I,(N + skip_values)*2)
        lhd_samples_rescaled = copy.deepcopy(lhd_samples)
        for i in range(D):
            lhd_samples_rescaled[:,i] = lhd_samples[:,i]*(problem['bounds'][i,1]-problem['bounds'][i,0])+problem['bounds'][i,0]

        W = hm.Wave()
        W.load(wave_file)   

        print('-----------------------------')
        print('Loading emulators for wave...')
        print('-----------------------------')  

        emulator_w = []
        for idx_w in wave_features_idx:
            loadpath_wave = wave_gpepath + str(idx_w) + "/"
            X_train_w = np.loadtxt(loadpath_wave + "X_train.txt", dtype=np.float64)
            y_train_w = np.loadtxt(loadpath_wave + "y_train.txt", dtype=np.float64)
            emul_w = GPEmul.load(X_train_w, y_train_w, loadpath=loadpath_wave)
            emulator_w.append(emul_w)
        
        W.emulator = emulator_w 

        W.find_regions(lhd_samples_rescaled[:,wave_idx_param])
        lhd_samples_NIMP = lhd_samples[W.nimp_idx,:]
        closest_N = 2*round(lhd_samples_NIMP.shape[0]/2)

        if closest_N>lhd_samples_NIMP.shape[0]:
            N = closest_N-2
        else:
            N = closest_N

        print('-------------------------------------------------------------------------')
        print('DONE: detected '+str(lhd_samples_NIMP.shape[0])+'/'+str(lhd_samples.shape[0])+' viable samples.')
        print('-------------------------------------------------------------------------')
        print('\n')
        print('-------------------------------------------------------------------------')
        print('                           Changing N to '+str(N)+'                      ')
        print('-------------------------------------------------------------------------')
        print('\n')

        base_sequence = np.zeros((N + skip_values, 2 * D),dtype=float)
        base_sequence[:,:D] = lhd_samples_NIMP[:N,:]
        base_sequence[:,D:] = lhd_samples_NIMP[N:,:]


    elif sampling_method=='sobol_NIMP':

        initial_N = N

        if (wave_file is None):
            raise Exception('You need to provide a wave file.')
        if (wave_gpepath is None):
            raise Exception('Please provide the path to the GPE you want to use to exclude points outside the NIMP.')
        if (wave_features_idx is None):
            raise Exception('Please provide the index of the output features you considered for the wave.')
        if  (wave_idx_param is None):
            raise Exception('Please provide the index of the parameters that are part of the NIMP.')

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('                   Setting skip_values to 0                ')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        skip_values = 0

        n_viable = 0

        while n_viable<initial_N:
            base_sequence_initial = sobol_sequence.sample(N + skip_values, 2 * D)
            A = base_sequence_initial[:,:D]
            B = base_sequence_initial[:,D:]
            A_rescaled = copy.deepcopy(A)
            B_rescaled = copy.deepcopy(B)   

            for i in range(D):
                A_rescaled[:,i] = A_rescaled[:,i]*(problem['bounds'][i,1]-problem['bounds'][i,0])+problem['bounds'][i,0]
                B_rescaled[:,i] = B_rescaled[:,i]*(problem['bounds'][i,1]-problem['bounds'][i,0])+problem['bounds'][i,0]    

            W = hm.Wave()
            W.load(wave_file)       

            print('-----------------------------')
            print('Loading emulators for wave...')
            print('-----------------------------')      

            emulator_w = []
            for idx_w in wave_features_idx:
                loadpath_wave = wave_gpepath + str(idx_w) + "/"
                X_train_w = np.loadtxt(loadpath_wave + "X_train.txt", dtype=np.float64)
                y_train_w = np.loadtxt(loadpath_wave + "y_train.txt", dtype=np.float64)
                emul_w = GPEmul.load(X_train_w, y_train_w, loadpath=loadpath_wave)
                emulator_w.append(emul_w)
            
            W.emulator = emulator_w     

            W.find_regions(A_rescaled[:,wave_idx_param])
            A_nimp_idx = W.nimp_idx 

            W.find_regions(B_rescaled[:,wave_idx_param])
            B_nimp_idx = W.nimp_idx 

            nimp_idx = np.intersect1d(A_nimp_idx,B_nimp_idx)
            n_viable = nimp_idx.shape[0]

            if n_viable<initial_N:
                print('-------------------------------------------------------------------------')
                print('DONE: detected '+str(nimp_idx.shape[0])+'/'+str(initial_N)+' viable samples - INCREASING N to '+str(N*2))
                print('-------------------------------------------------------------------------')
                print('\n')

                N = N*2
            else:
                print('-------------------------------------------------------------------------')
                print('DONE: detected '+str(nimp_idx.shape[0])+'/'+str(initial_N)+' viable samples - STOPPING')
                print('-------------------------------------------------------------------------')
                print('\n')

                base_sequence = np.zeros((n_viable,2*D),dtype=float)
                base_sequence[:,:D] = A[nimp_idx,:]
                base_sequence[:,D:] = B[nimp_idx,:]

                N = nimp_idx.shape[0]

                print('-------------------------------------------------------------------------')
                print('                           Changing N to '+str(nimp_idx.shape[0])+'                      ')
                print('-------------------------------------------------------------------------')
                print('\n')

                xlabels = [str(i) for i in range(D)]
                if not os.path.exists("./A.png"):
                    print('Plotting A samples...')
                    plot_dataset(base_sequence[:,:D],base_sequence[:,:D],xlabels,xlabels,"./")
                    os.system("mv ./X_vs_Y.png ./A.png")

                if not os.path.exists("./B.png"):
                    print('Plotting B samples...')
                    plot_dataset(base_sequence[:,D:],base_sequence[:,D:],xlabels,xlabels,"./")
                    os.system("mv ./X_vs_Y.png ./B.png")

    else:
        raise Exception('I do not recognise the sampling method you want.')

    if calc_second_order:
        saltelli_sequence = np.zeros([(2 * Dg + 2) * N, D])
    else:
        saltelli_sequence = np.zeros([(Dg + 2) * N, D])
    index = 0

    for i in range(skip_values, N + skip_values):

        # Copy matrix "A"
        for j in range(D):
            saltelli_sequence[index, j] = base_sequence[i, j]

        index += 1

        # Cross-sample elements of "B" into "A"
        for k in range(Dg):
            for j in range(D):
                if (not groups and j == k) or (groups and group_names[k] == groups[j]):
                    saltelli_sequence[index, j] = base_sequence[i, j + D]
                else:
                    saltelli_sequence[index, j] = base_sequence[i, j]

            index += 1

        # Cross-sample elements of "A" into "B"
        # Only needed if you're doing second-order indices (true by default)
        if calc_second_order:
            for k in range(Dg):
                for j in range(D):
                    if (not groups and j == k) or (groups and group_names[k] == groups[j]):
                        saltelli_sequence[index, j] = base_sequence[i, j]
                    else:
                        saltelli_sequence[index, j] = base_sequence[i, j + D]

                index += 1

        # Copy matrix "B"
        for j in range(D):
            saltelli_sequence[index, j] = base_sequence[i, j + D]

        index += 1

    scale_samples(saltelli_sequence, problem['bounds'])

    if not os.path.exists("./saltelli.png"):
        xlabels = [str(i) for i in range(D)]

        print('Plotting saltelli samples...')
        plot_dataset(saltelli_sequence,saltelli_sequence,xlabels,xlabels,"./")
        os.system("mv ./X_vs_Y.png ./saltelli.png")

    weights = np.ones((saltelli_sequence.shape[0],),dtype=int)    
    if 'NIMP' in sampling_method:

        print('-------------------------------------------------------------------------')
        print('Scanning A & B crossed samples to make sure they are in the NIMP...')
        print('-------------------------------------------------------------------------')
        print('\n')

        W.find_regions(saltelli_sequence[:,wave_idx_param])
        saltelli_nimp_idx = W.nimp_idx 

        weights[W.imp_idx] = 0

        print('-------------------------------------------------------------------------')
        print('DONE: detected '+str(saltelli_nimp_idx.shape[0])+'/'+str(saltelli_sequence.shape[0])+' viable samples')
        print('-------------------------------------------------------------------------')
        print('\n')

        if not os.path.exists("./saltelli_NIMP.png"):
            xlabels = [str(i) for i in range(D)]
            print('Plotting viable Saltelli samples...')
            plot_dataset(saltelli_sequence[saltelli_nimp_idx,:],saltelli_sequence[saltelli_nimp_idx,:],xlabels,xlabels,"./")
            os.system("mv ./X_vs_Y.png ./saltelli_NIMP.png")


    return saltelli_sequence,weights

def cli_parse_NIMP(parser):
    """Add method specific options to CLI parser.

    Parameters
    ----------
    parser : argparse object

    Returns
    ----------
    Updated argparse object
    """
    parser.add_argument('--max-order', type=int, required=False, default=2,
                        choices=[1, 2],
                        help='Maximum order of sensitivity indices \
                           to calculate')
    return parser


def cli_action_NIMP(args):
    """Run sampling method

    Parameters
    ----------
    args : argparse namespace
    """
    problem = read_param_file(args.paramfile)
    param_values = sample(problem, args.samples,
                          calc_second_order=(args.max_order == 2),
                          seed=args.seed)
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')


if __name__ == "__main__":
    common_args.run_cli(cli_parse_NIMP, cli_action_NIMP)
