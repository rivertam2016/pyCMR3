import mkl
mkl.set_num_threads(1)
import numpy as np
import os
import errno
from glob import glob
import time
import CMR3 as CMR2

"""
This code uses the particle swarm optimization (pso) python package, pyswarm. 
The pyswarm package and its documentation are available here:

https://pythonhosted.org/pyswarm/

The original PSO code is adapted here to enable it to distribute over cores using lock files.

Dependencies: A CMR2/CMR3 version (that has already been built) and any package imports above.
              Must also have access to data files of presented and recalled items,
              as well as an inter-item similarity file (LSA or Word2Vec).
"""

def recode_for_spc(data_recs, data_pres):
    '''Helper method to code the data for calculating the serial position curve
    and the probability of first recall curve.'''

    ll = data_pres.shape[1]
    maxlen = ll * 2

    rec_lists = []
    for i in range(len(data_recs)):
        this_list = data_recs[i]
        pres_list = data_pres[i]

        this_list = this_list[this_list > 0]

        # get indices of first place each unique value appears
        indices = np.unique(this_list, return_index=True)[1]

        # get each unique value in array (by first appearance)
        this_list_unique = this_list[sorted(indices)]

        # get the indices of these values in the other list, and add 1
        list_recoded = np.nonzero(this_list_unique[:, None] == pres_list)[1] + 1

        # re-pad with 0's so we can reformat this as a matrix again later
        recoded_row = np.pad(list_recoded, pad_width=(
            0, maxlen - len(list_recoded)),
                             mode='constant', constant_values=0)

        # append to running list of recoded rows
        rec_lists.append(recoded_row)

    # reshape as a matrix
    recoded_lists = np.asmatrix(rec_lists)

    return recoded_lists


def get_spc_pfr(rec_lists, ll):
    """Get serial position curve (SPC) and the probability of first recall (PFR)
    for the recoded lists."""

    spclists = []
    pfrlists = []
    for each_list in rec_lists:

        each_list = each_list[each_list > 0]

        # init. list to store whether or not an item was recalled
        spc_counts = np.zeros((1, ll))
        pfr_counts = np.zeros((1, ll))

        # get indices of where to put items in the list;
        # items start at 1, so index needs to -1
        spc_count_indices = each_list - 1
        spc_counts[0, spc_count_indices] = 1

        if each_list.shape[1] <= 0:
            continue
        else:
            # get index for first item in list
            pfr_count_index = each_list[0, 0] - 1
            pfr_counts[0, pfr_count_index] = 1

            spclists.append(np.squeeze(spc_counts))
            pfrlists.append(np.squeeze(pfr_counts))

    # if no items were recalled, output a matrix of 0's
    if not spclists:
        spcmat = np.zeros((rec_lists.shape[0], ll))
    else:
        spcmat = np.array(spclists)

    if not pfrlists:
        pfrmat = np.zeros((rec_lists.shape[0], ll))
    else:
        pfrmat = np.array(pfrlists)

    # get mean and unbiased sample variance for spc and pfc values,
    # taken across sessions
    spc_mean = np.nanmean(spcmat, axis=0)
    spc_var = np.var(spcmat, axis=0, ddof=1)

    pfr_mean = np.nanmean(pfrmat, axis=0)
    pfr_var = np.var(pfrmat, axi=0, ddof=1)

    return spc_mean, spc_var, pfr_mean, pfr_var


def obj_func(param_vec):
    """Define the error function that we want to minimize"""

    # the setup below takes in parameters for fitting the source-code
    # enabled version of the model.
    param_dict = {

        'beta_enc': param_vec[0],
        'beta_rec': param_vec[1],
        'gamma_fc': param_vec[2],
        'gamma_cf': param_vec[3],
        'scale_fc': 1 - param_vec[2],
        'scale_cf': 1 - param_vec[3],

        'phi_s': param_vec[4],
        'phi_d': param_vec[5],
        'kappa': param_vec[6],

        'eta': param_vec[7],
        's_cf': param_vec[8],
        's_fc': 0.0,
        'beta_rec_post': param_vec[9],
        'omega': param_vec[10],
        'alpha': param_vec[11],
        'c_thresh': param_vec[12],
        'dt': 10.0,

        'lamb': param_vec[13],
        'beta_source': param_vec[14],
        #'beta_source': 0.0,            # set beta_source to 0.0 to run a source-code free CMR2
        'beta_distract': param_vec[15],
        'rec_time_limit': 75000,

        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,

        'nlists_for_accumulator': 2,

        'L_CF_NW': param_vec[3],  # NW quadrant - set to gamma_cf value
        'L_CF_NE': param_vec[16],  # NE quadrant - set to gamma_emot value
        'L_CF_SW': 0.0,  # SW quadrant - set constant to 0.0
        'L_CF_SE': 0.0,  # SE quadrant - set constant to 0.0

        'L_FC_NW': param_vec[2],  # NW quadrant - set to gamma_fc value
        'L_FC_NE': 0.0,  # NE quadrant - set constant to 0.0
        'L_FC_SW': param_vec[2],  # SW quadrant - set to gamma_fc value
        'L_FC_SE': 0.0  # SE quadrant - set constant to 0.0
    }

    # The file located at this path tells CMR2/CMR3 how to break up a unified sheet
    # of lists for an individual subject into specific sessions. It contains a 1
    # for each list in session 1 (for a total of 24 one's),
    # a 2 for each list in session 2 (for a total of 24 two's),
    # and so forth.
    subject_id_path = './division_locs_ind1.txt'

    # Run the model and obtain its output
    rec_nos, times = CMR2.run_CMR2( LSA_path=LSA_path,
                                    LSA_mat=LSA_mat,
                                    data_path=data_path,
                                    params=param_dict,
                                    sep_files=False,
                                    source_info_path=source_info_path,
                                    nsource_cells=2,
                                    subj_id_path = subject_id_path)

    # Code the output so that it is in the proper format for the SPC and PFR
    # calculations.
    cmr_recoded_output = recode_for_spc(rec_nos, data_pres)

    # get the model's spc and pfr predictions:
    (this_spc, this_spc_var, this_pfr, this_pfr_var) = get_spc_pfr(cmr_recoded_output, ll)

    # be careful not to divide by 0! some param sets may output 0-variance vec's.
    # if this happens, just leave all outputs alone.
    if np.nansum(this_spc_var) == 0 or np.nansum(this_pfr_var) == 0:
        print("np.nansum equaled 0")
        this_spc_var[range(len(this_spc_var))] = 1
        this_pfr_var[range(len(this_pfr_var))] = 1

    ##########
    #
    #   Calculate error values
    #
    ##########

    # spc
    e1_a = np.subtract(target_spc[:10], this_spc[:10]) # re-fit for just originals
    e1_a_norm = np.divide(np.power(e1_a, 2), target_spc_var[:10]) # re-fit for just originals

    e1_b = np.subtract(target_spc[21:], this_spc[21:])
    e1_b_norm = np.divide(np.power(e1_b, 2), target_spc_var[21:])

    e1_c = np.subtract(target_spc[10:15], this_spc[10:15]) # re-fit for just originals
    e1_c_norm = np.divide(np.power(e1_c, 2), target_spc_var[10:15]) # re-fit for just originals

    # pfr
    e2_a = np.subtract(target_pfr[:15], this_pfr[:15])
    e2_a_norm = np.divide(np.power(e2_a, 2), target_pfr_var[:15])

    # this error includes subsections of the spc and pfr
    chi2_error = (np.nansum(e1_a_norm)      # spc, first 10 values
                  + np.nansum(e1_b_norm)    # spc, last 3 values
                  + np.nansum(e1_c_norm)    # spc, middle 5 values
                  + np.nansum(e2_a_norm))   # pfr, first 15 values

    # return the chi^2 error term
    return chi2_error


def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
        minstep=1e-8, minfunc=1e-8, debug=False, aimed_swarm=False):
    """
    Perform a particle swarm optimization (PSO)
   
    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
   
    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal 
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint 
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
   
    """
   
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
   
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    
    # Check for constraint function(s) #########################################
    obj = lambda x: func(x, *args, **kwargs)
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = lambda x: np.array([0])
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = lambda x: np.array([y(x, *args, **kwargs) for y in ieqcons])
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = lambda x: np.array(f_ieqcons(x, *args, **kwargs))
        
    def is_feasible(x):
        check = np.all(cons(x)>=0)
        return check

    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fp = np.zeros(S)  # best particle function values
    g = []  # best swarm position
    fg = 1e100  # artificial best swarm position starting value

    ###### Original code ######
    #fp_comp = np.zeros(S)
    #for i in range(S):
    #    fp_comp[i] = obj(p[i, :])
    ###########################
    iter0_tic = time.time()

    # os.O_CREAT --> create file if it does not exist
    # os.O_EXCL --> error if create and file exists
    # os.O_WRONLY --> open for writing only
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    obj_func_timer = time.time()
    rmse_list = []
    for idx, n in enumerate(p):

        match_file = subj_id + '0tempfile' + str(idx) + '.txt'
        try:
            # try to open the file
            fd = os.open(match_file, flags)

            # run this CMR object and get out the fit value
            rmse = func(n)
            rmse_list.append(rmse)

            # set up file contents
            file_input = str(rmse) + "," + str(idx)

            # open the empty file that accords with this
            os.write(fd, file_input.encode())
            os.close(fd)

            print("I did file: " + match_file)

        # If the error raised is that the file already exists, carry on.
        except OSError as e:
            if e.errno == errno.EEXIST:
                continue
            # Otherwise, if we got a different type of error, raise the error.
            else:
                raise

    print("Obj func round 0: " + str(time.time() - obj_func_timer))

    ########
    #
    #   Read in all the files and grab their fit values
    #
    ########

    # read in the files that start with "tempfile" and sort numerically
    rmse_paths = sorted(glob(subj_id + '0tempfile*'))

    # sanity check; make sure not too many temp files
    if len(rmse_paths) > S:
        raise ValueError("No. of temp files exceeds swarm size")

    # if length of rmse_paths is less than the swarm size (S),
    # then we are not done.  Wait 5 seconds and check again to see
    # if rmse_paths is now the right length.
    tic = time.time()
    while len(rmse_paths) < S:

        # don't check the paths more than once every 5 seconds
        time.sleep(5)

        # grab the paths again
        rmse_paths = sorted(glob(subj_id + '0tempfile*'))

        #####
        #
        #   Test all files to see if they are empty
        #
        #####

        # if more than wait_time minutes passes and it is not the right length,
        # then raise a value error / stop the code.
        if (time.time() - tic) > wait_time:
            raise ValueError(
                "Spent more than the allotted time waiting for processes to complete")

    ######
    #
    #   Check and see if any files are empty -- avoid race conditions
    #
    ######

    # check through all the paths to see if any are empty
    any_empty = True

    mini_tic = time.time()  # track time
    while any_empty:
        num_nonempty = 0

        # see if all paths are full
        for sub_path in rmse_paths:
            try:
                with open(
                        sub_path) as tfile:  # open file to avoid race c.
                    first_line = tfile.readline()  # read first line

                    if first_line == '':  # if first line is empty,
                        tfile.close()  # close file and break
                        break
                    else:  # if first line is not empty,
                        num_nonempty += 1  # increment count of non-empty files
                        tfile.close()
            except OSError as e:
                if e.errno == errno.EEXIST:  # as long as file exists, continue
                    continue
                else:           # if it was a different error, raise the error
                    raise

        if num_nonempty >= len(rmse_paths):
            any_empty = False

        # prevent infinite loops; run for max of the allotted time
        if (time.time() - mini_tic) > wait_time:
            raise ValueError("I crashed waiting >the allotted time for rmse files to be full")
            break

    # read in tempfiles and get their rmse's & indices
    rmse_list0 = []
    for mini_path in rmse_paths:
        rmse_vec = np.genfromtxt(mini_path, delimiter=',', dtype=None)
        rmse_list0.append(rmse_vec.tolist())

    rmse_list0_sorted = sorted(rmse_list0, key=lambda tup: tup[1])
    fp = [tup[0] for tup in rmse_list0_sorted]

    #############
    #
    #   Initialize particle positions, velocities, & best position prior to
    #   beginning the swarm
    #
    #############

    for i in range(S):
        # Initialize the particle's position
        x[i, :] = lb + x[i, :]*(ub - lb)
   
        # Initialize the particle's best known position
        p[i, :] = x[i, :]
       
        # Calculate the objective's value at the current particle's
        # fp[i] = obj(p[i, :])
       
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        # if i==0:
        #    g = p[0, :].copy()
        # # if i==0:
        # #     g = aim_direction
        #

        # if you want to aim the swarm in a particular direction,
        # place that direction's location in the set of particle swarm
        # positions (i.e., into x) and set its direction as the swarm's
        # best direction starting point (i.e., g)
        if aimed_swarm:
            if i==0:
                x[i, :] = aim_direction
                p[i, :] = aim_direction.copy()
                g = aim_direction.copy()
        else:
            if i==0:
                g = p[0, :].copy()

        # If the current particle's position is better than the swarm's,
        # update the best swarm position
        if fp[i]<fg and is_feasible(p[i, :]):
            fg = fp[i]
            g = p[i, :].copy()
       
        # Initialize the particle's velocity
        v[i, :] = vlow + np.random.rand(D)*(vhigh - vlow)

    # if not already saved by another program / node,
    # save out the parameters' positions (x), best known positions (p),
    # and velocities (v).
    param_files = [subj_id+'0xfile.txt', subj_id+'0pfile.txt', subj_id+'0vfile.txt']
    param_entries = [x, p, v]
    for i in range(3):

        # check and see if the xfile, pfile, and vfile files have been
        # written.  If not, write them.
        try:
            fd = os.open(param_files[i], flags)

            np.savetxt(param_files[i], param_entries[i])

            os.close(fd)

        # OSError -> type of error raised for operating system errors
        except OSError as e:
            if e.errno == errno.EEXIST:  # errno.EEXIST means file exists
                continue
            else:
                raise

        # save out the x, p, or v parameter values, respectively
        # np.savetxt(param_files[i], param_entries[i])


    toc = time.time()
    print("Iteration %i time: %f" % (0, toc - iter0_tic))

    ######
    #
    #   Swarm begins here
    #
    ######

    # Iterate until termination criterion met ##################################
    it = 1
    while it <= maxiter:

        # time how long this iteration took
        iter_tic = time.time()
        print("\nBeginning iteration " + str(it) + ":")

        # if the rmses file is already created for this iteration,
        # and it is non-empty (nbytes > 0),
        # then read in that file instead of re-calculating the rmse values
        this_rmses_file = subj_id+"rmses_iter"+str(it)
        if (os.path.isfile(this_rmses_file)
            and (os.path.getsize(this_rmses_file) > 0.0)):
            rmse_list0 = np.loadtxt(this_rmses_file)

        else:
            #rp = np.random.uniform(size=(S, D))
            #rg = np.random.uniform(size=(S, D))

            # read in the noise files for this iteration
            rp = np.loadtxt(noise_path + 'rp_iter' + str(it))
            rg = np.loadtxt(noise_path + 'rg_iter' + str(it))

            # every 10 iterations, cleanup old temp files from the cwd
            old_rmse_paths = [] # init just in case

            # leave a buffer of the last 3 iterations' files
            # e.g., on iteration 10, we'll only clean up iter files 0-7
            if it == 10:
                old_rmse_paths = glob(subj_id + '[0-7]tempfile*')
            elif it == 20:
                old_rmse_paths = glob(subj_id + '[8-9]tempfile*') + glob(subj_id + '1[0-7]tempfile*')
            elif it == 30:
                old_rmse_paths = glob(subj_id + '1*tempfile*') + glob(subj_id + '2[0-7]tempfile*')
            elif it == 40:
                old_rmse_paths = glob(subj_id + '2*tempfile*') + glob(subj_id + '3[0-7]tempfile*')
            elif it == 50:
                old_rmse_paths = glob(subj_id + '3*tempfile*') + glob(subj_id + '4[0-7]tempfile*')
            elif it == 60:
                old_rmse_paths = glob(subj_id + '4*tempfile*') + glob(subj_id + '5[0-7]tempfile*')
            elif it == 70:
                old_rmse_paths = glob(subj_id + '5*tempfile*') + glob(subj_id + '6[0-7]tempfile*')
            elif it == 80:
                old_rmse_paths = glob(subj_id + '6*tempfile*') + glob(subj_id + '7[0-7]tempfile*')
            elif it == 90:
                old_rmse_paths = glob(subj_id + '7*tempfile*') + glob(subj_id + '8[0-7]tempfile*')

            # mark cleanup points
            cleanup_points = [10, 20, 30, 40, 50, 60, 70, 80, 90]

            # if we have reached a cleanup point, clean up!
            if it in cleanup_points:
                for old_path in old_rmse_paths:
                    try:
                        # try to open the file (prevent race conditions)
                        cfile = os.open(old_path, os.O_RDONLY)

                        # if successfully opened the file, hold for a
                        # hundredth of a second with it open
                        time.sleep(.01)

                        # close the file
                        os.close(cfile)

                        # remove the file
                        os.remove(old_path)
                    except OSError as e:
                        # if can't open the file but file exists,
                        if e.errno == errno.EEXIST:
                            continue    # if file exists but is closed, move along to next file path
                        else:
                            continue    # if file does not exist, this is also okay; move along

            ###
            #   Read in the position, best, & velocity files from previous iteration
            ###
            x = []
            p = []
            v = []
            # make sure we get a full file with S entries
            no_inf_loops = time.time()
            while (len(x) < S) or (len(p) < S) or (len(v) < S):

                x = np.loadtxt(subj_id + str(it-1) + 'xfile.txt')
                p = np.loadtxt(subj_id + str(it-1) + 'pfile.txt')
                v = np.loadtxt(subj_id + str(it-1) + 'vfile.txt')

                # When we are getting out a full file, keep going
                if len(x) == S and len(p) == S and len(v) == S:
                    break
                else:
                    time.sleep(2)   # sleep 2 seconds before we try again

                if (time.time() - no_inf_loops) > wait_time:
                    raise ValueError("Incomplete entries in x, p, or v file")

            ###
            #   First update all particle positions
            ###
            for i in range(S):

                # Update the particle's velocity
                v[i, :] = omega*v[i, :] + phip*rp[i, :]*(p[i, :] - x[i, :]) + \
                          phig*rg[i, :]*(g - x[i, :])

                # Update the particle's position, correcting lower and upper bound
                # violations, then update the objective function value
                x[i, :] = x[i, :] + v[i, :]
                mark1 = x[i, :]<lb
                mark2 = x[i, :]>ub
                x[i, mark1] = lb[mark1]
                x[i, mark2] = ub[mark2]

            ###
            #  Then get the objective function for each particle
            ###

            obj_func_timer_it = time.time()
            rmse_list = []
            for idx, n in enumerate(x):

                match_file = subj_id + str(it) + 'tempfile' + str(idx) + '.txt'
                try:
                    # try to open the file
                    fd = os.open(match_file, flags)

                    # run this CMR object and get out the rmse
                    rmse = func(n)
                    rmse_list.append(rmse)

                    # set up file contents
                    file_input = str(rmse) + "," + str(idx)

                    # write the file contents
                    os.write(fd, file_input.encode())

                    # close the file
                    os.close(fd)

                    print("I did file: " + match_file)

                # OSError -> type of error raised for operating system errors
                except OSError as e:
                    if e.errno == errno.EEXIST:  # errno.EEXIST means file exists
                        continue
                    else:
                        raise

            print("Iteration " + str(it) + " timer: "
                  + str(time.time() - obj_func_timer_it))

            # read in the files that start with "tempfile" and sort numerically
            rmse_paths = sorted(glob(subj_id + str(it) + 'tempfile*'))

            # sanity check; make sure not too many temp files
            if len(rmse_paths) > S:
                raise ValueError("No. of temp files exceeds swarm size")

            # if length of rmse_paths is less than the swarm size (S),
            # then we are not done.  Wait 5 seconds and check again to see
            # if rmse_paths is now the right length.
            tic = time.time()
            while len(rmse_paths) < S:

                # don't check the paths more than once every 5 seconds
                time.sleep(5)

                # grab the paths again
                rmse_paths = sorted(glob(subj_id + str(it) + 'tempfile*'))

                # if more than wait_time minutes passes and it is not the right length,
                # then raise a value error / stop the code.
                if (time.time() - tic) > wait_time:
                    raise ValueError(
                        "Spent more than the allotted time waiting for processes to complete")

            ######
            #
            #   Check and see if files are empty -- avoid race conditions
            #
            ######

            # check through all the paths to see if any are empty
            any_empty = True

            mini_tic = time.time()  # track time
            while any_empty:

                num_nonempty = 0

                # see if all paths are full
                for sub_path in rmse_paths:
                    try:
                        with open(
                                sub_path) as tfile:  # open file to avoid race c.
                            first_line = tfile.readline()  # read first line

                            if first_line == '':  # if first line is empty,
                                tfile.close()  # close file and break
                                break
                            else:  # if first line is not empty,
                                num_nonempty += 1  # increment count of non-empty files
                                tfile.close()
                    except OSError as e:
                        if e.errno == errno.EEXIST:  # as long as file exists, continue
                            continue
                        else:
                            raise

                if num_nonempty >= len(rmse_paths):
                    any_empty = False

                # prevent infinite loops; run for max of 5 minutes
                if (time.time() - mini_tic) > wait_time:
                    raise ValueError("Exceeded the allotted wait time for full rmse file paths")
                    break

            # read in tempfiles and get their rmse's & indices
            rmse_list0 = []
            for mini_path in rmse_paths:
                rmse_vec = np.genfromtxt(mini_path, delimiter=',', dtype=None)
                rmse_list0.append(rmse_vec.tolist())

        # get all the rmse values into one array / list
        rmse_sorted = sorted(rmse_list0, key=lambda tup: tup[1])
        fx = [tup[0] for tup in rmse_sorted]

        np.savetxt(subj_id + 'rmses_iter'+str(it),rmse_sorted)

        ###
        # Then compare all the particles' positions
        ###
        for i in range(S):
            
            # Compare particle's best position (if constraints are satisfied)
            if fx[i]<fp[i] and is_feasible(x[i, :]):
                p[i, :] = x[i, :].copy()
                fp[i] = fx[i]

                # Compare swarm's best position to current particle's position
                # (Can only get here if constraints are satisfied)
                if fx[i]<fg:
                    if debug:
                        print('New best for swarm at iteration {:}: {:} {:}'.format(it, x[i, :], fx))

                    tmp = x[i, :].copy()
                    stepsize = np.sqrt(np.sum((g-tmp)**2))
                    if np.abs(fg - fx[i])<=minfunc:
                        print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
                        return tmp, fx[i]
                    elif stepsize<=minstep:
                        print('Stopping search: Swarm best position change less than {:}'.format(minstep))
                        return tmp, fx[i]
                    else:
                        g = tmp.copy()
                        fg = fx[i]

        ####
        #   Save this iteration of param files so that we can start again
        ####
        param_files = [subj_id + str(it)+'xfile.txt', subj_id + str(it)+'pfile.txt',
                       subj_id + str(it)+'vfile.txt']
        param_entries = [x, p, v]
        for i in range(3):

            # check and see if the xfile, pfile, and vfile files have been
            # written.  If not, write them.
            try:
                # try to open the file
                fd = os.open(param_files[i], flags)

                np.savetxt(param_files[i], param_entries[i])

                os.close(fd)

            # OSError -> type of error raised for operating system errors
            except OSError as e:
                if e.errno == errno.EEXIST:  # errno.EEXIST means file exists
                    continue
                else:
                    raise

        toc = time.time()
        print("Iteration %i time: %f" % (it, toc-iter_tic))

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    
    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    return g, fg


def main():
    #########
    #
    #   Define some helpful global (yikes, I know!) variables.
    #
    #########

    global ll, data_pres, data_rec, LSA_path, data_path, LSA_mat
    global target_spc, target_spc_var, target_pfr, target_pfr_var

    global subj_id
    global source_info_path

    global wait_time, root_path, noise_path

    global pres_nos_unique

    # set root path where data and most other files are stored
    root_path = 'PythonProjects/CMR3_Code/'
    noise_path = root_path + 'noise_files/'

    #############
    #
    #   set lower and upper bounds
    #
    #############

    # from left to right (lb and ub parameters):
    # beta_enc,
    # beta_rec,
    # gamma_fc,
    # gamma_cf,
    # phi_s,
    # phi_d,
    # kappa,
    # eta,
    # s_cf,
    # beta_rec_post,
    # omega,
    # alpha,
    # c_thresh,
    # lamb
    # beta_source
    # beta_distract
    # LCF_ne

         # be  #br  #gfc #gcf  #phS #phD #k   #eta   #s   #bp  #om  #al  #cth  #lam #source #distract #lcf_ne
    lb = [.2, .50, .60, .60, 0.1, 0.1, .01, .01, 0.5, 0.1, 15.0, 0.8, 0.1, .01, .10, .10, .10]
    ub = [.6, .99, .99, .99, 3.0, 1.5, .50, .50, 3.0, 1.0, 30.0, 1.0, 2.0, .50, .99, .99, .99]

    # set particle swarm configuration
    max_iter = 30
    S = 200

    # define the amount of time you want particles to wait for all other particles to finish
    nminutes = 10
    wait_time = 60 * nminutes

    ##########
    #
    #   Set dataset variables
    #
    ##########

    # set list length
    ll = 24
    # set n sessions
    nsessions = 24

    ##########
    #
    #   Set up semantic similarity matrices
    #
    ##########

    # set LSA path
    LSA_path = root_path + 'w2v.txt'

    # load inter-item similarity matrix
    LSA_mat = np.loadtxt(LSA_path)

    ##########
    #
    #   Fit all subjects
    #
    ##########

    # read in the list of subject ID's you want to fit
    subjects = np.loadtxt(root_path + 'complete_subjects_list.txt',
                          dtype='bytes', delimiter='\n').astype(str)
    subjects = subjects.tolist()

    # list any subjects that you have already run and want the code to skip
    already_complete = []

    for subject in subjects:
        print("I reached ", subject)

        # if subject has already been run, skip
        if subject in already_complete:
            print("I skipped ", subject)
            continue
        else:
            ##########
            #
            #   Obtain the target values that we will fit
            #
            ##########

            print("I started running ", subject)

            # set subject ID
            subj_id = subject

            # Set paths to where presented, recalled, and source-info files are located
            data_path = root_path + '/pres_files/pres_nos_' + subject + '.txt'
            source_info_path = root_path+'eval_files/eval_codes_'+subject+'.txt'
            rec_path = root_path + '/rec_files/rec_nos_' + subj_id + '.txt'

            # load presented items and recalled items
            data_pres = np.loadtxt(data_path, delimiter=',')
            data_rec = np.loadtxt(rec_path, delimiter=',')

            # get the person's presented-item numbers, flattened into a convenient shape
            pres_nos_flat = np.reshape(data_pres,
                                       newshape=(data_pres.shape[0] * data_pres.shape[1],))
            pres_nos_unique = np.unique(pres_nos_flat)

            # recode lists for spc, pfc, and lag-CRP analyses
            recoded_lists = recode_for_spc(data_rec, data_pres)

            # get target spc & pfr values
            target_spc, target_spc_var, target_pfr, target_pfr_var = get_spc_pfr(recoded_lists, ll)

            # set any variance values that are equal to 0.0 equal to 1.0 so that
            # we do not later divide by zero when we are norming these values.
            target_spc_var[target_spc_var == 0.0] = 1.0
            target_pfr_var[target_pfr_var == 0.0] = 1.0

            ###########
            #
            #   Get the right data files
            #
            ###########

            # make a new directory for this subject's information
            save_path = root_path + 'CMR3_Fit_Outputs/'
            new_dir = save_path + subject + '_runfiles'

            try:
                os.mkdir(new_dir)
            except OSError as e:
                # if directory already exists, that's okay; move along.
                if e.errno == errno.EEXIST:
                    pass
                # if it's some other error, raise it
                else:
                    raise

            # change working directory to the new folder
            os.chdir(new_dir)

            # define global subject ID
            subj_id = subject

            start_time = time.time()

            # run particle swarm
            xopt, fopt = pso(obj_func, lb, ub, swarmsize=S, maxiter=max_iter,
                             debug=False, aimed_swarm=False)

            print(xopt)
            print("Run time: " + str(time.time() - start_time))

            ##########
            #
            #   clean up temp files; space by 5-secs to avoid a race condition
            #
            ##########

            time.sleep(5)
            tempfile_paths = glob('*tempfile*')

            for mini_path in tempfile_paths:
                try:
                    # try to open the file (prevent race conditions)
                    cfile = os.open(mini_path, os.O_RDONLY)

                    # if successfully opened the file, hold for a
                    # hundredth of a second with it open
                    time.sleep(.01)

                    # close the file
                    os.close(cfile)

                    # remove the file
                    os.remove(mini_path)
                except OSError as e:
                    # if can't open the file but file exists,
                    if e.errno == errno.EEXIST:
                        # if file exists but is closed,
                        # move along to next file path
                        continue
                    else:
                        # if file does not exist, this is also okay;
                        # move along
                        continue

            fopt_list = []
            fopt_list.append(fopt)

            np.savetxt('xopt_'+subj_id+'.txt', xopt, delimiter=',', fmt='%f')
            np.savetxt('fopt_'+subj_id+'.txt', fopt_list, delimiter=',', fmt='%f')


if __name__ == "__main__": main()
