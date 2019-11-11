import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, './HelperFiles/')
from crp import recode_for_crp
from crp import crp
import sem_CRP as scrp
import pandas
import warnings

##########
#
#   Define some helper methods
#
##########

def recode_for_spc(data_recs, data_pres):
    """Helper method to recode data for an spc curve"""
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
    """Get spc and pfr for the recoded lists"""

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

    # get mean and sem's for spc and pfr
    spc_mean = np.nanmean(spcmat, axis=0)
    spc_sem  = scipy.stats.sem(spcmat, axis=0, nan_policy='omit')

    pfr_mean = np.nanmean(pfrmat, axis=0)
    pfr_sem  = scipy.stats.sem(pfrmat, axis=0, nan_policy='omit')

    return spc_mean, spc_sem, pfr_mean, pfr_sem


def emot_val(pres_lists, rec_lists):
    """Main method to calculate emotional valences, etc."""

    # Initialize lists to hold transition probability scores
    # for this subject, across lists
    list_probs = []

    # if no. presented lists != no. recalled lists, skip this session
    # (assumed some kind of recording error)
    if pres_lists.shape[0] != rec_lists.shape[0]:
        all_list_means = []
    else:
        for row in range(pres_lists.shape[0]):  # for each row:

            # get item nos. in row w/o 0's
            rec_row = rec_lists[row][rec_lists[row] != 0]
            pres_row = pres_lists[row]

            # if no responses, skip this list
            if len(rec_row) == 0:
                continue
            else:
                # get number of emotion words in presented list
                # Ng, P, N <-- order of counts in list
                emot_counts = getEmotCounts(pres_row)

                # if pres list doesn't have >= 1 of at least one emot. word,
                # then skip it
                if (emot_counts[0] < 1) \
                        or (emot_counts[1] < 1) or (emot_counts[2] < 1):
                    continue
                else:
                    # recode any repeats or intrusions as -1
                    # squeeze out all -1 values
                    cleaned_list = recode_rep_intrs(rec_row, pres_row)

                    # recode rec_row w/ emotional valence pool IDs
                    rec_list_emot = codeList(cleaned_list)

                    # remove all -1 values
                    rec_list_analyze = rec_list_emot[rec_list_emot != '-1']

                    # if N valid words recalled is < 2, skip list
                    if len(rec_list_analyze) < 2:
                        continue
                    else:
                        count_ng = 0
                        count_p  = 0
                        count_n  = 0

                        # iterate through the list and sum
                        # no. of val-val transitions that occur
                        for i in range(len(rec_list_analyze)):

                            # keep running count of # items of
                            # particular valence
                            this_item = rec_list_analyze[i]

                            # keep running count of # items of
                            # particular valence remaining
                            list_tot_ng = emot_counts[0]
                            list_tot_p  = emot_counts[1]
                            list_tot_n  = emot_counts[2]

                            ng_remaining = list_tot_ng - count_ng
                            p_remaining  = list_tot_p - count_p
                            n_remaining  = list_tot_n - count_n

                            # initialize transition scores
                            score_ng = 0
                            score_p  = 0
                            score_n  = 0

                            if i == 0:
                                if this_item == 'Ng':
                                    count_ng += 1
                                elif this_item == 'P':
                                    count_p += 1
                                elif this_item == 'N':
                                    count_n += 1
                            if i > 0:
                                # get previous item (base of the transition)
                                prev_item = rec_list_analyze[i - 1]

                                # get transition scores
                                if this_item == 'Ng':
                                    score_ng += 1
                                elif this_item == 'P':
                                    score_p  += 1
                                elif this_item == 'N':
                                    score_n  += 1

                                # get observed probabilities for this step
                                if ng_remaining != 0:
                                    obs_prob_ng  = score_ng / ng_remaining
                                else:
                                    obs_prob_ng  = np.nan

                                if p_remaining  != 0:
                                    obs_prob_p   = score_p / p_remaining
                                else:
                                    obs_prob_p   = np.nan

                                if n_remaining  != 0:
                                    obs_prob_n   = score_n / n_remaining
                                else:
                                    obs_prob_n   = np.nan

                                # append to appropriate base-pair list
                                if prev_item   == 'Ng':
                                    list_probs.append(
                                        [obs_prob_ng, obs_prob_p, obs_prob_n,
                                         np.nan, np.nan, np.nan,
                                         np.nan, np.nan, np.nan])
                                elif prev_item == 'P':
                                    list_probs.append(
                                        [np.nan, np.nan, np.nan,
                                         obs_prob_ng, obs_prob_p, obs_prob_n,
                                         np.nan, np.nan, np.nan])
                                elif prev_item == 'N':
                                    list_probs.append(
                                        [np.nan, np.nan, np.nan,
                                         np.nan, np.nan, np.nan,
                                         obs_prob_ng, obs_prob_p, obs_prob_n])

                                # update running item counts
                                if this_item == 'Ng':
                                    count_ng += 1
                                elif this_item == 'P':
                                    count_p += 1
                                elif this_item == 'N':
                                    count_n += 1

        # Ignore np.nanmean warning for averaging across NaN-only slices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # average down columns to get ave transition probs across lists
            all_list_means = np.nanmean(list_probs, axis=0)

            # get SEM down columns
            all_list_sem = scipy.stats.sem(list_probs, axis=0, nan_policy='omit')

    return all_list_means, all_list_sem


def getEmotCounts(presented_list):
    """Define helper function return the emotional valence counts
    for presented items"""

    # init no.s valenced items in this list
    sum_neg  = 0
    sum_pos  = 0
    sum_neut = 0

    # for each item in that list,
    for j in range(len(presented_list)):

        # if valid item ID,
        if presented_list[j] >= 0:

            # get index of this item in the valence key
            valence_key_index = int(presented_list[j]) - 1

            # get valence for this item
            val_j = word_valence_key.iloc[valence_key_index][1]

            # if item has valence, increment sum of valenced items
            if val_j < 4.0:
                sum_neg += 1
            elif val_j > 6.0:
                sum_pos += 1
            else:
                sum_neut += 1

    return [sum_neg, sum_pos, sum_neut]


def codeList(orig_list):
    """Define helper function to recode a given list into emotional pool IDs"""

    recoded_list = []
    for j in range(len(orig_list)):
        # if valid item ID,
        if orig_list[j] >= 0:

            # get index of this item in the valence key
            valence_key_index = int(orig_list[j]) - 1

            # get valence for this item
            val_j = word_valence_key.iloc[valence_key_index][1]

            # if item has valence, increment sum of valenced items
            if val_j < 4.0:
                recoded_list.append('Ng')
            elif val_j > 6.0:
                recoded_list.append('P')
            else:
                recoded_list.append('N')
        else:
            recoded_list.append('-1')

    return np.asarray(recoded_list)


def recode_rep_intrs(rec_list_0, pres_list_0):
    """Define helper function to recode repeats & intrusions as -1"""
    cleaned_list = np.zeros(len(rec_list_0))

    # for each item in list
    for i in range(len(rec_list_0)):

        item = rec_list_0[i]
        cleaned_list[i] = item

        # recode intrusions as -3
        if item not in pres_list_0:
            cleaned_list[i] = -3

        # recode repeats as -2
        if i > 0:
            if (item in rec_list_0[0:i - 1]):
                cleaned_list[i] = -2

    return cleaned_list


def main():

    """Graph results from previously generated model output, for one subject."""

    ###############
    #
    #   Choose which subject to graph.
    #
    ###############

    subject = 'LTP393'

    ###############
    #
    #   Define whether you are graphing output from the CMR3, CMR2, or eCMR version of the model.
    #   This won't change any of the analyses, but it will tell the code where to get the files from.
    #
    ###############

    use_CMR2 = False
    use_CMR3 = True
    use_eCMR = False

    # sanity check
    if (use_CMR2 and use_CMR3) or (use_CMR2 and use_eCMR) or (use_CMR3 and use_eCMR):
        raise ValueError("Please select just one model version to run.")
    elif not (use_CMR2 or use_CMR3 or use_eCMR):
        raise ValueError("Please select a model version to run.")

    if use_CMR2:
        model_name = 'CMR2'
    elif use_CMR3:
        model_name = 'CMR3'
    elif use_eCMR:
        model_name = 'eCMR'
    print("\nRunning %s on subject %s" % (model_name, subject))

    ###############
    #
    #   Set settings
    #
    ###############

    # set list length here
    ll = 24

    # decide whether to save figs out or not
    save_figs = True

    ###############
    #
    #   Get data output
    #
    ###############

    # set paths
    root_path = './'
    data_path = '../Data/pres_files/pres_nos_'+subject+'.txt'
    rec_folder = '../Data/rec_files/'
    data_rec_path = rec_folder + 'rec_nos_'+subject+'.txt'

    data_pres = np.loadtxt(data_path, delimiter=',', dtype='int')
    data_rec = np.loadtxt(data_rec_path, delimiter=',', dtype='int')

    # recode data lists for spc and pfr analyses
    recoded_lists = recode_for_spc(data_rec, data_pres)

    #----- Get spc & pfr
    target_spc, target_spc_sem, target_pfr, target_pfr_sem = \
        get_spc_pfr(recoded_lists, ll)

    #----- Get lag-crp values
    lag = 5
    data_crp_recoded_output = recode_for_crp(data_recs=data_rec, data_pres=data_pres)
    target_crp = crp(recalls=data_crp_recoded_output, listLength=ll, lag_num=lag)

    #----- Get Lag-CRP sections of interest
    center_val = lag

    target_left_crp = target_crp[center_val - 5:center_val]
    target_right_crp = target_crp[center_val + 1:center_val + lag + 1]

    #----- Get semantic crp

    # set desired no. of bins
    nbins = 6

    # set LSA path
    LSA_path = root_path + './HelperFiles/w2v.txt'

    # load inter-item similarity matrix
    LSA_mat = np.loadtxt(LSA_path)

    # get out the edges for where we would like 4 bins of the item similarity values
    hist, bin_edges = np.histogram(LSA_mat, bins=nbins)

    # drop the max value from bin edges, for inputting into digitize
    bin_middles = bin_edges[:-1]

    # get the person's presented item no's, flattened
    pres_nos_flat = np.reshape(data_pres, newshape=(data_pres.shape[0] * data_pres.shape[1],))
    pres_nos_unique = np.unique(pres_nos_flat)

    # make a mini sem-sim matrix from all items presented to this subj.
    mini_mat = scrp.get_subject_simmat(pres_nos_unique, LSA_mat)

    (target_sem_crp, target_sem_crp_sem,
     target_sem_crp_std, target_sem_crp_subj_var) = scrp.get_semantic_crp(data_rec, data_pres,
                                                           pres_nos_unique, mini_mat,
                                                           bin_middles)

    #----- Get emotional valence values

    # read in word valence key matching word ID to emotional valence
    global word_valence_key
    valence_key_path = root_path + 'HelperFiles/wordproperties_CSV.csv'
    word_valence_key = pandas.read_csv(valence_key_path)

    # get the emotional valence-coded responses for a single subject
    target_eval_mean, target_eval_sem = emot_val(data_pres, data_rec)

    ###############
    #
    #   Get the model output
    #
    ###############

    # Define where (what directory) the model-predicted responses are located in
    output_folder = 'output_files_' + model_name + '/'
    filestem = 'model_rec_nos_'

    rec_nos = np.loadtxt('./' + output_folder + filestem + subject + '.txt', delimiter=',')
    cmr_recoded_output = recode_for_spc(rec_nos, data_pres)

    #----- get the model's spc and pfr predictions:
    (this_spc, this_spc_sem, this_pfr,
    this_pfr_sem) = get_spc_pfr(cmr_recoded_output, ll)

    # ----- Get lag-crp values
    lag = 5
    cmr_crp_recoded_output = recode_for_crp(data_recs=rec_nos, data_pres=data_pres)

    this_crp = crp(recalls=cmr_crp_recoded_output, listLength=ll, lag_num=lag)

    center_val = lag
    this_left_crp = this_crp[center_val - 5:center_val]
    this_right_crp = this_crp[center_val + 1:center_val + lag + 1]

    #----- get semantic CRP
    (this_sem_crp, this_sem_crp_sem,
     this_sem_crp_std, this_sem_crp_var) = scrp.get_semantic_crp(rec_nos, data_pres,
                                                               pres_nos_unique, mini_mat,
                                                               bin_middles)

    #----- get the emotional valence-coded responses for a single subject
    this_eval_mean, this_eval_sem = emot_val(data_pres, rec_nos)


    ###############
    #
    #   Print out the output in case we want to see it
    #
    ###############

    print("\nData and Model spc's: ")
    print(target_spc)
    print(this_spc)

    print("\nData and Model pfr's: ")
    print(target_pfr)
    print(this_pfr)

    print("\nData and Model left lag crp's: ")
    print(target_left_crp)
    print(this_left_crp)

    print("\nData and Model right lag crp's: ")
    print(target_right_crp)
    print(this_right_crp)

    print("\nData and Model semantic crp's: ")
    print(target_sem_crp)
    print(this_sem_crp)

    print("\nData and Model eval means: ")
    print(target_eval_mean)
    print(this_eval_mean)

    ###############
    #
    #   Plot graphs
    #
    ###############

    # make a directory in which to save the figures
    figs_dir = 'Figs_' + model_name + '/'
    if not os.path.exists(figs_dir):
        os.mkdir(figs_dir)

    # line width
    lw = 2

    # gray color for CMR predictions
    gray = '0.50'

    #_______________________ plot spc
    plt.figure()
    xvals = range(1, ll+1, 1)     # ticks for x-axis

    plt.plot(xvals, this_spc, lw=lw, c=gray, linestyle='--', label='CMR3')
    plt.plot(xvals, target_spc, lw=lw, c='k', label='Data')

    plt.ylabel('Probability of Recall', size='large')
    plt.xlabel('Serial Position', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([0.5, ll+.5, 0, 1], size='large')
    plt.legend(loc='upper left')
    plt.title('Serial Position Curve', size='large')

    # save fig nicely
    if save_figs:
        plt.savefig(figs_dir + 'spc_'+subject+'.pdf', format='pdf', dpi=1000)

    #_______________________ plot pfr
    plt.figure()
    plt.plot(xvals, this_pfr, lw=lw, c=gray, linestyle='--', label='CMR3')
    plt.plot(xvals, target_pfr, lw=lw, c='k', label='Data')

    plt.title('Probability of First Recall', size='large')
    plt.xlabel('Serial Position', size='large')
    plt.ylabel('Probability of First Recall', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([0.5, ll+.5, 0, 1], size='large')
    plt.legend(loc='upper left')

    # save fig nicely
    if save_figs:
        plt.savefig(figs_dir + 'pfr_'+subject+'.pdf', format='pdf', dpi=1000)

    # _______________________ plot lag-CRP

    plt.figure()
    xvals_left = np.arange(-1 *len(target_left_crp), 0, step=1)
    xvals_right = np.arange(0, len(target_right_crp), step=1)

    plt.plot(xvals_left, this_left_crp, lw=lw, c=gray, linestyle='--', label='CMR3')
    plt.plot(xvals_left, target_left_crp, lw=lw, c='k', label='Data')

    plt.plot(xvals_right, this_right_crp, lw=lw, c=gray, linestyle='--', label='CMR3')
    plt.plot(xvals_right, target_right_crp, lw=lw, c='k', label='Data')

    plt.title('Lag-CRP', size='large')
    plt.xlabel('Inter-Item Lag', size='large')
    plt.ylabel('Conditional Probability of Recall', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([-5, 5, 0, 1], size='large')
    plt.legend(loc='upper left')

    # save fig nicely
    if save_figs:
        plt.savefig(figs_dir + 'lag_crp_'+subject+'.pdf', format='pdf', dpi=1000)

    # _______________________ plot semantic CRP

    plt.figure()
    xvals_scrp = np.arange(1, nbins+1, step=1)

    plt.plot(xvals_scrp, this_sem_crp, lw=lw, c=gray, linestyle='--', label='CMR3')
    plt.plot(xvals_scrp, target_sem_crp, lw=lw, c='k', label='Data')

    plt.title('Semantic CRP', size='large')
    plt.xlabel('Level of Semantic Similarity', size='large')
    plt.ylabel('Conditional Probability of Recall', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([0, nbins+1, 0, 1], size='large')
    plt.legend(loc='upper left')

    # save fig nicely
    if save_figs:
        plt.savefig(figs_dir + 'sem_crp_' + subject + '.pdf', format='pdf', dpi=1000)

    # _______________________ plot emotional valence interaction

    plt.figure()
    xvals_emot = np.arange(1, 3, step=1)

    neg_indices = [0, 3]
    pos_indices = [1, 4]
    target_neg_series = target_eval_mean[neg_indices]
    this_neg_series = this_eval_mean[neg_indices]

    target_pos_series = target_eval_mean[pos_indices]
    this_pos_series = this_eval_mean[pos_indices]

    plt.plot(xvals_emot, this_neg_series, lw=lw, c=gray, label='CMR3 Neg')
    plt.plot(xvals_emot, target_neg_series, lw=lw, c='k', label='Data Neg')

    plt.plot(xvals_emot, this_pos_series, lw=lw, c=gray, linestyle='--', label='CMR3 Pos')
    plt.plot(xvals_emot, target_pos_series, lw=lw, c='k', linestyle='--', label='Data Pos')

    plt.title('Emotional Clustering Effect', size='large')
    plt.xlabel('Base Emotion Type', size='large')
    plt.ylabel('Conditional Transition Probability', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.5), size='large')
    plt.axis([.5, 2.5, 0, .5], size='large')
    plt.legend(loc='upper left')

    # save fig nicely
    if save_figs:
        plt.savefig(figs_dir + 'eval_' + subject + '.pdf', format='pdf', dpi=1000)

    plt.show()


if __name__ == "__main__":
    main()


