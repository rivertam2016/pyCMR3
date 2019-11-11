import numpy as np
import scipy.stats
import warnings
from glob import glob

##########
#
#   Define helper functions
#
##########


def get_possible_transitions(item, remaining_items,
                             pres_nos_unique, mini_sim_mat, bin_middles):

    """Calculate the numbers of possible transitions left for this item.
    Could this item transition to an item from bin 1, 2, 3, etc? Or are
    there no items left in a particular category to which to transition?"""

    remaining_similarities = []

    row, = np.where(pres_nos_unique == item)[0]
    for remaining_item in remaining_items:

        # get column for comparison value
        col, = np.where(pres_nos_unique == remaining_item)[0]

        # we can't transition to the same item, so skip this comparison:
        if row == col:
            continue
        else:
            # get similarity value
            sim = mini_sim_mat[row, col]

            # append to remaining_similarities list
            remaining_similarities.append(sim)

    # get bin locs (index+1) for each remaining transition
    remaining_stamps = np.digitize(remaining_similarities, bin_middles,
                                   right=False)
    possible_transitions = np.bincount(remaining_stamps)

    # format into a 0 x 4 vector (i.e., 0 x nbins)
    poss_trans_formatted = np.zeros_like(bin_middles)

    # if there are nonzero, coded values in the counted up bins (i.e., we
    # have real stamps by which the transitions are coded & it wasn't empty):
    if len(possible_transitions) > 1:
        # format the possible transitions so they can easily form the
        # denominator for later dividing out observed transition values
        for bin_idx, bin in enumerate(possible_transitions[1:]):
            poss_trans_formatted[bin_idx] += bin

    return poss_trans_formatted


def get_actual_transitions(idx, item, test_list, pres_nos_unique,
                           mini_sim_mat, bin_middles):
    """get the actual transition that this item made"""

    # if we're at the end of the list, no transition was made, but a
    # transition *could* have been made.  So, all categories just get 0's.
    if idx == len(test_list) - 1:
        transition_scores = np.zeros_like(bin_middles)
    else:
        # get IDs for this item and the next item
        this_item = item
        next_item = test_list[idx + 1]

        # get indices for where these items' similarity values lie in the
        # semantic similarity matrix for this subjects' presented items
        row_idx, = np.where(pres_nos_unique == this_item)[0]
        col_idx, = np.where(pres_nos_unique == next_item)[0]

        # look up the similarity between this item and the next
        sim_val = mini_sim_mat[row_idx, col_idx]

        # get what bin this item is in
        stamp = np.digitize(sim_val, bin_middles, right=False)

        # initialize probability scores (0 = the transition occured;
        #                                1 = the transition did not occur.)
        transition_scores = np.zeros_like(bin_middles)
        transition_scores[stamp - 1] = 1.0

    return transition_scores


def get_sem_crp_one_list(rec_list, pres_list, pres_nos_unique,
                         mini_sim_mat, bin_middles):
    """Get the semantic crp probability scores for this list"""

    # keep a running list of which items have been recalled, and therefore
    # are no longer in the running as possible transition values
    recalled_items = []
    remaining_items = pres_list.copy()
    remaining_items = remaining_items.tolist()

    # track probability scores for each item's transition
    prob_scores = []

    # for each item that was recalled,
    for idx, item in enumerate(rec_list):

        # get no. and type of possible transitions remaining for this item
        poss_trans_formatted = get_possible_transitions(item, remaining_items,
                                                        pres_nos_unique,
                                                        mini_sim_mat, bin_middles)

        # get actual transition category that was made
        actual_trans = get_actual_transitions(idx, item, rec_list,
                                              pres_nos_unique,
                                              mini_sim_mat, bin_middles)

        # get probability score for each category
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            prob_score = np.divide(actual_trans, poss_trans_formatted)

        # gather the probability score for each item's transition
        prob_scores.append(prob_score)

        # pop the just-recalled item from remaining items and append to recalled
        remaining_items_arr = np.asarray(remaining_items, dtype=np.int16)

        # pop the item that was just recalled off of the remaining
        # items list and add it to the recalled items list.
        pop_idx, = np.where(remaining_items_arr == item)[0]

        recalled_items.append(remaining_items.pop(pop_idx))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_prob_scores = np.nanmean(prob_scores, axis=0)

    return mean_prob_scores


def get_semantic_crp(rec_nos, pres_nos, pres_nos_unique,
                     mini_sim_mat, bin_middles):

    """get semantic crp for each list in the recalled & presented matrices"""
    sem_crp_across_lists = []
    for ii in range(len(rec_nos)):

        # print("\nWe are on list: " + str(ii))
        rec_list = rec_nos[ii]
        pres_list = pres_nos[ii]

        # clean the list

        # remove all intrusions:
        for idx, recalled_item in enumerate(rec_list):
            if recalled_item not in pres_list:
                rec_list[idx] = 0

        # remove all vocalizations & intrusion leftovers
        rec_list = rec_list[rec_list > 0]

        # remove all repeats
        sorted_unique_vals, unique_indices = np.unique(rec_list,
                                                       return_index=True)
        rec_list = rec_list[np.sort(unique_indices)]

        # skip lists in which the participant recalled 2 or fewer items
        if len(rec_list) <= 2:
            # print("List %i had too few recalls." % ii)
            continue
        else:
            # get semantic crp for this list
            prob_score_list = get_sem_crp_one_list(rec_list, pres_list,
                                                   pres_nos_unique,
                                                   mini_sim_mat, bin_middles)

            # append to list of sem-crp's across all lists
            sem_crp_across_lists.append(prob_score_list)

    sem_crp_subj = np.nanmean(np.asarray(sem_crp_across_lists), axis=0)
    sem_crp_subj_sem = scipy.stats.sem(np.asarray(sem_crp_across_lists), axis=0, nan_policy='omit')
    sem_crp_subj_std = np.nanstd(np.asarray(sem_crp_across_lists), axis=0, ddof=1)
    sem_crp_subj_var = np.nanvar(np.asarray(sem_crp_across_lists), axis=0, ddof=1)

    return sem_crp_subj, sem_crp_subj_sem, sem_crp_subj_std, sem_crp_subj_var


def get_subject_simmat(pres_nos_unique, sem_sim_mat):

    # set var for number of unique items
    nitems_unique = len(pres_nos_unique)

    # Create a mini-LSA matrix with just the items presented to this Subj.
    mini_sim_mat = np.zeros((nitems_unique, nitems_unique), dtype=np.float32)

    # Get list-item LSA indices
    for row_idx, item_i in enumerate(pres_nos_unique):

        # get item's index in the larger LSA matrix
        this_item_idx = item_i - 1

        for col_idx, compare_item in enumerate(pres_nos_unique):
            # get ID of jth item for LSA cos theta comparison
            compare_item_idx = compare_item - 1

            # get cos theta value between this_item and compare_item
            cos_theta = sem_sim_mat[int(this_item_idx), int(compare_item_idx)]

            # print(cos_theta)

            # place into this session's LSA cos theta matrix
            mini_sim_mat[row_idx, col_idx] = cos_theta

    return mini_sim_mat

def main():

    ###########
    #
    #   Set paths, variables for the subject we are about to analyze
    #
    ##########

    # set path to semantic similarity matrix & root path for files
    sim_path = '/Users/KahaNinja/PycharmProjects/semCRP/w2v.txt'
    root_path = '/Users/KahaNinja/PycharmProjects/semCRP/'
    # rec_files_path = root_path + 'rec_files/'
    rec_files_path = root_path + 'output_files_dfr/'

    # get list of subjects
    # subjects = glob(rec_files_path + '*LTP*')
    subjects = glob(rec_files_path + '*resp_predicted_LTP*')

    global subject, sem_sim_mat, pres_nos, rec_nos, pres_nos_unique
    global bin_edges, bin_middles

    ##########
    #
    #   Create bins for what is a low, medium, and high sem. similarity value,
    #   using the entire word pool
    #
    ##########

    # set desired no. of bins
    nbins = 6

    # read in the full w2v matrix
    sem_sim_mat = np.loadtxt(sim_path, delimiter=' ', dtype=np.float32)

    # get out the edges for where we would like 4 bins of the item similarity values
    hist, bin_edges = np.histogram(sem_sim_mat, bins=nbins)

    # drop the max value from bin edges, for inputting into digitize
    bin_middles = bin_edges[:-1]

    # get sem crps across subjects
    sem_crps_subjects = []
    for path in subjects:

        subject = path[-10:-4]
        # print(subject)

        # get that subject's presented & recalled items paths
        pres_path = root_path + 'pres_files/pres_nos_' + subject + '.txt'

        # rec_path = root_path + 'rec_files/rec_nos_' + subject + '.txt'
        rec_path = rec_files_path + 'dfr_resp_predicted_' + subject + '.txt'

        ##########
        #
        #   Read in relevant files
        #
        ##########

        # read in the ID's for the items that were presented and recalled
        pres_nos = np.loadtxt(pres_path, delimiter=',', dtype=np.int16)
        rec_nos = np.loadtxt(rec_path, delimiter=',', dtype=np.int16)

        ##########
        #
        #   Create semantic similarity matrix for this participant
        #
        ##########

        # sort the items by ID number so we can make an orderly lookup matrix
        # / table, and, take only unique IDs
        pres_nos_flat = np.reshape(pres_nos, newshape=(
        pres_nos.shape[0] * pres_nos.shape[1],))
        pres_nos_unique = np.unique(pres_nos_flat)

        # get inter-item similarity matrix for just the items presented
        # to this subject
        mini_sim_mat = get_subject_simmat(pres_nos_unique, sem_sim_mat)

        # get the semantic crp for this subject
        sem_crp_this_subj = get_semantic_crp(rec_nos, pres_nos,
                                             pres_nos_unique,
                                             mini_sim_mat, bin_middles)

        sem_crps_subjects.append(sem_crp_this_subj)

    sem_crps_subjects = np.asarray(sem_crps_subjects)
    mean_sem_crp = np.nanmean(sem_crps_subjects, axis=0)
    sem_sem_crp = scipy.stats.sem(sem_crps_subjects, axis=0)

    # print(mean_sem_crp)
    # print(sem_sem_crp)

    np.savetxt('sem_crp_all_'+str(nbins)+'bins_cmr2_dfr.txt', np.asarray([mean_sem_crp,
                                                            sem_sem_crp]))

if __name__ == "__main__": main()