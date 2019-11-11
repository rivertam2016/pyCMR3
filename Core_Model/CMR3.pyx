import mkl
mkl.set_num_threads(1)
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, srand
import scipy.io
import math
from glob import glob
import time
import sys
import os

def norm_vec(vec):
    """Helper method to normalize a vector"""

    # get the square root of the sum of each element in the dat vector squared
    denom = np.sqrt(np.sum(vec**2))

    # if the resulting denom value equals 0.0, then set this equal to 1.0
    if denom == 0.0:
        return vec
    else:
        # divide each element in the vector by this denom
        return vec/denom


def advance_context(c_in_normed, c_temp, this_beta):
    """Helper function to advance context"""

    # if row vector, force c_in to be a column vector
    if c_in_normed.shape[1] > 1:
        c_in_normed = c_in_normed.T
    assert(c_in_normed.shape[1] == 1)  # sanity check

    # if col vector, force c_temp to be a row vector
    if c_temp.shape[0] > 1:
        c_temp = c_temp.T
    assert(c_temp.shape[0] == 1)  # sanity check

    # calculate rho
    rho = (math.sqrt(1 + (this_beta**2)*
                     ((np.dot(c_temp, c_in_normed)**2) - 1)) -
           this_beta*np.dot(c_temp, c_in_normed))

    # update context
    updated_c = rho*c_temp + this_beta * c_in_normed.T

    # send updated_c out as a col vector
    if updated_c.shape[1] > 1:
        updated_c = updated_c.T

    return updated_c


class CMR2(object):
    """Initialize CMR2 class"""

    def __init__(self, params, nsources, source_info, LSA_mat, data_mat):
        """
        Initialize CMR2 object

        :param params: dictionary containing desired parameter values for CMR2
        :param nsources: If nsources > 0, model will implement source coding
            (See Polyn et al., 2009).

            Note that nsources refers to the number of cells you want to devote
            to source code information, not to the overall number of sources.
            For instance, you might want to represent a single source
            (e.g., emotion) as a vector of multiple source cells.

        :param source_info: matrix containing source-coding information
        :param LSA_mat: matrix containing LSA cos theta values between each item
            in the word pool.
        :param data_mat: matrix containing the lists of items that were
            presented to a given participant. Dividing this up is taken care
            of in the run_CMR2 method.
            You can also divide the data according to session, rather than
            by subject, if desired.  The run_CMR2 method is where you would
            alter this; simply submit sheets of presented items a session
            at a time rather than a subject at a time.

        ndistractors: There are 2x as many distractors as there are lists,
            because presenting a distractor is how we model the shift in context
            that occurs between lists.
            And, each list has a distractor that represents the distractor task.

            Additionally, an initial orthogonal item is presented prior to the
            first list, so that the system does
            not start with context as an empty 0 vector.

            In the weight matrices & context vectors, the distractors' region
            is located after study item indices & before source indices.

        beta_in_play: The update_context_temp() method will always reference
            self.beta_in_play; beta_in_play changes between the
            different beta (context drift) values offered by the
            parameter set, depending on where we are in the simulation.
        """

        self.learn_while_retrieving = False

        # data we are simulating output from
        self.source_info = source_info
        self.pres_list_nos = data_mat.astype(np.int16)

        # data structure
        self.nlists = self.pres_list_nos.shape[0]
        self.listlength = self.pres_list_nos.shape[1]

        # total no. of study items presented to the subject in this session
        self.nstudy_items_presented = self.listlength * self.nlists

        # n cells for distractors in the f, c layers & weight matrices

        # DFR version has 2 d's per list + 1 at the beginning to start the sys.
        self.ndistractors = 2*self.nlists + 1

        # n cells in each subregion (here, temporal vs. source)
        self.nsources = nsources
        self.templength = self.nstudy_items_presented + self.ndistractors

        # total number of elements operating in the system,
        # including all study lists, distractors, and sources
        self.nelements = (self.nstudy_items_presented + self.ndistractors +
                          self.nsources)

        # make a list of all items ever presented to this subject & sort it
        self.all_session_items = np.reshape(self.pres_list_nos,
                                            (self.nlists*self.listlength))
        self.all_session_items_sorted = np.sort(self.all_session_items)

        # make a list of all item source codes for this subject:
        self.all_session_item_codes = np.reshape(self.source_info,
                                            (self.nlists*self.listlength))

        # set parameters to those input when instantiating the model
        self.params = params

        #########
        #
        #   Make mini LSA matrix
        #
        #########

        # Create a mini-LSA matrix with just the items presented to this Subj.
        self.exp_LSA = np.zeros(
            (self.nstudy_items_presented, self.nstudy_items_presented),
                                dtype=np.float32)

        # Get list-item LSA indices
        for row_idx, this_item in enumerate(self.all_session_items_sorted):

            # get item's index in the larger LSA matrix
            this_item_idx = this_item - 1

            for col_idx, compare_item in enumerate(self.all_session_items_sorted):
                # get ID of jth item for LSA cos theta comparison
                compare_item_idx = compare_item - 1

                # get cos theta value between this_item and compare_item
                cos_theta = LSA_mat[this_item_idx, compare_item_idx]

                # place into this session's LSA cos theta matrix
                self.exp_LSA[int(row_idx), int(col_idx)] = cos_theta

        ######
        #
        #   Continue on with rest of stuff
        #
        ######

        # beta used by update_context_temp(); more details in doc string
        self.beta_in_play = self.params['beta_enc']
        self.beta_source = self.params['beta_source']

        # init leaky accumulator vars
        self.steps = 0

        # track which study item has been presented
        self.study_item_idx = 0
        # track which list has been presented
        self.list_idx = 0

        # track where source codes are located in the feature & context vectors
        self.source_0_idx = self.templength
        self.source_1_idx = self.templength + 1
        self.source_2_idx = self.templength + 2

        # track which distractor item has been presented
        self.distractor_idx = self.nstudy_items_presented

        # list of items recalled throughout model run
        self.recalled_items = []

        #####
        #
        #   Set up the learning-weight matrices, L_CF and L_FC
        #
        #####

        # set up scalar matrix for M_CF
        self.L_CF = np.ones((self.nelements, self.nelements), dtype=np.float32)

        # NW quadrant
        self.L_CF[:self.templength, :self.templength] = self.params['L_CF_NW']
        # NE quadrant
        self.L_CF[:self.templength, self.templength:] = self.params['L_CF_NE']
        # SW quadrant
        self.L_CF[self.templength:, :self.templength] = self.params['L_CF_SW']
        # SE quadrant
        self.L_CF[self.templength:, self.templength:] = self.params['L_CF_SE']

        # set up scalar matrix for M_FC
        self.L_FC = np.ones((self.nelements, self.nelements), dtype=np.float32)

        # NW quadrant
        self.L_FC[:self.templength, :self.templength] = self.params['L_FC_NW']
        # NE quadrant
        self.L_FC[:self.templength, self.templength:] = self.params['L_FC_NE']
        # SW quadrant
        self.L_FC[self.templength:, :self.templength] = self.params['L_FC_SW']
        # SE quadrant
        self.L_FC[self.templength:, self.templength:] = self.params['L_FC_SE']

        ##########
        #
        #   Set up & initialize weight matrices
        #
        ##########

        self.M_FC = np.identity(
            self.nelements, dtype=np.float32) * self.params['scale_fc']
        self.M_CF = np.identity(
            self.nelements, dtype=np.float32) * self.params['scale_cf']

        # set up / initialize feature and context layers
        self.c_net = np.zeros((self.nelements, 1), dtype=np.float32)
        self.c_old = np.zeros((self.nelements, 1), dtype=np.float32)
        self.c_in_normed = np.zeros_like(self.c_net)
        self.f_net = np.zeros((self.nelements, 1), dtype=np.float32)

        # set up & track leaky accumulator & recall vars
        self.x_thresh_full = np.ones(
            self.nlists * self.listlength, dtype=np.float32)
        self.n_prior_recalls = np.zeros(
            [self.nstudy_items_presented, 1], dtype=np.float32)
        self.nitems_in_race = (self.listlength *
                               self.params['nlists_for_accumulator'])

        # track the list items that have been presented
        self.lists_presented = []

    def create_semantic_structure(self):
        """Layer semantic structure onto M_CF (and M_FC, if s_fc is nonzero)

        Dimensions of the LSA matrix for this subject are nitems x nitems.

        To get item indices, we will subtract 1 from the item ID, since
        item IDs begin at 1, not 0.
        """

        # scale the LSA values by s_cf, per Lohnas et al., 2015
        cf_exp_LSA = self.exp_LSA * self.params['s_cf']

        # add the scaled LSA cos theta values to NW quadrant of the M_CF matrix
        self.M_CF[:self.nstudy_items_presented, :self.nstudy_items_presented] \
            += cf_exp_LSA

        # scale the LSA values by s_fc, per Healey et al., 2016
        fc_exp_LSA = self.exp_LSA * self.params['s_fc']

        # add the scaled LSA cos theta values to NW quadrant of the M_FC matrix
        self.M_FC[:self.nstudy_items_presented, :self.nstudy_items_presented] \
            += fc_exp_LSA

    ################################################
    #
    #   Functions defining the recall process
    #
    ################################################

    def leaky_accumulator(self, float [:] in_act,
                                          float [:] x_thresholds, int ncycles):
        """

        :param in_act: Top listlength * 4 item activations
        :param noise_vec: noise values for the accumulator.  These are
            calculated outside this function to save some runtime.
        :param x_thresholds: Threshold each item in the race must reach
            to be recalled. Starts at 1.0 for each item but can increase
            after an item is recalled in order to prevent repetitions.
        :return: Method returns index of the item that won the race,
            the time that elapsed for this item during the process of
            running the accumulator, and the final state of item activations,
            which although not strictly necessary, can be helpful for
            debugging.

        To prevent items from repeating, specify appropriate omega & alpha
        parameter values in the params dictionary.

        To facilitate testing changes to this model, you can comment out
        the noise_vec where it is added to x_s in the while loop below.
        """
        cdef float lamb, kappa, dt_tau, sq_dt_tau, eta
        lamb = self.params['lamb']
        kappa = self.params['kappa']
        dt_tau = self.params['dt_tau']
        sq_dt_tau = self.params['sq_dt_tau']
        dt = self.params['dt']
        eta = self.params['eta']

        # get max number of accumulator cycles for this run of the accumulator
        cdef int nitems_in_race
        # ncycles = noise_vec.shape[1]
        nitems_in_race  = in_act.shape[0]

        # track whether an item has passed the threshold
        item_has_crossed = 0

        # initialize x_s vector with one 0 per accumulating item
        cdef float [:] x_s
        x_s = np.zeros(nitems_in_race, dtype=np.float32)

        # init counter at 1 instead of 0, for consistency with
        # Lynn & Sean's MATLAB code
        cdef int cycle_counter
        cycle_counter = 1

        cdef float sum_x_s
        cdef int k, i, j, nwinners, index_counter

        # init vector to hold 0 or 1 status of an item's having won (winners)
        # and shorter vec with the indices of just the winners (winner_vec),
        # from which we can select a random value if we need to
        cdef float [:] winners
        winners = np.zeros(nitems_in_race, dtype=np.float32)

        ## declare winner_vec
        cdef float [:] winner_vec

        # initialize list to track items that won the race
        rec_indices = []

        # declare noise vec
        cdef double [:] noise_vec

        # if no items were sufficiently activated to go anywhere,
        # do not bother running the leaky accumulator.
        if np.sum(in_act) == 0:
            #print("Zero item activations")
            out_statement = (None, self.params['rec_time_limit'], x_s)
        else:
            while cycle_counter < ncycles and not item_has_crossed:

                # we don't scale by eta the way Lynn & Sean do, because
                # np.random.normal has already scaled by eta for us.
                # Still have to scale by sq_dt_tau, though.

                # get sum of x_s before we change the values
                sum_x_s = 0.0
                for k in range (len(x_s)):
                    sum_x_s += x_s[k] * lamb

                # draw an nitems_in_race-long vector
                noise_vec = np.random.normal(0, eta, size=nitems_in_race)

                # update each value in x_s
                for j in range(len(x_s)):
                    x_s[j] += ((in_act[j] - kappa * x_s[j] - (
                    sum_x_s - x_s[j] * lamb)) * dt_tau
                               + noise_vec[j] * sq_dt_tau)

                # make sure no values are < 0.0
                # and declare if any value has crossed its threshold
                for j in range(len(x_s)):
                    if x_s[j] < 0.0:  # if item act is <= 0.0, set to 0.0
                        x_s[j] = 0.0

                    # if item has crossed, mark in winners
                    if x_s[j] >= x_thresholds[j]:
                        item_has_crossed = 1  # and declare the race over
                        winners[j] = 1.0

                cycle_counter += 1

            # print(np.array(x_thresholds))

            # calculate elapsed time:
            sub_time = cycle_counter * dt

            # get / print the items that won the race
            nwinners = 0

            # the case of exactly one winner is handled here,
            # where winner_index = j
            for j in range(len(winners)):
                if winners[j] == 1:
                    nwinners += 1
                    winner_index = j

            # if no winners, set winner_index to None
            if nwinners == 0:
                winner_index = None

            # if more than one winner, select one at random
            elif nwinners > 1:

                # initialize winner_vec to nwinners-sized vec of 0's
                winner_vec = np.zeros(nwinners, dtype=np.float32)

                # get the IDs of all the winners & put in a vec
                index_counter = 0
                for j in range(len(winners)):
                    if winners[j] == 1.0:
                        winner_vec[index_counter] = j
                        index_counter += 1

                # select random index to pull item from the winner_vec
                srand(time.time())
                rand_idx = rand() % nwinners
                winner_index = int(winner_vec[rand_idx])

            out_statement = (winner_index, sub_time, x_s)

        return out_statement

    ####################
    #
    #   Initialize and run a recall session
    #
    ####################


    def recall_session(self):
        """Simulate a recall portion of an experiment, following a list
        presentation. """

        time_passed = 0
        rec_time_limit = self.params['rec_time_limit']

        nlists_for_accumulator = self.params['nlists_for_accumulator']

        # set how many items get entered into the leaky accumulator process
        nitems_in_race = self.listlength * nlists_for_accumulator
        nitems_in_session = self.listlength * self.nlists

        # initialize list to store recalled items
        recalled_items = []
        RTs = []
        times_since_start = []

        # track & limit None responses to prevent a param vector
        # that yields a long series of None's from eating up runtime
        num_of_nones = 0

        # number of items allowed to recall beyond list length.
        num_extras = 3

        # run a recall session for the amount of time in rec_time_limit,
        # or until person recalls >= list length + num_extras,
        # or until person repeatedly recalls a "None" item too many times
        while ((time_passed < rec_time_limit)
               and (len(recalled_items) <= self.listlength + num_extras)
               and (num_of_nones <= self.listlength + num_extras)):

            # get item activations to input to the accumulator
            f_in = np.dot(self.M_CF, self.c_net)

            # sort f_in so we can get the ll * 4 items
            # with the highest activations
            sorted_indices = np.argsort(np.squeeze(f_in[:nitems_in_session]).T)
            sorted_activations = np.sort(f_in[:nitems_in_session].T)

            # get the top-40 activations, and the indices corresponding
            # to their position in the full list of presented items.
            # we need this second value to recover the item's ID later.
            in_activations = sorted_activations[0][
                             (nitems_in_session - nitems_in_race):]
            in_indices = sorted_indices[(nitems_in_session - nitems_in_race):]

            # determine max cycles for the accumulator.
            max_cycles = np.ceil(
                (rec_time_limit - time_passed) / self.params['dt'])

            # Based on model equations, max_cycles should never get down to 0;
            # generally, the accumulator should always run out of time first.
            # if max_cycles == 0:
            #    raise ValueError("max_cycles reached 0!")

            # initialize the x_threshold vector
            x_thresh = self.x_thresh_full[in_indices]

            # get the winner of the leaky accumulator, its reaction time,
            # and this race's activation values.
            # x_n isn't strictly necessary, but can be helpful for debugging.

            winner_accum_idx, this_RT, x_n = self.leaky_accumulator(
                in_activations, x_thresh, max_cycles)

            # increment time counter
            time_passed += this_RT

            # If an item was retrieved, recover the item info corresponding
            # to the activation value index retrieved by the accumulator
            if winner_accum_idx is not None:

                # recover item's index from the original pool of item indices
                winner_sorted_idx = in_indices[winner_accum_idx]

                #-------------- Lohnas et al. 2015 recall rule ------------------#
                # multiply all thresholds by the updated value
                self.x_thresh_full = np.add(np.multiply(
                    np.subtract(self.x_thresh_full, 1), self.params['alpha']), 1)

                # then just reset the new item threshold
                self.x_thresh_full[winner_sorted_idx] = 1 + self.params['omega']

                #----------------------------------------------------------------#

                # get original item ID for this item
                winner_ID = np.sort(self.all_session_items)[winner_sorted_idx]

                ##########
                #
                #   Present item & then update system,
                #   as per regular article equations
                #
                ##########

                # reinstantiate this item
                self.present_item(winner_sorted_idx)

            # if no item was retrieved, instantiate a zero-vector
            # in the feature layer
            else:
                # track number of "None"s so we can stop the recall session
                # if we have drawn up blank more than ll + 3 times or so.
                num_of_nones += 1
                self.f_net = np.zeros([1, self.nelements])

            ##########
            #
            #   Whether or not the item is reported for recall,
            #   the item will still update the current context, as below.
            #
            ##########

            self.beta_in_play = self.params['beta_rec']
            self.update_context_temp()
            if self.nsources > 0:
                self.update_context_source()

            ############
            #
            #   See if current context is similar enough to the old context;
            #   If not, censor the item
            #
            ############

            # get similarity between c_old and the c retrieved by the item.
            c_similarity = np.dot(self.c_old.T, self.c_in_normed)

            # if sim threshold is passed,
            if (winner_accum_idx is not None) and (
                        c_similarity >= self.params['c_thresh']):

                # store item ID, RT, and time since start of rec session
                recalled_items.append(winner_ID)
                RTs.append(this_RT)
                times_since_start.append(time_passed)

                # Update the item's recall threshold & its prior recalls count
                # self.x_thresh_full[winner_sorted_idx] = (
                #     1 + self.params['omega'] * (
                #         self.params['alpha'] ** self.n_prior_recalls[
                #             winner_sorted_idx]))
                # self.n_prior_recalls[winner_sorted_idx] += 1

                # if learning is enabled during recall process,
                # update the matrices
                if self.learn_while_retrieving:

                    # Update M_FC
                    M_FC_exp = np.dot(self.c_net, self.f_net)
                    self.M_FC += np.multiply(M_FC_exp, self.L_FC)

                    # Update M_CF
                    M_CF_exp = np.dot(self.f_net.T, self.c_net.T)
                    self.M_CF += np.multiply(M_CF_exp, self.L_CF) # * prim_vec[i]

                    raise ValueError("Shouldn't be using this method just yet.")


            else:
                continue

        # update counter of what list we're on
        self.list_idx += 1

        return recalled_items, RTs, times_since_start

    def present_item(self, item_idx):
        """Set the f layer to a row vector of 0's with a 1 in the
        presented item location.  The model code will arrange this as a column
        vector where appropriate."""

        # init feature layer vector
        self.f_net = np.zeros([1, self.nelements], dtype=np.float32)

        # code a 1 in the temporal region in that item's index
        self.f_net[0][item_idx] = 1

        ##########
        #
        #   Code sources, but only if item is not a distractor.
        #
        ##########

        # if item is not a distractor:
        if item_idx != self.distractor_idx:

            # get loc of this item's source information
            source_idx, = np.where(
                self.all_session_items == self.all_session_items_sorted[item_idx])[0]

            # get item's source info
            item_source_code = self.all_session_item_codes[source_idx]

            # if there is just one source cell:
            if self.nsources == 1:
                # positive
                if item_source_code == 1.0:
                    self.f_net[0][self.source_0_idx] = 1
                # neutral
                elif item_source_code == 0.0:
                    self.f_net[0][self.source_0_idx] = 0
                # negative
                elif item_source_code == -1.0:
                    self.f_net[0][self.source_0_idx] = 1

            # if there are two source cells,
            elif self.nsources == 2:
                # positive
                if item_source_code == 1.0:
                    self.f_net[0][self.source_0_idx] = 0
                    self.f_net[0][self.source_1_idx] = 1
                # neutral
                elif item_source_code == 0.0:
                    self.f_net[0][self.source_0_idx] = 0
                    self.f_net[0][self.source_1_idx] = 0
                # negative
                elif item_source_code == -1.0:
                    self.f_net[0][self.source_0_idx] = 1
                    self.f_net[0][self.source_1_idx] = 0
                # other
                else:
                    raise ValueError("Something wrong with source encoding")

        # if item is a distractor, set source code to neutral
        else:
            # if there is just one source cell:
            if self.nsources == 1:
                self.f_net[0][self.source_0_idx] = 0
            # if two source cells,
            elif self.nsources == 2:
                self.f_net[0][self.source_0_idx] = 0
                self.f_net[0][self.source_1_idx] = 0


    def update_context_temp(self):
        """Updates the temporal region of the context vector.
        This includes all presented items, distractors, and the orthogonal
        initial item."""

        # get copy of the old c_net vector prior to updating it
        self.c_old = self.c_net.copy()

        # get input to the new context vector
        net_cin = np.dot(self.M_FC, self.f_net.T)

        # get nelements in temporal subregion
        nelements_temp = self.nstudy_items_presented + 2*self.nlists + 1

        # get region of context that includes all items presented
        # (items, distractors, & orthogonal initial item)
        cin_temp = net_cin[:nelements_temp]

        # norm the temporal region of the c_in vector
        cin_normed = norm_vec(cin_temp)
        self.c_in_normed[:nelements_temp] = cin_normed

        # grab the current values in the temp region of the network c vector
        net_c_temp = self.c_net[:nelements_temp]

        # get updated values for the temp region of the network c vector
        ctemp_updated = advance_context(
            cin_normed, net_c_temp, self.beta_in_play)

        # incorporate updated temporal region of c into the network's c vector
        self.c_net[:nelements_temp] = ctemp_updated

    def update_context_source(self):
        """Updates the source region of the context vector."""

        # we don't make another copy of c_old b/c it has already been done
        # in the c_temp update method
        # self.c_old = self.c_net.copy()

        # get source input to the source region of the new context vector
        net_cin = np.dot(self.M_FC, self.f_net.T)

        # get source region of context
        cin_source = net_cin[self.templength:self.templength + self.nsources]

        # norm the source region of the c_in vector
        cin_normed = norm_vec(cin_source)

        if self.nsources == 2:
            self.c_in_normed[self.source_0_idx] = cin_normed[0]
            self.c_in_normed[self.source_1_idx] = cin_normed[1]
        elif self.nsources == 1:
            self.c_in_normed[self.source_0_idx] = cin_normed[0]

        # get current vals in the source region of the network c vector
        net_c_source = self.c_net[self.templength:self.templength + self.nsources]

        # get updated source region of the network c vector
        csource_updated = advance_context(
            cin_normed, net_c_source, self.beta_source)

        # incorporate updated temporal region of c into the network's c vector
        if self.nsources == 2:
            self.c_net[self.source_0_idx] = csource_updated[0]
            self.c_net[self.source_1_idx] = csource_updated[1]
        elif self.nsources == 1:
            self.c_net[self.source_0_idx] = csource_updated[0]

    def present_list(self):
        """
        Presents a single list of items.

        Update context using post-recall beta weight if distractor comes
        between lists.  Use beta_enc if distractor is the first item
        in the system; this item serves to init. context to non-zero values.

        Subjects do not learn the distractor, so we do not update
        the weight matrices following it.

        :return:
        """

        # present distractor prior to this list
        self.present_item(self.distractor_idx)

        # if this is a between-list distractor,
        if self.list_idx > 0:
            self.beta_in_play = self.params['beta_rec_post']

        # else if this is the orthogonal item that starts up the system,
        elif self.list_idx == 0:
            self.beta_in_play = 1.0

        # update context regions
        self.update_context_temp()
        if self.nsources > 0:
            self.update_context_source()

        # update distractor location for the next list
        self.distractor_idx += 1

        # calculate a vector of primacy gradients (ahead of presenting items)
        prim_vec = (self.params['phi_s'] * np.exp(-self.params['phi_d']
                             * np.asarray(range(self.listlength)))
                      + np.ones(self.listlength))

        # get presentation indices for this particular list:
        thislist_pattern = self.pres_list_nos[self.list_idx]
        thislist_pres_indices = np.searchsorted(
            self.all_session_items_sorted, thislist_pattern)

        # for each item in the current list,
        for i in range(self.listlength):

            #present the item at its appropriate index
            presentation_idx = thislist_pres_indices[i]
            self.present_item(presentation_idx)

            # update the context layer (for now, just the temp region)
            self.beta_in_play = self.params['beta_enc']
            self.update_context_temp()
            if self.nsources > 0:
                self.update_context_source()

            # Update the weight matrices

            # Update M_FC
            M_FC_exp = np.dot(self.c_net, self.f_net)
            self.M_FC += np.multiply(M_FC_exp, self.L_FC)

            # Update M_CF
            M_CF_exp = np.dot(self.f_net.T, self.c_net.T)
            self.M_CF += np.multiply(M_CF_exp, self.L_CF) * prim_vec[i]

            # Update location of study item index
            self.study_item_idx += 1

        ##########
        #
        #   present distractor at the end of this list,
        #   prior to the recall session
        #
        ##########

        # present current distractor index
        self.present_item(self.distractor_idx)

        # for now, use beta post; but, we can later allow this to be its
        # own parameter, if we want.
        self.beta_in_play = self.params['beta_distract']

        self.update_context_temp()
        if self.nsources > 0:
            self.update_context_source()

        # update distractor location
        self.distractor_idx += 1


def separate_files(data_path, source_info_path, subj_id_path):
    """If data is in one big file, separate out the data into sheets, by subject.

    :param data_path: If using this method, data_path should refer directly
        to a single data file containing the consolidated data across all
        subjects.
    :return: a list of data matrices, separated out by individual subjects.

    """

    # will contain stimulus matrices presented to each subject
    Ss_data = []
    source_subj_sheets = []

    # for test subject
    data_pres_list_nos = np.loadtxt(data_path, delimiter=',')
    # load in source item information for this pt
    source_mat = np.loadtxt(source_info_path, delimiter=',')

    # get list of unique subject IDs

    # use this if dividing a multiple-session subject into sessions
    subj_id_map = np.loadtxt(subj_id_path)
    unique_subj_ids = np.unique(subj_id_map)

    # Get locations where each Subj's data starts & stops.
    new_subj_locs = np.unique(
        np.searchsorted(subj_id_map, subj_id_map))

    # Separate data into sets of lists presented to each subject
    for i in range(new_subj_locs.shape[0]):

        # for all but the last list, get the lists that were presented
        # between the first time that subj ID occurs and when the next ID occurs
        if i < new_subj_locs.shape[0] - 1:
            start_lists = new_subj_locs[i]
            end_lists = new_subj_locs[i + 1]

        # once you have reached the last subj, get all the lists from where
        # that ID first occurs until the final list in the dataset
        else:
            start_lists = new_subj_locs[i]
            end_lists = data_pres_list_nos.shape[0]

        # append subject's sheet
        Ss_data.append(data_pres_list_nos[start_lists:end_lists, :])
        source_subj_sheets.append(source_mat[start_lists:end_lists, :])

    return Ss_data, unique_subj_ids, source_subj_sheets


def run_CMR2_singleSubj(data_mat, LSA_mat, params, source_info, nsource_cells):

    """Run CMR2 for an individual subject / data sheet

    Uses the recall_session() method for CMR2"""

    # init. lists to store CMR2 output
    resp_values = []
    RT_values = []
    time_values = []

    # create CMR2 object
    this_CMR = CMR2(
        params=params, nsources=nsource_cells, source_info=source_info,
        LSA_mat=LSA_mat, data_mat=data_mat)

    # layer LSA cos theta values onto the weight matrices
    this_CMR.create_semantic_structure()

    # Run CMR2 for all lists after the 0th list
    for i in range(len(this_CMR.pres_list_nos)):
        # present new list
        this_CMR.present_list()

        # recall session
        rec_items_i, RTs_list_i, times_from_start_i \
            = this_CMR.recall_session()

        # append recall responses & times
        resp_values.append(rec_items_i)
        RT_values.append(RTs_list_i)
        time_values.append(times_from_start_i)

    return resp_values, RT_values, time_values


def run_CMR2(LSA_path, LSA_mat, data_path, params, sep_files,
             filename_stem="", source_info_path=".",
             nsource_cells=0, subj_id_path="."):
    """Run CMR2 for all subjects

    time_values = time for each item since beginning of recall session

    For later zero-padding the output, we will get list length from the
    width of presented-items matrix. This assumes equal list lengths
    across Ss and sessions, unless you are inputting each session
    individually as its own matrix, in which case, list length will
    update accordingly.

    If all Subjects' data are combined into one big file, as in some files
    from prior CMR2 papers, then divide data into individual sheets per subj.

    If you want to simulate CMR2 for individual sessions, then you can
    feed in individual session sheets at a time, rather than full subject
    presented-item sheets.
    """

    #print("Entered CMR2 code at: " + str(time.time()))

    now_test = time.time()

    # set diagonals of LSA matrix to 0.0
    np.fill_diagonal(LSA_mat, 0)

    # init. lists to store CMR2 output
    resp_vals_allSs = []
    RT_vals_allSs = []
    time_vals_allSs = []

    # Simulate each subject's responses.
    if not sep_files:

        # divide up the data
        subj_presented_data, unique_subj_ids, subj_source_sheets = separate_files(
            data_path, source_info_path, subj_id_path)

        # get list length
        listlength = subj_presented_data[0].shape[1]

        # for each subject's data matrix,
        for m, pres_sheet in enumerate(subj_presented_data):

            #if m > 0:
            #    break

            subj_id = unique_subj_ids[m]
            # print('Subject ID is: ' + str(subj_id))

            resp_Subj, RT_Subj, time_Subj = run_CMR2_singleSubj(
                pres_sheet, LSA_mat,
                params=params, source_info=subj_source_sheets[m],
                nsource_cells=nsource_cells)

            resp_vals_allSs.append(resp_Subj)
            RT_vals_allSs.append(RT_Subj)
            time_vals_allSs.append(time_Subj)

    # If files are separate, then read in each file individually
    else:

        # get all the individual data file paths
        indiv_file_paths = glob(data_path + filename_stem + "*.mat")

        # read in the data for each path & stick it in a list of data matrices
        for file_path in indiv_file_paths:

            data_file = scipy.io.loadmat(
                file_path, squeeze_me=True, struct_as_record=False)  # get data
            data_mat = data_file['data'].pres_itemnos  # get presented items

            resp_Subj, RT_Subj, time_Subj = run_CMR2_singleSubj(
                data_mat=data_mat, LSA_mat=LSA_mat,
                params=params, source_info=[])

            resp_vals_allSs.append(resp_Subj)
            RT_vals_allSs.append(RT_Subj)
            time_vals_allSs.append(time_Subj)

        # for later zero-padding the output, get list length from one file.
        data_file = scipy.io.loadmat(indiv_file_paths[0], squeeze_me=True,
                                     struct_as_record=False)
        data_mat = data_file['data'].pres_itemnos

        listlength = data_mat.shape[1]


    ##############
    #
    #   Zero-pad the output
    #
    ##############

    # If more than one subject, reshape the output into a single,
    # consolidated sheet across all Ss
    if len(resp_vals_allSs) > 0:
        resp_values = [item for submat in resp_vals_allSs for item in submat]
        RT_values = [item for submat in RT_vals_allSs for item in submat]
        time_values = [item for submat in time_vals_allSs for item in submat]
    else:
        resp_values = resp_vals_allSs
        RT_values = RT_vals_allSs
        time_values = time_vals_allSs

    # set max width for zero-padded response matrix
    maxlen = listlength * 2

    nlists = len(resp_values)

    # init. zero matrices of desired shape
    resp_mat  = np.zeros((nlists, maxlen))
    RTs_mat   = np.zeros((nlists, maxlen))
    times_mat = np.zeros((nlists, maxlen))

    # place output in from the left
    for row_idx, row in enumerate(resp_values):

        resp_mat[row_idx][:len(row)]  = resp_values[row_idx]
        RTs_mat[row_idx][:len(row)]   = RT_values[row_idx]
        times_mat[row_idx][:len(row)] = time_values[row_idx]

    #print('Analyses complete.')

    # print("CMR Time: " + str(time.time() - now_test))

    return resp_mat, times_mat


def main():
    """Main method"""

    # set desired parameters. Example below is for Kahana et al. (2002)

    # Polyn et al. 2009 parameters
    polyn = {

        'beta_enc': 0.77,
        'beta_rec': 0.51,
        'beta_source': 0.59,
        'gamma_fc': 0.898,
        'gamma_cf': 0.129,
        'scale_fc': 1 - 0.898,
        'scale_cf': 1 - 1.000,

        'phi_s': 1.07,
        'phi_d': 0.98,
        'kappa': 0.111,

        'eta': 0.380,
        's_cf': 1.08,   # 2.78
        's_fc': 0.0,
        'beta_rec_post': 0.980,
        'omega': 15.907,
        'alpha': .765,
        'c_thresh': 0.0001,
        'dt': 10.0,

        'lamb': 0.338,
        'rec_time_limit': 75000,

        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,

        'nlists_for_accumulator': 2,

        'L_CF_NW': 1.0,             # NW quadrant
        'L_CF_NE': 0.129,           # NE quadrant
        'L_CF_SW': 0.0,             # SW quadrant
        'L_CF_SE': 0.0,             # SE quadrant

        'L_FC_NW': 0.898,           # NW quadrant
        'L_FC_NE': 0.0,             # NE quadrant
        'L_FC_SW': 0.898,           # SW quadrant
        'L_FC_SE': 0.0              # SE quadrant
    }

    # format printing nicely
    np.set_printoptions(precision=5)

    # Set LSA and data paths -- K02 data
    on_rhino = True
    if on_rhino:
        LSA_path = 'w2v.txt'
        data_path = 'pres_nos_LTP228.txt'
    else:
        LSA_path = 'polyn_lsa.txt'
        data_path = 'polyn_pres_nos.txt'

    # read in LSA matrix from text file (scipy.io.loadmat has large overhead)
    LSA_mat = np.loadtxt(LSA_path, delimiter=' ')

    source_path = 'eval_codes_LTP228.txt'

    nsource_cells = 1

    start_time = time.time()
    rec_nos, times = run_CMR2(LSA_path=LSA_path, LSA_mat=LSA_mat,
                              data_path=data_path,
                              params=polyn, sep_files=False,
                              source_info_path=source_path,
                              nsource_cells=nsource_cells)

    print("End of time: " + str(time.time() - start_time))

    # save CMR2 results
    np.savetxt('resp_source_LTP228.txt', np.asmatrix(rec_nos), delimiter=',', fmt='%i')
    np.savetxt('times_source_LTP228.txt', np.asmatrix(times), delimiter=',', fmt='%i')


if __name__ == "__main__": main()
