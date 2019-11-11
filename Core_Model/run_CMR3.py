import CMR3 as CMR2
import numpy as np
import os
from glob import glob
import errno


def get_param_dict_CMR3(param_vec):
    """helper function to properly format a parameter vector"""
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
        'beta_distract': param_vec[15],
        'rec_time_limit': 75000,

        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,

        'nlists_for_accumulator': 2,

        'L_CF_NW': param_vec[3],  # NW quadrant - set constant to 1.0 in polyn
        'L_CF_NE': param_vec[16],  # NE quadrant - set to gamma_cf value
        'L_CF_SW': 0.0,  # SW quadrant - set constant to 0.0
        'L_CF_SE': 0.0,  # SE quadrant - set constant to 0.0

        'L_FC_NW': param_vec[2],  # NW quadrant - set to gamma_fc value
        'L_FC_NE': 0.0,  # NE quadrant - set constant to 0.0
        'L_FC_SW': param_vec[2],  # SW quadrant - set to gamma_fc value
        'L_FC_SE': 0.0  # SE quadrant - set constant to 0.0
    }

    return param_dict

def get_param_dict_CMR2(param_vec):
    """helper function to properly format a parameter vector"""
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
        'beta_source': 0.0,
        'beta_distract': param_vec[15],
        'rec_time_limit': 75000,

        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,

        'nlists_for_accumulator': 2,

        'L_CF_NW': param_vec[3],  # NW quadrant - set constant to 1.0 in polyn
        'L_CF_NE': 0.0,  # NE quadrant - set to gamma_cf value
        'L_CF_SW': 0.0,  # SW quadrant - set constant to 0.0
        'L_CF_SE': 0.0,  # SE quadrant - set constant to 0.0

        'L_FC_NW': param_vec[2],  # NW quadrant - set to gamma_fc value
        'L_FC_NE': 0.0,  # NE quadrant - set constant to 0.0
        'L_FC_SW': 0.0,  # SW quadrant - set to gamma_fc value
        'L_FC_SE': 0.0  # SE quadrant - set constant to 0.0
    }

    return param_dict


def main():

    ##########
    #
    #   Select which subject you want to run the model on
    #
    ##########

    subject = 'LTP393'

    ##########
    #
    #   Choose whether to use the CMR3, CMR2, or eCMR version of the model
    #
    ##########
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

    ##########
    #
    #   Define paths to needed files here
    #
    ##########

    # set path to inter-item semantic similarity matrix & read it in
    LSA_path = './HelperFiles/w2v.txt'
    LSA_mat = np.loadtxt(LSA_path, delimiter=' ')

    # Current directory and where the subjects' data files are stored
    root_path = './'
    data_root = '../Data/'

    # Where you want the model-predicted output to be saved
    output_folder = 'output_files_' + model_name + '/'
    output_directory_path = root_path + output_folder

    # Where the collection of parameter files is located
    param_file_folder = '../Params/params_CMR3/'

    # Filestem with which to save the model output
    filestem = 'model_rec_nos_'

    ##########
    #
    #   Run the model
    #
    ##########

    # read in subject's parameters
    subject_params = np.loadtxt(root_path + param_file_folder + 'xopt_' + subject + '.txt')

    # format parameters into a dictionary
    if use_CMR3 or use_eCMR:
        params_to_test = get_param_dict_CMR3(subject_params)
    elif use_CMR2:
        params_to_test = get_param_dict_CMR2(subject_params)

    # set data path where presented-item files are located
    pres_items_path = data_root + 'pres_files/pres_nos_' + subject + '.txt'

    # set path where presented items' eval codes are located
    source_code_path = data_root + 'eval_files/eval_codes_' + subject + '.txt'

    # set path where session divisions file is located
    session_div_path = './HelperFiles/division_locs_ind1.txt'

    if use_CMR3:
        # run CMR3
        model_rec_nos, times = CMR2.run_CMR2(LSA_path=LSA_path, LSA_mat=LSA_mat,
                                       data_path=pres_items_path,
                                       params=params_to_test, sep_files=False,
                                       source_info_path=source_code_path,
                                       nsource_cells=2,
                                       subj_id_path=session_div_path)
    elif use_CMR2:
        # run CMR2
        # note: CMR2 is implemented by setting beta_emot and gamma_emot parameters to 0.0,
        #       not (currently) by modifying the input number of source cells, which would have
        #       the equivalent effect.
        model_rec_nos, times = CMR2.run_CMR2(LSA_path=LSA_path, LSA_mat=LSA_mat,
                                             data_path=pres_items_path,
                                             params=params_to_test, sep_files=False,
                                             source_info_path=source_code_path,
                                             nsource_cells=2,
                                             subj_id_path=session_div_path)

    elif use_eCMR:
        # run eCMR
        # note: eCMR is implemented by setting nsource_cells equal to 1.
        model_rec_nos, times = CMR2.run_CMR2(LSA_path=LSA_path, LSA_mat=LSA_mat,
                                             data_path=pres_items_path,
                                             params=params_to_test, sep_files=False,
                                             source_info_path=source_code_path,
                                             nsource_cells=1,
                                             subj_id_path=session_div_path)

    # If the output folder doesn't exist, create it:
    if not os.path.exists(output_directory_path):
            os.mkdir(output_directory_path)

    # save the model's output to the output directory
    np.savetxt(output_directory_path + filestem + subject + '.txt',
               np.asmatrix(model_rec_nos), delimiter=',', fmt='%.0d')


if __name__ == "__main__":
    main()
