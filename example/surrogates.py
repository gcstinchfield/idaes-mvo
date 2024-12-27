import os
import numpy as np

from process_family.type.surrogates import SurrogatesProcessFamily
from process_family.utils.parameters.surrogates import Parameters

if __name__=="__main__":

    # initialize parameters
    params = Parameters("case study")
    params.make_results_dir("discretized")

    # grab paths to surrogates
    cwd = os.getcwd()
    classification_surrogate = os.path.join(cwd, f"surrogates/classification-nn")
    regression_surrogate = os.path.join(cwd, f"surrogates/regression-nn")

    # add surrogates we want
    params.add_surrogates(classification_type = "nn",
                          classification_path = classification_surrogate,
                          regression_type = "nn",
                          regression_path = regression_surrogate,
                          classification_threshold = 0.5)
    
    # hack in the output scaling
    params.regression_scaling["offset_outputs"] = np.array([0], dtype="float32")
    params.regression_scaling["factor_outputs"] = np.array([1], dtype="float32")
    
    # create the surrogates based process family
    spfd = SurrogatesProcessFamily(params)

    # build & solve surrogates formulation
    spfd.build_model(params.labels_for_common_unit_module_designs)
    spfd.solve_model()

    # plot results, store in plot_pathstring
    plot_pathstring = os.path.join(params.results_dir, "surrogates-results.png")
    spfd.plot(directory=plot_pathstring,
              process_variant_column_names = params.process_variant_column_names,
              common_module_type_columns = params.common_module_type_column_names)

    results_pathstring=os.path.join(params.results_dir,"surrogates-results.txt")
    spfd.results_summary(directory=results_pathstring)