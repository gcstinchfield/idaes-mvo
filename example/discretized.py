import os

from process_family.type.discretized import DiscretizedProcessFamily
from process_family.utils.parameters.base import Parameters

if __name__=="__main__":

    # initialize parameters
    params = Parameters("case study")
    params.make_results_dir("discretized")
    # params.add_num_common_unit_type_designs({"ABSDIAM": 4,"STRDIAM": 3})

    # initializes all sets
    dpfd = DiscretizedProcessFamily(params)
    
    # build & solve model
    dpfd.build_model()
    dpfd.solve_model()

    # save the first stage solution
    # dpfd.save_var_results(os.path.join(params.results_dir, "var_results.csv"))

    # due to how we plot, non-gridded process variants shouldn't have ticks set.
    if system_name=="water-desalination":
        set_ticks = False
    else:
        set_ticks = True
    
    dpfd.plot(directory = os.path.join(params.results_dir, "discretized-results.png"),
              set_ticks = set_ticks,
              process_variant_column_names = params.process_variant_column_names,
              common_module_type_columns = params.common_module_type_column_names)

    # create results summary, store in results_pathstring
    dpfd.results_summary(directory = \
                         os.path.join(params.results_dir, "discretized-results.txt"))

    # economies of numbers analysis
    dpfd.eon_summary(directory = os.path.join(params.results_dir, "discretized-eon-stats.txt"),
                    alpha = 0.8,
                    DF_max = 0.7,
                    unit_module_capex_columns = params.unit_module_capex_columns)