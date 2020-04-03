
import cProfile
import pstats
from ts_torch import main_torch_mini

def main():
    model_predictor_list, features_structure = main_torch_mini.example2()
    main_torch_mini.run_weekend(i=5, use_macro=False, use_swa=False, model_predictor_list=model_predictor_list,
                                features_structure=features_structure, country='kr')


if __name__ == '__main__':
    main()
    # prof = cProfile.Profile()
    # prof.runctx("main_torch_mini.run_weekend(i=4, use_macro=False, use_swa=False, model_predictor_list=model_predictor_list, features_structure=features_structure, country='kr')", globals(), locals())
    #
    # stats = pstats.Stats(prof)
    # stats.sort_stats('cumtime')
    # stats.dump_stats('output.prof')
    #
    # stream = open('output.txt', 'w')
    # stats = pstats.Stats('output.prof', stream=stream)
    # stats.sort_stats('cumtime')
    # stats.print_stats()
    #
