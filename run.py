
import builtins
import cProfile
import pstats
import torch
from ts_torch import main_torch_mini
import torch.multiprocessing as mp
from anp import anpp


try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile


def main():
    model_predictor_list, features_structure = main_torch_mini.example2()
    main_torch_mini.run_weekend(i=12, use_macro=False, use_swa=False, model_predictor_list=model_predictor_list,
                                features_structure=features_structure, country='kr')


def main2(model):
    from ts_torch.torch_util_mini import np_ify
    TRAINING_ITERATIONS = 1000 #@param {type:"number"}
    MAX_CONTEXT_POINTS = 250 #@param {type:"number"}
    PLOT_AFTER = 1000 #@param {type:"number"}
    batch_size = 16
    base_i = 100
    dataset = anpp.TimeSeries(batch_size=batch_size, max_num_context=MAX_CONTEXT_POINTS, predict_length=120)
    base_y = dataset.get_timeseries('kospi')
    dataset.generate_set(base_y)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for it in range(TRAINING_ITERATIONS):
        data_train = dataset.generate(base_i, seq_len=1, is_train=True)
        # data_train = dataset_train.generate_curves()
        # data_train = to_device(data_train, 'cuda:0')
        model.train()
        # Define the loss
        ((c_x, c_y), t_x), t_y = data_train.query,  data_train.target_y
        train_query = ((c_x[0], c_y[0]), t_x[0])
        train_target_y = t_y[0]
        _, _, log_prob, _, _, loss = model(train_query, train_target_y)
        # _, _, log_prob, _, loss = model(data_train.query,  data_train.target_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot the predictions in `PLOT_AFTER` intervals
        if it % PLOT_AFTER == 0:
            for ii, date_i in enumerate([base_i - 20, base_i, base_i + 20]):
                data_test = dataset.generate(date_i, seq_len=1, is_train=False)
                # data_test = dataset_test.generate_curves()
                # data_test = to_device(data_test, 'cuda:0')
                model.eval()
                with torch.set_grad_enabled(False):
                    ((c_x, c_y), t_x), t_y = data_test.query,  data_test.target_y
                    test_query = ((c_x[0], c_y[0]), t_x[0])
                    test_target_y = t_y[0]
                    _, _, log_prob, _, _, loss = model(test_query, test_target_y)
                    # _, _, log_prob, _, loss = model(data_train.query, data_train.target_y)

                    # Get the predicted mean and variance at the target points for the testing set
                    mu, sigma, _, _, _, _ = model(test_query)
                    # mu, sigma, _, _, _ = model(data_test.query)
                loss_value, pred_y, std_y, target_y, whole_query = loss, mu, sigma, test_target_y, test_query

                (context_x, context_y), target_x = whole_query
                print('Iteration: {}, loss: {}'.format(it, np_ify(loss_value)))

                # Plot the prediction and the context
                anpp.plot_functions(it + ii, np_ify(target_x), np_ify(target_y), np_ify(context_x), np_ify(context_y), np_ify(pred_y), np_ify(std_y))

        # Plot the predictions in `PLOT_AFTER` intervals
        if it > 0 and it % 10000 == 0:
            for ii in range(-base_i, dataset.max_len - base_i - 1):
                data_test = dataset.generate(date_i + ii, seq_len=1, is_train=False)
                # data_test = dataset_test.generate_curves()
                # data_test = to_device(data_test, 'cuda:0')
                model.eval()
                with torch.set_grad_enabled(False):
                    ((c_x, c_y), t_x), t_y = data_test.query,  data_test.target_y
                    test_query = ((c_x[0], c_y[0]), t_x[0])
                    test_target_y = t_y[0]
                    _, _, log_prob, _, _, loss = model(test_query, test_target_y)
                    # _, _, log_prob, _, loss = model(data_train.query, data_train.target_y)

                    # Get the predicted mean and variance at the target points for the testing set
                    mu, sigma, _, _, _, _ = model(test_query)
                    # mu, sigma, _, _, _ = model(data_test.query)
                loss_value, pred_y, std_y, target_y, whole_query = loss, mu, sigma, test_target_y, test_query

                (context_x, context_y), target_x = whole_query
                print('Iteration: {}, loss: {}'.format(it, np_ify(loss_value)))

                # Plot the prediction and the context
                anpp.plot_functions(it + ii, np_ify(target_x), np_ify(target_y), np_ify(context_x), np_ify(context_y), np_ify(pred_y), np_ify(std_y))






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

# if __name__ == '__main__':
#     num_processes = 4
#
#     model = anpp.LatentModel(128)  # .cuda()
#     model.train()
#     # NOTE: this is required for the ``fork`` method to work
#     model.share_memory()
#     processes = []
#     for rank in range(num_processes):
#         p = mp.Process(target=main2, args=(model,))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()