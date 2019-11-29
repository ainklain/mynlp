
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

from ts_torch import torch_util_mini as tu

def weight_scale(score, method='L_60'):
    method = method.lower()
    m_args = method.split('_')

    scale = np.zeros_like(score)
    # score값에 따라 기본 ew/mw weight * scale 해주는 값
    if m_args[0] == 'bm':
        scale[:] = 1.
    else:
        rank_ = np.argsort(-score)  # 값이 큰 순서
        if m_args[0] == 'l':
            # 상위 n_percent% 투자
            # ex) 'L_60', 'L_80', ...
            assert len(m_args[1:]) == 1
            n_percent = int(m_args[1]) / 100
            lower_bound = 0
            upper_bound = int(len(rank_) * n_percent)
            scale[rank_[lower_bound:upper_bound]] = 1.
        elif m_args[0] == 'ls':
            # 롱숏 ntile분위, 각분위 당 wgt_diff 씩 조정 배분
            # ex) 'ls_5_20', 'LS_4_10', ...
            assert len(m_args[1:]) == 2
            ntile = int(m_args[1])
            wgt_diff = int(m_args[2]) / 100
            for i in range(ntile):
                lower_bound = int(len(rank_) * (i / ntile))
                upper_bound = int(len(rank_) * ((i + 1.) / ntile))
                scale[rank_[lower_bound:upper_bound]] = (1 + wgt_diff * (ntile - 1) / 2) - wgt_diff * i
        elif m_args[0] == 'each':
            ntile = int(m_args[1])
            scale = np.zeros([len(score), ntile])
            for i in range(ntile):
                lower_bound = int(len(rank_) * (i / ntile))
                upper_bound = int(len(rank_) * ((i + 1.) / ntile))
                scale[rank_[lower_bound:upper_bound], i] = 1.

    return scale


class Performance:
    def __init__(self, configs):
        self.configs = configs
        self.label_feature = configs.label_feature
        self.pred_feature = configs.pred_feature
        self.cost_rate = configs.cost_rate

        self.adj_feature = 'nmlogy'

    def define_variables(self, t_steps, assets, f_keys=None):
        var_dict = dict(y=np.zeros([t_steps, 1]),
                        turnover=np.zeros([t_steps, 1]),
                        total_cost=np.zeros([t_steps, 1]),
                        y_w_cost=np.zeros([t_steps, 1]),
                        wgt=pd.DataFrame({'old': np.zeros_like(assets), 'new': np.zeros_like(assets)},
                                         index=assets,
                                         dtype=np.float32))  # column 0: old wgt, 1: new wgt
        if f_keys is not None:
            for key in f_keys:
                var_dict[key] = np.zeros([t_steps, 1])

        return var_dict

    def define_variables_ntile(self, t_steps, assets, n_tile, f_keys=None):
        var_dict = dict(y=np.zeros([t_steps, 1]),
                        y_each=np.zeros([t_steps, n_tile]),
                        turnover=np.zeros([t_steps, 1]),
                        total_cost=np.zeros([t_steps, 1]),
                        y_w_cost=np.zeros([t_steps, 1]),
                        wgt=pd.DataFrame({'old': np.zeros_like(assets), 'new': np.zeros_like(assets)},
                                         index=assets,
                                         dtype=np.float32))  # column 0: old wgt, 1: new wgt
        if f_keys is not None:
            for key in f_keys:
                var_dict[key] = np.zeros([t_steps, 1])
                var_dict[key + '_each'] = np.zeros([t_steps, n_tile])

        return var_dict

    # call by reference (var_dict를 파라미터로 받아서 업데이트)
    def calculate_cost(self, t, var_dict, assets, label_y):
        # nickname
        wgt_ = var_dict['wgt']

        var_dict['turnover'][t] = np.sum(np.abs(wgt_['new'] - wgt_['old']))
        var_dict['total_cost'][t] = var_dict['total_cost'][t - 1] + var_dict['turnover'][t] * self.cost_rate
        wgt_.loc[:, 'old'] = 0.0
        wgt_.loc[assets, 'old'] = ((1 + label_y) * wgt_.loc[assets, 'new']) / np.sum((1 + label_y) * wgt_.loc[assets, 'new'])
        var_dict['y_w_cost'][t] = np.sum(label_y * wgt_.loc[assets, 'new']) - var_dict['turnover'][t] * self.cost_rate

    def scale_to_wgt(self, base_wgt, scale, rate_, mc, w_method='ew'):
        port = np.unique(scale)
        if w_method == 'ew':
            scale_normalized = scale / np.sum(scale)
            scale_normalized = scale_normalized - np.mean(scale_normalized)
            new_wgt = np.array(base_wgt) + scale_normalized * rate_  # 100% 반영
            new_wgt[new_wgt < 0] = 0.
            new_wgt = new_wgt / np.sum(new_wgt)
        elif w_method == 'mw':
            mc_port = np.zeros(len(port))
            for i, p in enumerate(port):
                mc_port[i] = np.sum(mc[scale == p])
            mc_adj_value = np.min(mc_port) / mc_port

            for i, p in enumerate(port):
                scale[scale == p] = (p - np.mean(port)) * rate_ * mc_adj_value[i]

            new_wgt = np.array(base_wgt) * (1 + scale)
            new_wgt = new_wgt / np.sum(new_wgt)
        else:
            raise NotImplementedError

        return new_wgt

    def extract_portfolio(self, model, dataloader_set_t, rate_=1.):
        c = self.configs
        dataloader, features_list, all_assets_list, start_d, end_d = dataloader_set_t
        features, add_info = dataloader

        mc = np.array(add_info['mktcap'], dtype=np.float32).squeeze()
        assets = np.array(add_info['asset_list'])

        result_t = add_info['univ'].set_index('infocode').loc[assets]
        result_t['wgt'] = result_t['wgt'] / np.sum(result_t['wgt'])

        # ############ For BM ############
        result_t['bm_wgt_ew'] = 1. / len(assets)
        result_t['bm_wgt_mw'] = mc / np.sum(mc)

        # ############ For Model ############
        # prediction
        predictions = model.predict_mtl(features)
        for key in predictions.keys():
            predictions[key] = tu.np_ify(predictions[key])

        value_ = dict()
        value_[self.adj_feature] = predictions[self.adj_feature][:, 0, 0]
        value_['main'] = predictions[self.pred_feature][:, 0, 0]

        # long-short score
        scale_ls = weight_scale(value_['main'], method='ls_5_20')
        # long-only score
        scale_l_temp = weight_scale(value_['main'], method='l_60')
        scale_l = scale_l_temp * weight_scale(value_[self.adj_feature], method='l_60')

        result_t['model_wgt_ls_ew'] = scale_ls / np.sum(scale_ls)
        result_t['model_wgt_ls_mw'] = mc * scale_ls / np.sum(mc * scale_ls)

        result_t['model_wgt_l_ew'] = scale_l / np.sum(scale_l)
        result_t['model_wgt_l_mw'] = mc * scale_l / np.sum(mc * scale_l)

        # LONG SHORT
        # Equal Weight Mix
        wgt_ls_ew = self.scale_to_wgt(result_t['wgt'], scale_ls, rate_, mc, w_method='ew')
        # Market Weight Mix
        wgt_ls_mw = self.scale_to_wgt(result_t['wgt'], scale_ls, rate_, mc, w_method='mw')

        # Equal Weight Mix
        wgt_l_ew = self.scale_to_wgt(result_t['wgt'], scale_l, rate_, mc, w_method='ew')
        # Market Weight Mix
        wgt_l_mw = self.scale_to_wgt(result_t['wgt'], scale_l, rate_, mc, w_method='mw')

        result_t['fm_wgt_ls_ew'] = wgt_ls_ew
        result_t['fm_wgt_ls_mw'] = wgt_ls_mw

        result_t['fm_wgt_l_ew'] = wgt_l_ew
        result_t['fm_wgt_l_mw'] = wgt_l_mw

        return result_t

    def predict_plot_mtl_cross_section_test(self, model, dataloader_set, save_dir
                                            , file_nm='test.png'
                                            , ylog=False
                                            , ls_method='ls_5_20'
                                            , plot_all_features=True):
        # save_dir = test_out_path; file_nm = 'test_{}.png'.format(0); ylog = False; ls_method = 'ls_5_20'; plot_all_features = True

        c = self.configs

        m_args = ls_method.split('_')
        if m_args[0] == 'ls':
            n_tile = int(m_args[1])
            def_variables = self.define_variables_ntile
            kw = {'n_tile': n_tile}
        else:
            n_tile = -1
            def_variables = self.define_variables
            kw = {}

        # dataloader: (features: dict of torch.Tensor, add_infos: dict of np.array)
        dataloader, features_list, all_assets_list, start_d, end_d = dataloader_set

        t_stepsize = self.configs.k_days // self.configs.sampling_days
        t_steps = int(np.ceil(len(dataloader[0]) / t_stepsize)) + 1

        # define variables to save values
        if plot_all_features:
            model_keys = list(model.predictor.keys())
            features_for_plot = ['main'] + model_keys
        else:
            model_keys = None
            features_for_plot = ['main']

        ew_dict = dict(bm=self.define_variables(t_steps=t_steps, assets=all_assets_list),
                       model=def_variables(t_steps=t_steps, assets=all_assets_list, f_keys=model_keys, **kw))
        mw_dict = dict(bm=self.define_variables(t_steps=t_steps, assets=all_assets_list),
                       model=def_variables(t_steps=t_steps, assets=all_assets_list, f_keys=model_keys, **kw))

        # nickname
        bm_ew = ew_dict['bm']
        bm_mw = mw_dict['bm']
        model_ew = ew_dict['model']
        model_mw = mw_dict['model']

        for i, (features, add_info) in enumerate(zip(*dataloader)):
            # i=0; ein_t, din_t, dout_t, add_info = ie_list[i], od_list[i], td_list[i], add_infos[i]
            if i % t_stepsize != 0:
                continue
            t = i // t_stepsize + 1

            mc = np.array(add_info['mktcap'], dtype=np.float32).squeeze()
            label_y = np.array(add_info['next_y'])

            assets = np.array(add_info['asset_list'], dtype=np.float32)

            # ############ For BenchMark ############

            bm_ew['y'][t] = np.mean(label_y)
            bm_mw['y'][t] = np.sum(label_y * mc) / np.sum(mc)

            # cost calculation
            bm_ew['wgt'].loc[:, 'new'] = 0
            bm_ew['wgt'].loc[assets, 'new'] = 1. / len(assets)
            self.calculate_cost(t, bm_ew, assets, label_y)

            bm_mw['wgt'].loc[:, 'new'] = 0
            bm_mw['wgt'].loc[assets, 'new'] = mc / np.sum(mc)
            self.calculate_cost(t, bm_mw, assets, label_y)

            # ############ For Model ############
            # prediction
            predictions = model.predict_mtl(features)
            for key in predictions.keys():
                predictions[key] = tu.np_ify(predictions[key])
            value_ = dict()

            value_[self.adj_feature] = predictions[self.adj_feature][:, 0, 0]
            for f_ in features_for_plot:
                f_for_y = ('y' if f_ == 'main' else f_)
                if f_ == 'main':
                    value_['main'] = predictions[self.pred_feature][:, 0, 0]
                else:
                    value_[f_] = predictions[f_][:, 0, 0]

                if m_args[0] == 'ls':
                    # ntile 별 수익률
                    scale_n = weight_scale(value_[f_], method='each_{}'.format(n_tile))
                    model_ew[f_for_y + '_each'][t, :] = np.matmul(label_y, scale_n) / np.sum(scale_n, axis=0)
                    model_mw[f_for_y + '_each'][t, :] = np.matmul(label_y * mc, scale_n) / np.matmul(mc, scale_n)
                    # or np.sum((label_y * mc).reshape([-1, 1]) * scale, axis=0) / np.sum(mc.reshape(-1, 1) * scale, axis=0)

                    # pf 수익률
                    scale = weight_scale(value_[f_], method=ls_method)
                elif m_args[0] == 'l':
                    scale1 = weight_scale(value_[f_], method=ls_method)
                    scale = scale1 * weight_scale(value_[self.adj_feature], method=ls_method)
                elif m_args[0] == 'limit-LS':
                    pass

                model_ew[f_for_y][t] = np.sum(label_y * scale) / np.sum(scale)
                model_mw[f_for_y][t] = np.sum(label_y * mc * scale) / np.sum(mc * scale)

                if f_ == 'main':
                    # cost calculation
                    model_ew['wgt'].loc[:, 'new'] = 0
                    model_ew['wgt'].loc[assets, 'new'] = scale / np.sum(scale)
                    self.calculate_cost(t, model_ew, assets, label_y)

                    model_mw['wgt'].loc[:, 'new'] = 0
                    model_mw['wgt'].loc[assets, 'new'] = mc * scale / np.sum(mc * scale)
                    self.calculate_cost(t, model_mw, assets, label_y)

        for f_ in features_for_plot:
            f_for_y = ('y' if f_ == 'main' else f_)
            if m_args[0] == 'ls':
                y_arr = np.concatenate([bm_ew['y'], bm_mw['y']
                                        , model_ew[f_for_y]
                                        , model_mw[f_for_y]
                                        , model_ew[f_for_y + '_each']
                                        , model_mw[f_for_y + '_each']], axis=-1)
                data = pd.DataFrame(np.cumprod(1. + y_arr, axis=0)
                                    , columns=['bm_ew', 'bm_mw', 'model_ew', 'model_mw']
                                              + ['model_e{}'.format(i+1) for i in range(n_tile)]
                                              + ['model_m{}'.format(i+1) for i in range(n_tile)])
                data['model_ls_ew'] = np.cumprod(1. + np.mean(model_ew[f_for_y + '_each'][:, :1], axis=1) - np.mean(model_ew[f_for_y + '_each'][:, -1:], axis=1))
                data['model_ls_mw'] = np.cumprod(1. + np.mean(model_mw[f_for_y + '_each'][:, :1], axis=1) - np.mean(model_mw[f_for_y + '_each'][:, -1:], axis=1))

            else:
                y_arr = np.concatenate([bm_ew['y'], bm_mw['y'], model_ew[f_for_y], model_mw[f_for_y]], axis=-1)
                data = pd.DataFrame(np.cumprod(1. + y_arr, axis=0), columns=['bm_ew', 'bm_mw', 'model_ew', 'model_mw'])
            data['diff_ew'] = np.cumprod(1. + model_ew[f_for_y] - bm_ew['y'])
            data['diff_mw'] = np.cumprod(1. + model_mw[f_for_y] - bm_mw['y'])

            if f_ == 'main':
                data['bm_cost_ew'] = bm_ew['total_cost']
                data['bm_cost_mw'] = bm_mw['total_cost']
                data['bm_turnover_ew'] = bm_ew['turnover']
                data['bm_turnover_mw'] = bm_mw['turnover']
                data['bm_y_w_cost_ew'] = np.cumprod(1. + bm_ew['y_w_cost'], axis=0)
                data['bm_y_w_cost_mw'] = np.cumprod(1. + bm_mw['y_w_cost'], axis=0)
                data['model_cost_ew'] = model_ew['total_cost']
                data['model_cost_mw'] = model_mw['total_cost']
                data['model_turnover_ew'] = model_ew['turnover']
                data['model_turnover_mw'] = model_mw['turnover']
                data['model_y_w_cost_ew'] = np.cumprod(1. + model_ew['y_w_cost'], axis=0)
                data['model_y_w_cost_mw'] = np.cumprod(1. + model_mw['y_w_cost'], axis=0)
                data['diff_w_cost_ew'] = np.cumprod(1. + model_ew['y_w_cost'] - bm_ew['y_w_cost'], axis=0)
                data['diff_w_cost_mw'] = np.cumprod(1. + model_mw['y_w_cost'] - bm_mw['y_w_cost'], axis=0)

            # ################################ figure 1
            if m_args[0] == 'ls':
                # equal fig
                fig = plt.figure()
                fig.suptitle('{} ~ {}'.format(start_d, end_d))
                if f_ == 'main':
                    grid = plt.GridSpec(ncols=2, nrows=4, figure=fig)
                    ax1 = fig.add_subplot(grid[0, 0])
                    ax2 = fig.add_subplot(grid[0, 1])
                    ax3 = fig.add_subplot(grid[1, 0])
                    ax4 = fig.add_subplot(grid[1, 1])
                    ax5 = fig.add_subplot(grid[2, :])
                    ax6 = fig.add_subplot(grid[3, :])
                else:
                    ax1 = fig.add_subplot(221)
                    ax2 = fig.add_subplot(222)
                    ax3 = fig.add_subplot(223)
                    ax4 = fig.add_subplot(224)
                ax1.plot(data[['bm_ew', 'model_ew', 'model_ls_ew', 'model_e1', 'model_e{}'.format(n_tile)]])
                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax1.legend(['bm_ew', 'model_ew', 'long-short', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
                if ylog:
                    ax1.set_yscale('log', basey=2)
                # print max point
                x_pt1 = np.argmax(data['model_ew'].values)
                y_pt1 = data['model_ew'].iloc[x_pt1]
                ax1.annotate("{:.3f}".format(y_pt1), xy=(x_pt1 - 0.5, y_pt1))

                ax2.plot(data[['bm_ew'] + ['model_e{}'.format(i + 1) for i in range(n_tile)]])
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2.legend(['true_y'] + ['q{}'.format(i + 1) for i in range(n_tile)], loc='center left',
                           bbox_to_anchor=(1, 0.5))
                ax2.set_yscale('log', basey=2)

                # value fig
                ax3.plot(data[['bm_mw', 'model_mw', 'model_ls_mw', 'model_m1', 'model_m{}'.format(n_tile)]])
                box = ax3.get_position()
                ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax3.legend(['bm_mw', 'model_mw', 'long-short', 'long', 'short'], loc='center left',
                           bbox_to_anchor=(1, 0.5))
                if ylog:
                    ax3.set_yscale('log', basey=2)
                # print max point
                x_pt3 = np.argmax(data['model_mw'].values)
                y_pt3 = data['model_mw'].iloc[x_pt3]
                ax3.annotate("{:.3f}".format(y_pt3), xy=(x_pt3 - 0.5, y_pt3))

                ax4.plot(data[['bm_mw'] + ['model_m{}'.format(i + 1) for i in range(n_tile)]])
                box = ax4.get_position()
                ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax4.legend(['true_y(mw)'] + ['q{}'.format(i + 1) for i in range(n_tile)], loc='center left',
                           bbox_to_anchor=(1, 0.5))
                ax4.set_yscale('log', basey=2)

                if f_ == 'main':
                    data[['bm_y_w_cost_ew', 'model_y_w_cost_ew', 'bm_y_w_cost_mw', 'model_y_w_cost_mw']].plot(ax=ax5, colormap=cm.Set2)
                    box = ax5.get_position()
                    ax5.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax5.legend(['bm_y_w_cost_ew', 'model_y_w_cost_ew', 'bm_y_w_cost_mw', 'model_y_w_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.8))
                    if ylog:
                        ax5.set_yscale('log', basey=2)

                    ax5_2 = ax5.twinx()
                    data[['diff_w_cost_ew', 'diff_w_cost_mw']].plot(ax=ax5_2, colormap=cm.jet)
                    box = ax5_2.get_position()
                    ax5_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax5_2.legend(['diff_w_cost_ew', 'diff_w_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
                    if ylog:
                        ax5_2.set_yscale('log', basey=2)

                    # print max point
                    x_pt_ew = np.argmax(data['diff_w_cost_ew'].values)
                    y_pt_ew = data['diff_w_cost_ew'].iloc[x_pt_ew]
                    x_pt_mw = np.argmax(data['diff_w_cost_mw'].values)
                    y_pt_mw = data['diff_w_cost_mw'].iloc[x_pt_mw]
                    ax5_2.annotate("{:.3f}".format(y_pt_ew), xy=(x_pt_ew - 0.5, y_pt_ew))
                    ax5_2.annotate("{:.3f}".format(y_pt_mw), xy=(x_pt_mw - 0.5, y_pt_mw))

                    data[['bm_cost_ew', 'bm_cost_mw', 'model_cost_ew', 'model_cost_mw']].plot(ax=ax6, colormap=cm.Set2)
                    box = ax6.get_position()
                    ax6.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax6.legend(['bm_cost_ew', 'bm_cost_mw', 'model_cost_ew', 'model_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.3))
                    if ylog:
                        ax6.set_yscale('log', basey=2)

                    # ax6_2 = ax6.twinx()
                    # data[['bm_turnover_ew', 'bm_turnover_mw', 'model_turnover_ew', 'model_turnover_mw']].plot(ax=ax6_2, colormap=cm.jet)
                    # box = ax6_2.get_position()
                    # ax6_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    # ax6_2.legend(['bm_turnover_ew', 'bm_turnover_mw', 'model_turnover_ew', 'model_turnover_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
                    # if ylog:
                    #     ax6_2.set_yscale('log', basey=2)

                if file_nm is None:
                    save_file_name = '{}/{}'.format(save_dir, '_all.png')
                else:
                    save_dir_v = os.path.join(save_dir, f_)
                    os.makedirs(save_dir_v, exist_ok=True)
                    file_nm_v = file_nm.replace(file_nm[-4:], "_{}{}".format(f_, file_nm[-4:]))
                    save_file_name = '{}/{}'.format(save_dir_v, file_nm_v)

                fig.savefig(save_file_name)
                # print("figure saved. (dir: {})".format(save_file_name))
                plt.close(fig)

            elif m_args[0] == 'l':
                # equal fig
                fig = plt.figure()
                fig.suptitle('{} ~ {}'.format(start_d, end_d))
                if f_ == 'main':
                    ax1 = fig.add_subplot(411)
                    ax2 = fig.add_subplot(412)
                    ax3 = fig.add_subplot(413)
                    ax4 = fig.add_subplot(414)
                else:
                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)

                data[['bm_ew', 'model_ew']].plot(ax=ax1, colormap=cm.Set2)
                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax1.legend(['bm_ew', 'model_ew'], loc='center left', bbox_to_anchor=(1, 0.8))
                if ylog:
                    ax1.set_yscale('log', basey=2)

                ax1_2 = ax1.twinx()
                data[['diff_ew']].plot(ax=ax1_2, colormap=cm.jet)
                box = ax1_2.get_position()
                ax1_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax1_2.legend(['diff_ew'], loc='center left', bbox_to_anchor=(1, 0.2))
                if ylog:
                    ax1_2.set_yscale('log', basey=2)

                # print max point
                x_pt1 = np.argmax(data['diff_ew'].values)
                y_pt1 = data['diff_ew'].iloc[x_pt1]
                ax1_2.annotate("{:.3f}".format(y_pt1), xy=(x_pt1 - 0.5, y_pt1))

                data[['bm_mw', 'model_mw']].plot(ax=ax2, colormap=cm.Set2)
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2.legend(['bm_mw', 'model_mw'], loc='center left', bbox_to_anchor=(1, 0.8))
                if ylog:
                    ax2.set_yscale('log', basey=2)

                ax2_2 = ax2.twinx()
                data[['diff_mw']].plot(ax=ax2_2, colormap=cm.jet)
                box = ax2_2.get_position()
                ax2_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2_2.legend(['diff_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
                if ylog:
                    ax2_2.set_yscale('log', basey=2)

                # print max point
                x_pt2 = np.argmax(data['diff_mw'].values)
                y_pt2 = data['diff_mw'].iloc[x_pt2]
                ax2_2.annotate("{:.3f}".format(y_pt2), xy=(x_pt2 - 0.5, y_pt2))

                if f_ == 'main':
                    data[['bm_y_w_cost_ew', 'model_y_w_cost_ew', 'bm_y_w_cost_mw', 'model_y_w_cost_mw']].plot(ax=ax3, colormap=cm.Set2)
                    box = ax3.get_position()
                    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax3.legend(['bm_y_w_cost_ew', 'model_y_w_cost_ew', 'bm_y_w_cost_mw', 'model_y_w_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.8))
                    if ylog:
                        ax3.set_yscale('log', basey=2)

                    ax3_2 = ax3.twinx()
                    data[['diff_w_cost_ew', 'diff_w_cost_mw']].plot(ax=ax3_2, colormap=cm.jet)
                    box = ax3_2.get_position()
                    ax3_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax3_2.legend(['diff_w_cost_ew', 'diff_w_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
                    if ylog:
                        ax3_2.set_yscale('log', basey=2)

                    # print max point
                    x_pt_ew = np.argmax(data['diff_w_cost_ew'].values)
                    y_pt_ew = data['diff_w_cost_ew'].iloc[x_pt_ew]
                    x_pt_mw = np.argmax(data['diff_w_cost_mw'].values)
                    y_pt_mw = data['diff_w_cost_mw'].iloc[x_pt_mw]
                    ax3_2.annotate("{:.3f}".format(y_pt_ew), xy=(x_pt_ew - 0.5, y_pt_ew))
                    ax3_2.annotate("{:.3f}".format(y_pt_mw), xy=(x_pt_mw - 0.5, y_pt_mw))

                    data[['bm_cost_ew', 'bm_cost_mw', 'model_cost_ew', 'model_cost_mw']].plot(ax=ax4, colormap=cm.Set2)
                    box = ax4.get_position()
                    ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax4.legend(['bm_cost_ew', 'bm_cost_mw', 'model_cost_ew', 'model_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.5))
                    if ylog:
                        ax4.set_yscale('log', basey=2)

                    # ax4_2 = ax4.twinx()
                    # data[['bm_turnover_ew', 'bm_turnover_mw', 'model_turnover_ew', 'model_turnover_mw']].plot(ax=ax4_2, colormap=cm.jet)
                    # box = ax4_2.get_position()
                    # ax4_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    # ax4_2.legend(['bm_turnover_ew', 'bm_turnover_mw', 'model_turnover_ew', 'model_turnover_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
                    # if ylog:
                    #     ax4_2.set_yscale('log', basey=2)

                if file_nm is None:
                    save_file_name = '{}/{}'.format(save_dir, '_all.png')
                else:
                    save_dir_v = os.path.join(save_dir, f_)
                    os.makedirs(save_dir_v, exist_ok=True)
                    file_nm_v = file_nm.replace(file_nm[-4:], "_{}{}".format(f_, file_nm[-4:]))
                    save_file_name = '{}/{}'.format(save_dir_v, file_nm_v)

                fig.savefig(save_file_name)
                # print("figure saved. (dir: {})".format(save_file_name))
                plt.close(fig)

    def predict_plot_monthly(self, model, dataloader_set, save_dir
                             , file_nm='test.png'
                             , ylog=False
                             , ls_method='ls_5_20'
                             , plot_all_features=True
                             , debug=False):

        c = self.configs
        rate_ = c.app_rate
        # file_nm = 'test.png'; ylog = False; t_stepsize = 4; ls_method = 'ls_5_20'; plot_all_features = True
        m_args = ls_method.split('_')
        if m_args[0] == 'ls':
            n_tile = int(m_args[1])
            def_variables = self.define_variables_ntile
            kw = {'n_tile': n_tile}
        else:
            n_tile = -1
            def_variables = self.define_variables
            kw = {}

        # dataloader: (features: dict of Torch.Tensor, add_infos: dict of np.array)
        dataloader, features_list, all_assets_list, start_d, end_d = dataloader_set

        results = list()

        t_steps = len(dataloader[0]) + 1

        # define variables to save values
        if plot_all_features:
            model_keys = list(model.predictor.keys())
            # features_for_plot = ['main'] + model_keys
            features_for_plot = ['main']
        else:
            model_keys = None
            features_for_plot = ['main']

        ew_dict = dict(bm=self.define_variables(t_steps=t_steps, assets=all_assets_list),
                       model=def_variables(t_steps=t_steps, assets=all_assets_list, f_keys=model_keys, **kw),
                       model_w_factor=def_variables(t_steps=t_steps, assets=all_assets_list, **kw))
        mw_dict = dict(bm=self.define_variables(t_steps=t_steps, assets=all_assets_list),
                       model=def_variables(t_steps=t_steps, assets=all_assets_list, f_keys=model_keys, **kw),
                       factor=def_variables(t_steps=t_steps, assets=all_assets_list, **kw),
                       model_w_factor=def_variables(t_steps=t_steps, assets=all_assets_list, **kw))

        # nickname
        bm_ew = ew_dict['bm']
        bm_mw = mw_dict['bm']
        model_ew = ew_dict['model']
        model_mw = mw_dict['model']

        factor_mw = mw_dict['factor']
        fm_ew = ew_dict['model_w_factor']
        fm_mw = mw_dict['model_w_factor']

        for i, (features, add_info) in enumerate(zip(*dataloader)):
            t = i + 1
            if debug:
                print('{} / {}'.format(add_info['factor_d'], add_info['model_d']))

            mc = np.array(add_info['mktcap'], dtype=np.float32).squeeze()
            label_y = np.array(add_info['next_y'])

            assets = np.array(add_info['asset_list'])

            result_t = add_info['univ'].set_index('infocode').loc[assets]
            result_t.wgt[result_t.wgt.isna()] = 0.
            result_t['wgt'] = result_t['wgt'] / np.sum(result_t['wgt'])

            # ############ For BenchMark ############

            bm_ew['y'][t] = np.mean(label_y)
            bm_mw['y'][t] = np.sum(label_y * mc) / np.sum(mc)

            # cost calculation
            bm_ew['wgt'].loc[:, 'new'] = 0
            bm_ew['wgt'].loc[assets, 'new'] = 1. / len(assets)
            self.calculate_cost(t, bm_ew, assets, label_y)

            bm_mw['wgt'].loc[:, 'new'] = 0
            bm_mw['wgt'].loc[assets, 'new'] = mc / np.sum(mc)
            self.calculate_cost(t, bm_mw, assets, label_y)

            # ############ For Factor ############
            factor_mw['y'][t] = np.sum(label_y * np.array(result_t['wgt']))

            factor_mw['wgt'].loc[:, 'new'] = 0
            factor_mw['wgt'].loc[assets, 'new'] = np.array(result_t['wgt'])
            self.calculate_cost(t, factor_mw, assets, label_y)

            # ############ For Model ############
            # prediction
            predictions = model.predict_mtl(features)
            for key in predictions.keys():
                predictions[key] = tu.np_ify(predictions[key])

            value_ = dict()

            value_[self.adj_feature] = predictions[self.adj_feature][:, 0, 0]
            for f_ in features_for_plot:
                f_for_y = ('y' if f_ == 'main' else f_)
                if f_ == 'main':
                    value_['main'] = predictions[self.pred_feature][:, 0, 0]
                else:
                    value_[f_] = predictions[f_][:, 0, 0]

                if m_args[0] == 'ls':
                    # ntile 별 수익률
                    scale_n = weight_scale(value_[f_], method='each_{}'.format(n_tile))
                    model_ew[f_for_y + '_each'][t, :] = np.matmul(label_y, scale_n) / np.sum(scale_n, axis=0)
                    model_mw[f_for_y + '_each'][t, :] = np.matmul(label_y * mc, scale_n) / np.matmul(mc, scale_n)
                    # or np.sum((label_y * mc).reshape([-1, 1]) * scale, axis=0) / np.sum(mc.reshape(-1, 1) * scale, axis=0)

                    # pf 수익률
                    scale = weight_scale(value_[f_], method=ls_method)
                elif m_args[0] == 'l':
                    scale1 = weight_scale(value_[f_], method=ls_method)
                    scale = scale1 * weight_scale(value_[self.adj_feature], method=ls_method)
                elif m_args[0] == 'limit-LS':
                    pass

                model_ew[f_for_y][t] = np.sum(label_y * scale) / np.sum(scale)
                model_mw[f_for_y][t] = np.sum(label_y * mc * scale) / np.sum(mc * scale)

                if f_ == 'main':
                    # cost calculation
                    model_ew['wgt'].loc[:, 'new'] = 0
                    model_ew['wgt'].loc[assets, 'new'] = scale / np.sum(scale)
                    self.calculate_cost(t, model_ew, assets, label_y)

                    model_mw['wgt'].loc[:, 'new'] = 0
                    model_mw['wgt'].loc[assets, 'new'] = mc * scale / np.sum(mc * scale)
                    self.calculate_cost(t, model_mw, assets, label_y)

                    # Equal Weight Mix
                    new_wgt_ew = self.scale_to_wgt(result_t['wgt'], scale, rate_, mc, w_method='ew')
                    # Market Weight Mix
                    new_wgt_mw = self.scale_to_wgt(result_t['wgt'], scale, rate_, mc, w_method='mw')

                    fm_ew['y'][t] = np.sum(label_y * new_wgt_ew)
                    fm_ew['wgt'].loc[:, 'new'] = 0
                    fm_ew['wgt'].loc[assets, 'new'] = new_wgt_ew
                    self.calculate_cost(t, fm_ew, assets, label_y)

                    fm_mw['y'][t] = np.sum(label_y * new_wgt_mw)
                    fm_mw['wgt'].loc[:, 'new'] = 0
                    fm_mw['wgt'].loc[assets, 'new'] = new_wgt_mw
                    self.calculate_cost(t, fm_mw, assets, label_y)

            result_t['bm_wgt_ew'] = bm_ew['wgt'].loc[assets, 'new']
            result_t['bm_wgt_mw'] = bm_mw['wgt'].loc[assets, 'new']
            result_t['model_wgt_ew'] = model_ew['wgt'].loc[assets, 'new']
            result_t['model_wgt_mw'] = model_mw['wgt'].loc[assets, 'new']
            results.append(result_t)

        for f_ in features_for_plot:
            f_for_y = ('y' if f_ == 'main' else f_)
            if m_args[0] == 'ls':
                y_arr = np.concatenate([bm_ew['y'], bm_mw['y']
                                        , model_ew[f_for_y]
                                        , model_mw[f_for_y]
                                        , model_ew[f_for_y + '_each']
                                        , model_mw[f_for_y + '_each']
                                        , factor_mw['y']
                                        , fm_ew['y'], fm_mw['y']], axis=-1)
                data = pd.DataFrame(np.cumprod(1. + y_arr, axis=0)
                                    , columns=['bm_ew', 'bm_mw', 'model_ew', 'model_mw']
                                              + ['model_e{}'.format(i+1) for i in range(n_tile)]
                                              + ['model_m{}'.format(i+1) for i in range(n_tile)]
                                              + ['factor_mw', 'fm_ew', 'fm_mw'])
                data['model_ls_ew'] = np.cumprod(1. + np.mean(model_ew[f_for_y + '_each'][:, :1], axis=1) - np.mean(model_ew[f_for_y + '_each'][:, -1:], axis=1))
                data['model_ls_mw'] = np.cumprod(1. + np.mean(model_mw[f_for_y + '_each'][:, :1], axis=1) - np.mean(model_mw[f_for_y + '_each'][:, -1:], axis=1))

            else:
                y_arr = np.concatenate([bm_ew['y'], bm_mw['y'], model_ew[f_for_y], model_mw[f_for_y]
                                        , factor_mw['y']
                                        , fm_ew['y'], fm_mw['y']], axis=-1)
                data = pd.DataFrame(np.cumprod(1. + y_arr, axis=0), columns=['bm_ew', 'bm_mw', 'model_ew', 'model_mw','factor_mw', 'fm_ew', 'fm_mw'])
            data['diff_ew'] = np.cumprod(1. + model_ew[f_for_y] - bm_ew['y'])
            data['diff_mw'] = np.cumprod(1. + model_mw[f_for_y] - bm_mw['y'])

            if f_ == 'main':
                data['bm_cost_ew'] = bm_ew['total_cost']
                data['bm_cost_mw'] = bm_mw['total_cost']
                data['bm_turnover_ew'] = bm_ew['turnover']
                data['bm_turnover_mw'] = bm_mw['turnover']
                data['bm_y_w_cost_ew'] = np.cumprod(1. + bm_ew['y_w_cost'], axis=0)
                data['bm_y_w_cost_mw'] = np.cumprod(1. + bm_mw['y_w_cost'], axis=0)
                data['model_cost_ew'] = model_ew['total_cost']
                data['model_cost_mw'] = model_mw['total_cost']
                data['model_turnover_ew'] = model_ew['turnover']
                data['model_turnover_mw'] = model_mw['turnover']
                data['model_y_w_cost_ew'] = np.cumprod(1. + model_ew['y_w_cost'], axis=0)
                data['model_y_w_cost_mw'] = np.cumprod(1. + model_mw['y_w_cost'], axis=0)
                data['diff_w_cost_ew'] = np.cumprod(1. + model_ew['y_w_cost'] - bm_ew['y_w_cost'], axis=0)
                data['diff_w_cost_mw'] = np.cumprod(1. + model_mw['y_w_cost'] - bm_mw['y_w_cost'], axis=0)


                data['factor_cost_mw'] = factor_mw['total_cost']
                data['fm_cost_ew'] = fm_ew['total_cost']
                data['fm_cost_mw'] = fm_mw['total_cost']

                data['factor_turnover_mw'] = factor_mw['turnover']
                data['fm_turnover_ew'] = fm_ew['turnover']
                data['fm_turnover_mw'] = fm_mw['turnover']

                data['factor_y_w_cost_mw'] = np.cumprod(1. + factor_mw['y_w_cost'], axis=0)
                data['fm_y_w_cost_ew'] = np.cumprod(1. + fm_ew['y_w_cost'], axis=0)
                data['fm_y_w_cost_mw'] = np.cumprod(1. + fm_mw['y_w_cost'], axis=0)

                data['diff_ew_m'] = np.cumprod(1. + model_ew['y_w_cost'] - bm_mw['y_w_cost'])
                data['diff_mw_m'] = np.cumprod(1. + model_mw['y_w_cost'] - bm_mw['y_w_cost'])
                data['diff_factor_m'] = np.cumprod(1. + factor_mw['y_w_cost'] - bm_mw['y_w_cost'])
                data['diff_fme_m'] = np.cumprod(1. + fm_ew['y_w_cost'] - bm_mw['y_w_cost'])
                data['diff_fmm_m'] = np.cumprod(1. + fm_mw['y_w_cost'] - bm_mw['y_w_cost'])
                data['diff_fme_f'] = np.cumprod(1. + fm_ew['y_w_cost'] - factor_mw['y_w_cost'])
                data['diff_fmm_f'] = np.cumprod(1. + fm_mw['y_w_cost'] - factor_mw['y_w_cost'])

            # ################################ figure 1

            # ############################# figure main (compare)
            fig = plt.figure()
            fig.suptitle('{} ~ {}'.format(start_d, end_d))

            grid = plt.GridSpec(ncols=1, nrows=2, figure=fig)
            ax1 = fig.add_subplot(grid[0, 0])
            ax2 = fig.add_subplot(grid[1, 0])

            ax1.plot(data[['diff_mw_m', 'diff_factor_m', 'diff_fme_m', 'diff_fmm_m']])
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax1.legend(['diff_mw_m', 'diff_factor_m', 'diff_fme_m', 'diff_fmm_m'], loc='center left', bbox_to_anchor=(1, 0.8))
            # ax2.set_yscale('log', basey=2)

            # ax1_2 = ax1.twinx()
            # data[['diff_ew_m']].plot(ax=ax1_2, colormap=cm.jet)
            # box = ax1_2.get_position()
            # ax1_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # ax1_2.legend(['diff_ew_m'], loc='center left', bbox_to_anchor=(1, 0.2))
            # if ylog:
            #     ax1_2.set_yscale('log', basey=2)
            ax1_2 = ax1.twinx()
            data[['diff_fme_f', 'diff_fmm_f']].plot(ax=ax1_2, colormap=cm.jet)
            box = ax1_2.get_position()
            ax1_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax1_2.legend(['diff_fme_f', 'diff_fmm_f'], loc='center left', bbox_to_anchor=(1, 0.2))
            if ylog:
                ax1_2.set_yscale('log', basey=2)

            data[['bm_turnover_ew', 'bm_turnover_mw', 'factor_turnover_mw', 'fm_turnover_ew', 'fm_turnover_mw']].plot(ax=ax2, colormap=cm.Set2)
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax2.legend(['bm_turnover_ew', 'bm_turnover_mw', 'factor_turnover_mw', 'fm_turnover_ew', 'fm_turnover_mw'], loc='center left',
                       bbox_to_anchor=(1, 0.5))
            if ylog:
                ax2.set_yscale('log', basey=2)

            if file_nm is None:
                save_file_name = '{}/{}'.format(save_dir, '_all.png')
            else:
                save_dir_v = os.path.join(save_dir, 'compare')
                os.makedirs(save_dir_v, exist_ok=True)
                file_nm_v = file_nm.replace(file_nm[-4:], "_{}{}".format('compare', file_nm[-4:]))
                save_file_name = '{}/{}'.format(save_dir_v, file_nm_v)

            fig.savefig(save_file_name)
            # print("figure saved. (dir: {})".format(save_file_name))
            plt.close(fig)

            # ############################# figure main (compare) end

            if m_args[0] == 'ls':
                # equal fig
                fig = plt.figure()
                fig.suptitle('{} ~ {}'.format(start_d, end_d))
                if f_ == 'main':
                    grid = plt.GridSpec(ncols=2, nrows=4, figure=fig)
                    ax1 = fig.add_subplot(grid[0, 0])
                    ax2 = fig.add_subplot(grid[0, 1])
                    ax3 = fig.add_subplot(grid[1, 0])
                    ax4 = fig.add_subplot(grid[1, 1])
                    ax5 = fig.add_subplot(grid[2, :])
                    ax6 = fig.add_subplot(grid[3, :])
                else:
                    ax1 = fig.add_subplot(221)
                    ax2 = fig.add_subplot(222)
                    ax3 = fig.add_subplot(223)
                    ax4 = fig.add_subplot(224)
                ax1.plot(data[['bm_ew', 'model_ew', 'model_ls_ew', 'model_e1', 'model_e{}'.format(n_tile)]])
                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax1.legend(['bm_ew', 'model_ew', 'long-short', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
                if ylog:
                    ax1.set_yscale('log', basey=2)
                # print max point
                x_pt1 = np.argmax(data['model_ew'].values)
                y_pt1 = data['model_ew'].iloc[x_pt1]
                ax1.annotate("{:.3f}".format(y_pt1), xy=(x_pt1 - 0.5, y_pt1))

                ax2.plot(data[['bm_ew'] + ['model_e{}'.format(i + 1) for i in range(n_tile)]])
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2.legend(['true_y'] + ['q{}'.format(i + 1) for i in range(n_tile)], loc='center left',
                           bbox_to_anchor=(1, 0.5))
                ax2.set_yscale('log', basey=2)

                # value fig
                ax3.plot(data[['bm_mw', 'model_mw', 'model_ls_mw', 'model_m1', 'model_m{}'.format(n_tile)]])
                box = ax3.get_position()
                ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax3.legend(['bm_mw', 'model_mw', 'long-short', 'long', 'short'], loc='center left',
                           bbox_to_anchor=(1, 0.5))
                if ylog:
                    ax3.set_yscale('log', basey=2)
                # print max point
                x_pt3 = np.argmax(data['model_mw'].values)
                y_pt3 = data['model_mw'].iloc[x_pt3]
                ax3.annotate("{:.3f}".format(y_pt3), xy=(x_pt3 - 0.5, y_pt3))

                ax4.plot(data[['bm_mw'] + ['model_m{}'.format(i + 1) for i in range(n_tile)]])
                box = ax4.get_position()
                ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax4.legend(['true_y(mw)'] + ['q{}'.format(i + 1) for i in range(n_tile)], loc='center left',
                           bbox_to_anchor=(1, 0.5))
                ax4.set_yscale('log', basey=2)

                if f_ == 'main':
                    data[['bm_y_w_cost_ew', 'model_y_w_cost_ew', 'bm_y_w_cost_mw', 'model_y_w_cost_mw']].plot(ax=ax5, colormap=cm.Set2)
                    box = ax5.get_position()
                    ax5.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax5.legend(['bm_y_w_cost_ew', 'model_y_w_cost_ew', 'bm_y_w_cost_mw', 'model_y_w_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.8))
                    if ylog:
                        ax5.set_yscale('log', basey=2)

                    ax5_2 = ax5.twinx()
                    data[['diff_w_cost_ew', 'diff_w_cost_mw']].plot(ax=ax5_2, colormap=cm.jet)
                    box = ax5_2.get_position()
                    ax5_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax5_2.legend(['diff_w_cost_ew', 'diff_w_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
                    if ylog:
                        ax5_2.set_yscale('log', basey=2)

                    # print max point
                    x_pt_ew = np.argmax(data['diff_w_cost_ew'].values)
                    y_pt_ew = data['diff_w_cost_ew'].iloc[x_pt_ew]
                    x_pt_mw = np.argmax(data['diff_w_cost_mw'].values)
                    y_pt_mw = data['diff_w_cost_mw'].iloc[x_pt_mw]
                    ax5_2.annotate("{:.3f}".format(y_pt_ew), xy=(x_pt_ew - 0.5, y_pt_ew))
                    ax5_2.annotate("{:.3f}".format(y_pt_mw), xy=(x_pt_mw - 0.5, y_pt_mw))

                    data[['bm_cost_ew', 'bm_cost_mw', 'model_cost_ew', 'model_cost_mw']].plot(ax=ax6, colormap=cm.Set2)
                    box = ax6.get_position()
                    ax6.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax6.legend(['bm_cost_ew', 'bm_cost_mw', 'model_cost_ew', 'model_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.3))
                    if ylog:
                        ax6.set_yscale('log', basey=2)

                    # ax6_2 = ax6.twinx()
                    # data[['bm_turnover_ew', 'bm_turnover_mw', 'model_turnover_ew', 'model_turnover_mw']].plot(ax=ax6_2, colormap=cm.jet)
                    # box = ax6_2.get_position()
                    # ax6_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    # ax6_2.legend(['bm_turnover_ew', 'bm_turnover_mw', 'model_turnover_ew', 'model_turnover_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
                    # if ylog:
                    #     ax6_2.set_yscale('log', basey=2)

                if file_nm is None:
                    save_file_name = '{}/{}'.format(save_dir, '_all.png')
                else:
                    save_dir_v = os.path.join(save_dir, f_)
                    os.makedirs(save_dir_v, exist_ok=True)
                    file_nm_v = file_nm.replace(file_nm[-4:], "_{}{}".format(f_, file_nm[-4:]))
                    save_file_name = '{}/{}'.format(save_dir_v, file_nm_v)

                fig.savefig(save_file_name)
                # print("figure saved. (dir: {})".format(save_file_name))
                plt.close(fig)

            elif m_args[0] == 'l':
                # equal fig
                fig = plt.figure()
                fig.suptitle('{} ~ {}'.format(start_d, end_d))
                if f_ == 'main':
                    ax1 = fig.add_subplot(411)
                    ax2 = fig.add_subplot(412)
                    ax3 = fig.add_subplot(413)
                    ax4 = fig.add_subplot(414)
                else:
                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)

                data[['bm_ew', 'model_ew']].plot(ax=ax1, colormap=cm.Set2)
                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax1.legend(['bm_ew', 'model_ew'], loc='center left', bbox_to_anchor=(1, 0.8))
                if ylog:
                    ax1.set_yscale('log', basey=2)

                ax1_2 = ax1.twinx()
                data[['diff_ew']].plot(ax=ax1_2, colormap=cm.jet)
                box = ax1_2.get_position()
                ax1_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax1_2.legend(['diff_ew'], loc='center left', bbox_to_anchor=(1, 0.2))
                if ylog:
                    ax1_2.set_yscale('log', basey=2)

                # print max point
                x_pt1 = np.argmax(data['diff_ew'].values)
                y_pt1 = data['diff_ew'].iloc[x_pt1]
                ax1_2.annotate("{:.3f}".format(y_pt1), xy=(x_pt1 - 0.5, y_pt1))

                data[['bm_mw', 'model_mw']].plot(ax=ax2, colormap=cm.Set2)
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2.legend(['bm_mw', 'model_mw'], loc='center left', bbox_to_anchor=(1, 0.8))
                if ylog:
                    ax2.set_yscale('log', basey=2)

                ax2_2 = ax2.twinx()
                data[['diff_mw']].plot(ax=ax2_2, colormap=cm.jet)
                box = ax2_2.get_position()
                ax2_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2_2.legend(['diff_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
                if ylog:
                    ax2_2.set_yscale('log', basey=2)

                # print max point
                x_pt2 = np.argmax(data['diff_mw'].values)
                y_pt2 = data['diff_mw'].iloc[x_pt2]
                ax2_2.annotate("{:.3f}".format(y_pt2), xy=(x_pt2 - 0.5, y_pt2))

                if f_ == 'main':
                    data[['bm_y_w_cost_ew', 'model_y_w_cost_ew', 'bm_y_w_cost_mw', 'model_y_w_cost_mw']].plot(ax=ax3, colormap=cm.Set2)
                    box = ax3.get_position()
                    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax3.legend(['bm_y_w_cost_ew', 'model_y_w_cost_ew', 'bm_y_w_cost_mw', 'model_y_w_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.8))
                    if ylog:
                        ax3.set_yscale('log', basey=2)

                    ax3_2 = ax3.twinx()
                    data[['diff_w_cost_ew', 'diff_w_cost_mw']].plot(ax=ax3_2, colormap=cm.jet)
                    box = ax3_2.get_position()
                    ax3_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax3_2.legend(['diff_w_cost_ew', 'diff_w_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
                    if ylog:
                        ax3_2.set_yscale('log', basey=2)

                    # print max point
                    x_pt_ew = np.argmax(data['diff_w_cost_ew'].values)
                    y_pt_ew = data['diff_w_cost_ew'].iloc[x_pt_ew]
                    x_pt_mw = np.argmax(data['diff_w_cost_mw'].values)
                    y_pt_mw = data['diff_w_cost_mw'].iloc[x_pt_mw]
                    ax3_2.annotate("{:.3f}".format(y_pt_ew), xy=(x_pt_ew - 0.5, y_pt_ew))
                    ax3_2.annotate("{:.3f}".format(y_pt_mw), xy=(x_pt_mw - 0.5, y_pt_mw))

                    data[['bm_cost_ew', 'bm_cost_mw', 'model_cost_ew', 'model_cost_mw']].plot(ax=ax4, colormap=cm.Set2)
                    box = ax4.get_position()
                    ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax4.legend(['bm_cost_ew', 'bm_cost_mw', 'model_cost_ew', 'model_cost_mw'], loc='center left', bbox_to_anchor=(1, 0.5))
                    if ylog:
                        ax4.set_yscale('log', basey=2)

                    # ax4_2 = ax4.twinx()
                    # data[['bm_turnover_ew', 'bm_turnover_mw', 'model_turnover_ew', 'model_turnover_mw']].plot(ax=ax4_2, colormap=cm.jet)
                    # box = ax4_2.get_position()
                    # ax4_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    # ax4_2.legend(['bm_turnover_ew', 'bm_turnover_mw', 'model_turnover_ew', 'model_turnover_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
                    # if ylog:
                    #     ax4_2.set_yscale('log', basey=2)

                if file_nm is None:
                    save_file_name = '{}/{}'.format(save_dir, '_all.png')
                else:
                    save_dir_v = os.path.join(save_dir, f_)
                    os.makedirs(save_dir_v, exist_ok=True)
                    file_nm_v = file_nm.replace(file_nm[-4:], "_{}{}".format(f_, file_nm[-4:]))
                    save_file_name = '{}/{}'.format(save_dir_v, file_nm_v)

                fig.savefig(save_file_name)
                # print("figure saved. (dir: {})".format(save_file_name))
                plt.close(fig)

