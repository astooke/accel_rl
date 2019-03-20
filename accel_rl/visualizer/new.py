
import os
from rllab.misc import ext
import csv
import numpy as np
import json
import itertools
import plotly.offline as po
import plotly.graph_objs as go
from plotly import tools

# from IPython.core.debugger import set_trace

exps_data = None
plottable_keys = None
distinct_params = None


def plot(
        y_key,
        x_key=None,
        x_scale=1.,
        split_key=None,
        group_key=None,
        filters=None,
        custom_filter=None,
        clip_plot_value=None,
        plot_width=None,
        plot_height=None,
        filter_nan=False,
        smooth_curve=False,
        legend_post_processor=None,
        normalize_error=False,
        squeeze_nan=False,
        xlim=None, ylim=None,
        show_exp_count=False,
        sub_plots=False,
        sort_int_legend=False,
        same_legend=False,
        legend_title=None,
        legend_title_x=1.05,
        legend_title_y=1.01,
        plot_stds=True,
        font_size=12,
        legend_font_size=None,
        line_dash_list=None,
        n_rows=1,
        dtick=None,
):
    if legend_font_size is None:
        legend_font_size = font_size
    print(y_key, split_key, group_key, filters)
    if filter_nan:
        nonnan_exps_data = list(filter(check_nan, exps_data))
        selector = Selector(nonnan_exps_data)
    else:
        selector = Selector(exps_data)
    if legend_post_processor is None:
        legend_post_processor = lambda x: x
    if filters is None:
        filters = dict()
    for k, v in filters.items():
        selector = selector.where(k, str(v))
    if custom_filter is not None:
        selector = selector.custom_filter(custom_filter)
    # print selector._filters

    if split_key is not None:
        vs = [vs for k, vs in distinct_params if k == split_key][0]
        split_selectors = [selector.where(split_key, v) for v in vs]
        split_legends = list(map(str, vs))
    else:
        split_selectors = [selector]
        split_legends = ["Plot"]
    plots = []
    counter = 1
    if sub_plots:
        n_per_row = -(-len(split_legends) // n_rows)
        fig = tools.make_subplots(rows=n_rows, cols=-(-len(split_legends) // n_rows), subplot_titles=split_legends)
        n_groups = None
    for split_selector, split_legend in zip(split_selectors, split_legends):
        print("split_legend: ", split_legend)
        if group_key and group_key is not "exp_name":
            vs = [vs for k, vs in distinct_params if k == group_key][0]
            group_selectors = [split_selector.where(group_key, v) for v in vs]
            group_legends = [str(x) for x in vs]
        else:
            group_key = "exp_name"
            vs = sorted([x.params["exp_name"] for x in split_selector.extract()])
            group_selectors = [split_selector.where(group_key, v) for v in vs]
            group_legends = [summary_name(x.extract()[0], split_selector) for x in group_selectors]

        if sort_int_legend:
            _, old_idxs = \
                zip(*sorted((int(g), i) for i, g in enumerate(group_legends)))
            group_selectors = [group_selectors[i] for i in old_idxs]
            group_legends = [group_legends[i] for i in old_idxs]
        to_plot = []
        for group_selector, group_legend in zip(group_selectors, group_legends):
            print("group_legend: ", group_legend)
            filtered_data = group_selector.extract()
            if show_exp_count:
                group_legend += " (%d)"%(len(filtered_data))
            print("len(filtered_data): ", len(filtered_data))
            if len(filtered_data) > 0:

                progresses = [
                    exp.progress.get(y_key, np.array([np.nan])) for exp in filtered_data]
                sizes = list(map(len, progresses))
                # more intelligent:
                max_size = max(sizes)
                progresses = [
                    np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]
                window_size = np.maximum(int(np.round(max_size / float(1000))), 1)

                means = np.nanmean(progresses, axis=0)
                stds = np.nanstd(progresses, axis=0)
                if not plot_stds:
                    stds[:] = 0

                if x_key is not None:
                    xs = [exp.progress.get(x_key, np.array([np.nan])) for exp in filtered_data]
                    # set_trace()
                    # IPython import embed; embed()
                    if not all([len(xi) == len(xs[0]) for xi in xs]):
                        print("WARNING: different length xs within group, using longest")
                        lengths = [len(xi) for xi in xs]
                        max_i = np.argmax(lengths)
                        xs[0] = xs[max_i]
                    elif not all([all(xi == xs[0]) for xi in xs]):
                        print("WARNING: different xs within group, using one of them for all")
                    xs = xs[0]
                else:
                    xs = list(range(len(means)))
                xs = [x * x_scale for x in xs]

                if smooth_curve:
                    means = sliding_mean(means,
                                         window=window_size)
                    stds = sliding_mean(stds,
                                        window=window_size)
                print(len(xs), len(means))
                assert len(xs) == len(means)

                if clip_plot_value is not None:
                    means = np.clip(means, -clip_plot_value, clip_plot_value)
                    stds = np.clip(stds, -clip_plot_value, clip_plot_value)
                to_plot.append(
                    ext.AttrDict(xs=xs, means=means, stds=stds, legend=legend_post_processor(group_legend)))
            elif same_legend:
                to_plot.append(
                    ext.AttrDict(xs=np.array([np.nan]), means=np.array([np.nan]), stds=np.array([np.nan]), legend=legend_post_processor(group_legend)))
        
        if len(to_plot) > 0:
            if sub_plots:
                plot_data = make_plot_data(to_plot, showlegend=(counter == 1))
                this_row = ((counter - 1) // n_per_row) + 1
                this_col = ((counter - 1) % n_per_row) + 1
                print("counter, row, col: ", counter, this_row, this_col)
                for data in plot_data:
                    fig.append_trace(data, this_row, this_col)
            else:
                fig_title = "%s: %s" % (split_key, split_legend)

                fig = make_plot(
                    to_plot,
                    title=fig_title,
                    plot_width=plot_width, plot_height=plot_height,
                    xlim=xlim, ylim=ylim,
                    font_size=font_size,
                    legend_font_size=legend_font_size,
                    line_dash_list=line_dash_list,
                )

                if legend_title is not None:
                    legend_title_annot = go.Annotation(
                        x=legend_title_x,
                        y=legend_title_y,
                        align="right",
                        valign="top",
                        text=legend_title,
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        xanchor="middle",
                        yanchor="top",
                    )
                    
                    fig['layout']["annotations"] += [legend_title_annot]
                po.iplot(fig)

        counter += 1

    if sub_plots:
        fig['layout'].update(height=plot_height, width=plot_width)
        # set_trace()
        if xlim is not None:
            for i in range(1, counter):
                fig['layout']['xaxis' + str(i)].update(range=[0, xlim])
        if dtick is not None:
            for i in range(1, counter):
                fig['layout']['xaxis' + str(i)].update(dtick=dtick)
            # fig['layout']['xaxis'].update(range=[0, xlim])
        # set_trace()
        fig['layout']['font'].update(size=font_size)
        fig['layout']['legend']['font']['size'] = legend_font_size
        for subtitle in fig['layout']['annotations']:
            subtitle['font']['size'] = font_size
        # fig['layout']['titlefont'].update(size=font_size)
        if legend_title is not None:
            legend_title = go.Annotation(
                x=legend_title_x,
                y=legend_title_y,
                align="right",
                valign="top",
                text=legend_title,
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor="middle",
                yanchor="top",
            )
            
            fig['layout']["annotations"] += [legend_title]
        po.iplot(fig)
    # return plots


def make_plot(plot_list, plot_width=None, plot_height=None, title=None,
              xlim=None, ylim=None, font_size=12, legend_font_size=12,
              line_dash_list=None):
    data = []
    p25, p50, p75 = [], [], []
    p0, p100 = [], []
    line_dash_list = list() if line_dash_list is None else line_dash_list.copy()
    for idx, plt in enumerate(plot_list):
        color = color_defaults[idx % len(color_defaults)]
        # x = list(range(len(plt.means)))
        x = list(plt.xs)
        y = list(plt.means)
        y_upper = list(plt.means + plt.stds)
        y_lower = list(plt.means - plt.stds)
        y_extras = []

        # # NOTE: Adam Edits
        # x = [i / 10 for i in x]

        # if hasattr(plt, "custom_x"):
        #     x = list(plt.custom_x)

        data.append(go.Scatter(
            x=x + x[::-1],
            y=y_upper + y_lower[::-1],
            fill='tozerox',
            fillcolor=hex_to_rgb(color, 0.2),
            line=go.Line(color='transparent'),
            showlegend=False,
            legendgroup=plt.legend,
            hoverinfo='none'
        ))
        dash = "solid" if not line_dash_list else line_dash_list.pop(0)
        data.append(go.Scatter(
            x=x,
            y=y,
            name=plt.legend,
            legendgroup=plt.legend,
            line=dict(color=hex_to_rgb(color), dash=dash),
        ))

        for y_extra in y_extras:
            data.append(go.Scatter(
                x=x,
                y=y_extra,
                showlegend=False,
                legendgroup=plt.legend,
                line=dict(color=hex_to_rgb(color), dash='dot')
                # choices: solid, dot, dash, longdash, dashdot, longdashdot
            ))

    # def numeric_list_to_string(numbers):
    #     s = '['
    #     for num in numbers:
    #         s += (str(num) + ',')
    #     s += ']'
    #     return s

    # print(numeric_list_to_string(p25))
    # print(numeric_list_to_string(p50))
    # print(numeric_list_to_string(p75))

    layout = go.Layout(
        legend=dict(
            x=1,
            y=1,
            # xanchor="left",
            # yanchor="bottom",
            font=dict(size=legend_font_size),
        ),
        width=plot_width,  # 500,  # NOTE: Adam edit
        height=plot_height,
        title=title,
        # xaxis=go.XAxis(range=xlim),
        # yaxis=go.YAxis(range=ylim),
        xaxis=dict(range=[0, xlim]),
        yaxis=dict(range=[0, ylim]),
        font=dict(size=font_size),
        titlefont=dict(size=font_size),
    )
    # return data

    fig = go.Figure(data=data, layout=layout)
    # po.iplot(fig)
    # # # fig_div = po.plot(fig, output_type='div', include_plotlyjs=False)
    # return fig_div
    return fig


def make_plot_data(plot_list, showlegend=True):
    data = []
    for idx, plt in enumerate(plot_list):
        color = color_defaults[idx % len(color_defaults)]
        # x = list(range(len(plt.means)))
        x = list(plt.xs)
        y = list(plt.means)
        y_upper = list(plt.means + plt.stds)
        y_lower = list(plt.means - plt.stds)
        y_extras = []

        data.append(go.Scatter(
            x=x + x[::-1],
            y=y_upper + y_lower[::-1],
            fill='tozerox',
            fillcolor=hex_to_rgb(color, 0.2),
            line=go.Line(color='transparent'),
            showlegend=False,
            legendgroup=plt.legend,
            hoverinfo='none'
        ))
        data.append(go.Scatter(
            x=x,
            y=y,
            name=plt.legend,
            legendgroup=plt.legend,
            line=dict(color=hex_to_rgb(color)),
            showlegend=showlegend,
        ))

        for y_extra in y_extras:
            data.append(go.Scatter(
                x=x,
                y=y_extra,
                showlegend=False,
                legendgroup=plt.legend,
                line=dict(color=hex_to_rgb(color), dash='dot')
                # choices: solid, dot, dash, longdash, dashdot, longdashdot
            ))

    return data






def load_data(exp_folder_path, disable_variant=False, ignore_missing_keys=False):
    global exps_data
    global plottable_keys
    global distinct_params
    exps_data = load_exps_data([exp_folder_path], disable_variant, ignore_missing_keys)
    plottable_keys = sorted(list(
        set(flatten(list(exp.progress.keys()) for exp in exps_data))))
    distinct_params = sorted(extract_distinct_params(exps_data))
    return exps_data, plottable_keys, distinct_params


def load_exps_data(exp_folder_paths, disable_variant=False, ignore_missing_keys=False):
    exps = []
    for exp_folder_path in exp_folder_paths:
        exps += [x[0] for x in os.walk(exp_folder_path)]
    print("finished walking exp folders")
    exps_data = []
    for exp in exps:
        try:
            exp_path = exp
            params_json_path = os.path.join(exp_path, "params.json")
            variant_json_path = os.path.join(exp_path, "variant.json")
            progress_csv_path = os.path.join(exp_path, "progress.csv")
            progress = load_progress(progress_csv_path)
            if disable_variant:
                params = load_params(params_json_path)
            else:
                try:
                    params = load_params(variant_json_path)
                except IOError:
                    params = load_params(params_json_path)
            exps_data.append(ext.AttrDict(
                progress=progress, params=params, flat_params=flatten_dict(params)))
        except IOError as e:
            print(e)

    # a dictionary of all keys and types of values
    all_keys = dict()
    for data in exps_data:
        for key in data.flat_params.keys():
            if key not in all_keys:
                all_keys[key] = type(data.flat_params[key])

    # if any data does not have some key, specify the value of it
    if not ignore_missing_keys:
        default_values = dict()
        for data in exps_data:
            for key in sorted(all_keys.keys()):
                if key not in data.flat_params:
                    if key not in default_values:
                        default = input("Please specify the default value of \033[93m %s \033[0m: " % (key))
                        try:
                            if all_keys[key].__name__ == 'NoneType':
                                default = None
                            elif all_keys[key].__name__ == 'bool':
                                try:
                                    default = eval(default)
                                except:
                                    default = False
                            else:
                                default = all_keys[key](default)
                        except ValueError:
                            print("Warning: cannot cast %s to %s" % (default, all_keys[key]))
                        default_values[key] = default
                    data.flat_params[key] = default_values[key]

    return exps_data


def load_progress(progress_csv_path):
    print("Reading %s" % progress_csv_path)
    entries = dict()
    with open(progress_csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k, v in row.items():
                if k not in entries:
                    entries[k] = []
                try:
                    entries[k].append(float(v))
                except:
                    entries[k].append(0.)
    entries = dict([(k, np.array(v)) for k, v in entries.items()])
    return entries


def load_params(params_json_path):
    with open(params_json_path, 'r') as f:
        data = json.loads(f.read())
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-2]
    return data


def flatten_dict(d):
    flat_params = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = flatten_dict(v)
            for subk, subv in flatten_dict(v).items():
                flat_params[k + "." + subk] = subv
        else:
            flat_params[k] = v
    return flat_params


def extract_distinct_params(exps_data, excluded_params=('exp_name', 'seed', 'log_dir'), l=1):
    try:
        stringified_pairs = sorted(
            map(eval, unique(flatten(
                [list(map(smart_repr, list(d.flat_params.items()))) for d in exps_data]))),
            key=lambda x: (tuple("" if it is None else str(it) for it in x),))
    except Exception as e:
        print(e)
    proposals = [(k, [x[1] for x in v])
                 for k, v in itertools.groupby(stringified_pairs, lambda x: x[0])]
    filtered = [(k, v) for (k, v) in proposals if len(v) > l and all(
        [k.find(excluded_param) != 0 for excluded_param in excluded_params])]
    return filtered


def unique(l):
    return list(set(l))


def flatten(l):
    return [item for sublist in l for item in sublist]


def smart_repr(x):
    if isinstance(x, tuple):
        if len(x) == 0:
            return "tuple()"
        elif len(x) == 1:
            return "(%s,)" % smart_repr(x[0])
        else:
            return "(" + ",".join(map(smart_repr, x)) + ")"
    else:
        if hasattr(x, "__call__"):
            return "__import__('pydoc').locate('%s')" % (x.__module__ + "." + x.__name__)
        else:
            return repr(x)


class Selector(object):
    def __init__(self, exps_data, filters=None, custom_filters=None):
        self._exps_data = exps_data
        if filters is None:
            self._filters = tuple()
        else:
            self._filters = tuple(filters)
        if custom_filters is None:
            self._custom_filters = []
        else:
            self._custom_filters = custom_filters

    def where(self, k, v):
        return Selector(self._exps_data, self._filters + ((k, v),), self._custom_filters)

    def custom_filter(self, filter):
        return Selector(self._exps_data, self._filters, self._custom_filters + [filter])

    def _check_exp(self, exp):
        # or exp.flat_params.get(k, None) is None
        return all(
            ((str(exp.flat_params.get(k, None)) == str(v) or (k not in exp.flat_params)) for k, v in self._filters)
        ) and all(custom_filter(exp) for custom_filter in self._custom_filters)

    def extract(self):
        return list(filter(self._check_exp, self._exps_data))

    def iextract(self):
        return filter(self._check_exp, self._exps_data)


def check_nan(exp):
    return all(not np.any(np.isnan(vals)) for vals in list(exp.progress.values()))


def summary_name(exp, selector=None):
    return exp.params["exp_name"]


def sliding_mean(data_array, window=5):
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = list(range(max(i - window + 1, 0),
                             min(i + window + 1, len(data_array))))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


# Taken from plot.ly
color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]


def hex_to_rgb(hex, opacity=1.0):
    if hex[0] == '#':
        hex = hex[1:]
    assert (len(hex) == 6)
    return "rgba({0},{1},{2},{3})".format(int(hex[:2], 16), int(hex[2:4], 16), int(hex[4:6], 16), opacity)
