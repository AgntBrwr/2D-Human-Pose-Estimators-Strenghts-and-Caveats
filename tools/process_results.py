import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = "".join([os.getcwd()[:-6], '\\results\\results_new_labels.xlsx'])

full = pd.read_excel(path, sheet_name="Full", index_col=0)
occlusion = pd.read_excel(path, sheet_name="Occlusion", index_col=0)
occlusion_type = pd.read_excel(path, sheet_name="Occlusion type", index_col=0)
truncation = pd.read_excel(path, sheet_name="Truncation", index_col=0)
grouping = pd.read_excel(path, sheet_name="Grouping", index_col=0)
person_size = pd.read_excel(path, sheet_name="Resolution", index_col=0)


def rename_methods(method):
    method.rename(index={"ResNet_50": "ResNet-50", "ResNet_101": "ResNet-101", "ResNet_152": "ResNet-152",
                         "HRNet_W32": "HRNet-W32",
                         "HRNet_W48": "HRNet-W48", "DEKR_w32_ms": "DEKR-W32 ms", "DEKR_w32": "DEKR-W32",
                         "DEKR_w48_ms": "DEKR-W48 ms", "DEKR_w48": "DEKR-W48", "DarkPose_w32": "DarkPose-W32",
                         "DarkPose_w48": "DarkPose-W48"}, inplace=True)



rename_methods(full)
rename_methods(occlusion)
rename_methods(occlusion_type)
rename_methods(truncation)
rename_methods(grouping)
rename_methods(person_size)


def create_graph(sheet, graph, cta):
    methods = []
    aps = []
    variables = []

    for ind, method in enumerate(sheet.index):
        if (method == "ResNet-152" and sheet["Input size"].values[ind] == "384x288" or (method == "HRNet-W48" and
                                                                                        sheet["Input size"].values[
                                                                                            ind] == "384x288") or (
                method == "DEKR-W48 ms" and
                sheet["Input size"].values[ind] == "640x640") or
                (method == "DarkPose-W48" and sheet["Input size"].values[ind] == "384x288")):
            # print(method, sheet["Variable"].values[ind], sheet["AP"].values[ind])
            '''if method == "DEKR-W48 ms":
                method = "DEKR-W48"'''
            methods.append(method)  # + " (" + sheet["Input size"].values[ind] + ")")
            variables.append(sheet["Variable"].values[ind])
            aps.append(sheet["AP"].values[ind])

    methods = list(dict.fromkeys(methods))
    amount = int(len(aps) / len(methods))
    variables = list(dict.fromkeys(variables))
    scores_dict = {}
    mi_ = 1.0  # min value variable
    ma_ = 0.0  # max value variable
    for j in range(0, amount):
        scores_dict[variables[j]] = aps[j::amount]
        if mi_ > min(aps[j::amount]):
            mi_ = min(aps[j::amount])
        if ma_ < max(aps[j::amount]):
            ma_ = max(aps[j::amount])
    if graph == "bargraph":
        print(scores_dict)
        df = pd.DataFrame(scores_dict, index=methods)
        print(df)
        # print(df)
        sns.set()

        ax = df.plot.bar(rot=0, alpha=0.7)
        plt.xlabel("Methods", fontdict={'fontsize': 28})
        plt.ylabel("Average Precision", fontdict={'fontsize': 28})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        if mi_ - (ma_ - mi_) > 0:
            ax.set_ylim(mi_ - (ma_ - mi_))
        else:
            ax.set_ylim(0, 1)
        plt.legend(prop={"size": 28}, ncol=4)
        sns.set()
        sns.color_palette("mako", as_cmap=True)
        plt.show()
    elif graph == "piechart":
        methods = {"methods": methods}
        print(methods)
        print(scores_dict)
        df = pd.DataFrame(scores_dict, methods)
        sns.set(palette="gist_earth")
        print(f"df: {df}")
        df.groupby(['-']).sum().plot(kind="pie")


def result_differences(model_name, input_size, challenge):
    AP = "AP"
    model = challenge.loc[model_name]
    model = model.loc[model["Input size"] == input_size]
    if model["Variable"].iloc[0] == 'keyp==0':
        model = model.loc[model["Variable"] == 'keyp==0'][AP] - model.loc[model["Variable"] == 'keyp>0'][AP]
        print("SKR)")
        print(round(model.iloc[0], 3), model_name)
        return model.iloc[0], 0, 0, 0
    if model["Variable"].iloc[0] == 'Visible':
        model = model.loc[model["Variable"] == 'Visible'][AP] - model.loc[model["Variable"] == 'Occluded'][AP]
        print(round(model.iloc[0], 3), model_name)
        return model.iloc[0]
    if model["Variable"].iloc[0] == 'Self':
        se = model.loc[model["Variable"] == 'Self'][AP] - model.loc[model["Variable"] == 'Environment'][AP]
        sp = model.loc[model["Variable"] == 'Self'][AP] - model.loc[model["Variable"] == 'Person'][AP]
        ep = model.loc[model["Variable"] == 'Environment'][AP] - model.loc[model["Variable"] == 'Person'][AP]
        print(round(se.iloc[0], 3), round(sp.iloc[0], 3), round(ep.iloc[0], 3), model_name)
        return se.iloc[0], sp.iloc[0], ep.iloc[0]
    if model["Variable"].iloc[0] == 'area<32^2':
        se = model.loc[model["Variable"] == 'area<32^2'][AP] - model.loc[model["Variable"] == '96^2<area'][AP]
        sp = model.loc[model["Variable"] == 'area<32^2'][AP] - model.loc[model["Variable"] == '32^2<area<96^2'][AP]
        ep = model.loc[model["Variable"] == '96^2<area'][AP] - model.loc[model["Variable"] == '32^2<area<96^2'][AP]
        print(round(se.iloc[0], 3), round(sp.iloc[0], 3), round(ep.iloc[0], 3), model_name)
        return se.iloc[0], sp.iloc[0], ep.iloc[0]
    if model["Variable"].iloc[0] == 'keyp==0':
        se = model.loc[model["Variable"] == '5>keyp>0'][AP] - model.loc[model["Variable"] == '5<=keyp<9'][AP]
        sp = model.loc[model["Variable"] == '5<=keyp<9'][AP] - model.loc[model["Variable"] == '9<=keyp<13'][AP]
        ep = model.loc[model["Variable"] == '9<=keyp<13'][AP] - model.loc[model["Variable"] == '13<=keyp'][AP]
        print(round(se.iloc[0], 3), round(sp.iloc[0], 3), round(ep.iloc[0], 3), model_name)
        return se.iloc[0], sp.iloc[0], ep.iloc[0], 0


def execute_results_differences():
    ro = result_differences("ResNet-152", "384x288", occlusion)
    ho = result_differences("HRNet-W48", "384x288", occlusion)
    dp = result_differences("DarkPose-W48", "384x288", occlusion)
    de = result_differences("DEKR-W48 ms", "640x640", occlusion)
    occl_res_ = np.array([ro, ho, dp, de])
    print(f"mean for occlusion is: {np.mean(occl_res_)} & std: {np.std(occl_res_)}")
    print("\n")

    r1, r2, r3 = result_differences("ResNet-152", "384x288", occlusion_type)
    h1, h2, h3 = result_differences("HRNet-W48", "384x288", occlusion_type)
    dp1, dp2, dp3 = result_differences("DarkPose-W48", "384x288", occlusion_type)
    de1, de2, de3 = result_differences("DEKR-W48 ms", "640x640", occlusion_type)

    seen = np.array([r1, h1, dp1, de1])
    sepe = np.array([r2, h2, dp2, de2])
    enpe = np.array([r3, h3, dp3, de3])
    print(
        f"The difference between Self and Environment: mean = {round(np.mean(seen), 3)} std = {round(np.std(seen), 3)}")
    print(f"The difference between Self and Person: mean = {round(np.mean(sepe), 3)} std = {round(np.std(sepe), 3)}")
    print(f"The difference between Env and Person: mean = {round(np.mean(enpe), 3)} std = {round(np.std(enpe), 3)}")

    r1, r2, r3 = result_differences("ResNet-152", "384x288", person_size)
    h1, h2, h3 = result_differences("HRNet-W48", "384x288", person_size)
    dp1, dp2, dp3 = result_differences("DarkPose-W48", "384x288", person_size)
    de1, de2, de3 = result_differences("DEKR-W48 ms", "640x640", person_size)

    seen = np.array([r1, h1, dp1, de1])
    sepe = np.array([r2, h2, dp2, de2])
    enpe = np.array([r3, h3, dp3, de3])
    print(f"The difference between Small and Large: mean = {round(np.mean(seen), 3)} std = {round(np.std(seen), 3)}")
    print(f"The difference between Small and Medium: mean = {round(np.mean(sepe), 3)} std = {round(np.std(sepe), 3)}")
    print(f"The difference between Medium and Large: mean = {round(np.mean(enpe), 3)} std = {round(np.std(enpe), 3)}")

    r1, r2, r3, r4 = result_differences("ResNet-152", "384x288", truncation)
    h1, h2, h3, h4 = result_differences("HRNet-W48", "384x288", truncation)
    dp1, dp2, dp3, dp4 = result_differences("DarkPose-W48", "384x288", truncation)
    de1, de2, de3, de4 = result_differences("DEKR-W48 ms", "640x640", truncation)

    seen = np.array([r1, h1, dp1, de1])
    sepe = np.array([r2, h2, dp2, de2])
    enpe = np.array([r3, h3, dp3, de3])
    print(
        f"The difference between bin 1 and bin 2: mean = {round(np.mean(seen), 3)} std = {round(np.std(seen), 3)}")
    print(
        f"The difference between bin 2 and bin 3: mean = {round(np.mean(sepe), 3)} std = {round(np.std(sepe), 3)}")
    print(
        f"The difference between bin 3 and bin 4: mean = {round(np.mean(enpe), 3)} std = {round(np.std(enpe), 3)}")


# execute_results_differences()


def create_latex_table(df):
    return df.to_latex(float_format="{:0.3f}".format, escape=False)


def mark_down_table(df):
    return df.to_markdown()


def mark_down_tables():
    full.drop(["Multi-scale", "Subset", "Variable", "APL", "APM", "ARL", "ARM"], axis=1, inplace=True)
    print(mark_down_table(full))

    occlusion.drop(["Multi-scale", "Subset", "APL", "APM", "ARL", "ARM"], axis=1, inplace=True)
    print(mark_down_table(occlusion))

    occlusion_type.drop(["Multi-scale", "Subset", "APL", "APM", "ARL", "ARM"], axis=1, inplace=True)
    print(mark_down_table(occlusion_type))

    truncation.drop(["Multi-scale", "Subset", "APL", "APM", "ARL", "ARM"], axis=1, inplace=True)
    truncation.replace({"keyp==0": "k=0", "5>keyp>0": "0<k<5", "5<=keyp<9": "5<=k<9", "9<=keyp<13":
        "9<=k<13", "13<=keyp": "13<=k", "keyp>0": "0<k"}, inplace=True)

    truth_cond_truncation1 = ["0<k<5", "5<=k<9", "9<=k<13", "13<=k"]
    truth_cond_truncation2 = ["k=0", "0<k"]
    truncation_subset = truncation[truncation["Variable"].isin(truth_cond_truncation1)]
    print(mark_down_table(truncation_subset))
    truncation_subset = truncation[truncation["Variable"].isin(truth_cond_truncation2)]
    print(mark_down_table(truncation_subset))

    person_size.drop(["Multi-scale", "Subset", "APL", "APM", "ARL", "ARM"], axis=1, inplace=True)
    person_size.replace({"area<32^2": "Low", "32^2<area<96^2": "Medium", "96^2<area": "High"},
                        inplace=True)
    print(mark_down_table(person_size))


# mark_down_tables()

def create_ap_graph(data_aps, aps, general, tit):
    sns.set()
    print(len(data_aps))
    r = 0
    tit_f = 24
    if general == 0:
        fig, axes = plt.subplots(1, 1, figsize=(15, 15))
        df = pd.DataFrame(data_aps, index=aps)
        sns.lineplot(
            data=df,  # df.head(5)
            markers=True, dashes=False, linewidth=5
        )
        plt.xlabel("OKS Threshold", fontdict={'fontsize': 28})
        plt.xticks(np.arange(min(aps), max(aps) + 0.05, 0.05), fontsize=24)
        plt.ylabel("Average Precision", fontdict={'fontsize': 28})
        plt.yticks(fontsize=24)
        plt.legend(prop={"size": 24})  # , loc='lower left')
        tit[0] = "AP at .0.5 : .05 : .95"
        plt.title(tit[0], fontsize=tit_f)
    elif general == 1 or 2:
        fig, axes = plt.subplots(1, len(data_aps), figsize=(15, 15))
        if len(data_aps) == 2:
            df_v, df_o = pd.DataFrame(data_aps[0], index=aps), pd.DataFrame(data_aps[1], index=aps)
        else:
            df_v, df_o, df_q = pd.DataFrame(data_aps[0], index=aps), pd.DataFrame(data_aps[1], index=aps), pd.DataFrame(
                data_aps[2], index=aps)
        sns.lineplot(ax=axes[0], data=df_v.head(8),
                     markers=True, dashes=False, linewidth=5
                     )
        plt.sca(axes[0])
        axes[0].set_ylim([0.7, 1.0])
        # axes[0].set_xlim([0.49, 0.81])
        plt.xlabel("OKS Threshold", fontdict={'fontsize': 28})
        plt.xticks(np.arange(min(aps[r:]), max(aps[r:]) + 0.05, 0.1), fontsize=24)
        plt.ylabel("Average Precision", fontdict={'fontsize': 28})
        plt.yticks(fontsize=24)
        plt.legend(prop={"size": 24})
        plt.title(tit[0], fontsize=tit_f)

        sns.lineplot(ax=axes[1], data=df_o.head(8),
                     markers=True, dashes=False, linewidth=5
                     )
        plt.sca(axes[1])
        axes[1].set_ylim([0.7, 1.0])
        # axes[1].set_xlim([0.49, 0.81])
        plt.xlabel("OKS Threshold", fontdict={'fontsize': 28})
        plt.xticks(np.arange(min(aps[r:]), max(aps[r:]) + 0.05, 0.1), fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(prop={"size": 24})
        plt.title(tit[1], fontsize=tit_f)

        if len(data_aps) > 2:
            sns.lineplot(ax=axes[2], data=df_q,
                         markers=True, dashes=False, linewidth=5
                         )
            plt.sca(axes[2])
            axes[2].set_ylim([0.0, 1.0])
            plt.xlabel("OKS Threshold", fontdict={'fontsize': 28})
            plt.xticks(np.arange(min(aps[r:]), max(aps[r:]) + 0.05, 0.1), fontsize=24)
            plt.yticks(fontsize=24)
            plt.legend(prop={"size": 24})
            plt.title(tit[2], fontsize=tit_f)
    plt.show()


def latex_tables():
    full.drop(["Multi-scale", "Subset", "Variable", "APL", "APM", "ARL", "ARM"], axis=1, inplace=True)
    print("full", create_latex_table(full))

    # occlusion:
    occlusion.drop(["Multi-scale", "Subset", "APL", "APM", "ARL", "ARM"], axis=1, inplace=True)
    print("occlusion", create_latex_table(occlusion))

    # occlusion type
    occlusion_type.drop(["Multi-scale", "Subset", "APL", "APM", "ARL", "ARM"], axis=1, inplace=True)
    print("occlusion type", create_latex_table(occlusion_type))

    # truncation
    truncation.drop(["Multi-scale", "Subset", "APL", "APM", "ARL", "ARM"], axis=1, inplace=True)
    # print(create_latex_table(truncation))
    truncation.replace({"keyp==0": "$k=0$", "5>keyp>0": "$0<k<5$", "5<=keyp<9": "$5<=k<9$", "9<=keyp<13":
        "$9<=k<13$", "13<=keyp": "$13<=k$", "keyp>0": "$0<k$"}, inplace=True)
    print(create_latex_table(truncation))

    # grouping
    grouping.drop(["Multi-scale", "Subset", "APL", "APM", "ARL", "ARM"], axis=1, inplace=True)
    print(create_latex_table(grouping))

    # Resolution
    person_size.drop(["Multi-scale", "Subset", "APL", "APM", "ARL", "ARM"], axis=1, inplace=True)
    person_size.replace({"area<32^2": "Small", "32^2<area<96^2": "Medium", "96^2<area": "Large"},
                        inplace=True)
    print("person size", create_latex_table(person_size))


Thresholds = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
ResNet = [0.905, 0.897, 0.889, 0.874, 0.853, 0.821, 0.772, 0.689, 0.556, 0.258]
HRNet = [0.914, 0.91, 0.901, 0.89, 0.868, 0.837, 0.796, 0.719, 0.584, 0.293]
DEKR = [0.892, 0.884, 0.873, 0.856, 0.833, 0.796, 0.744, 0.662, 0.524, 0.253]
DarkPose = [0.914, 0.91, 0.901, 0.89, 0.871, 0.843, 0.799, 0.725, 0.598, 0.324]

ResNet_og = [0.896, 0.888, 0.879, 0.864, 0.843, 0.811, 0.762, 0.682, 0.55, 0.255]
HRNet_og = [0.908, 0.901, 0.891, 0.88, 0.858, 0.829, 0.786, 0.71, 0.576, 0.29]
DEKR_og = [0.882, 0.874, 0.863, 0.845, 0.822, 0.786, 0.734, 0.654, 0.516, 0.25]
DarkPose_og = [0.907, 0.902, 0.892, 0.88, 0.862, 0.833, 0.789, 0.716, 0.591, 0.32]

ResNet_visible = [0.915, 0.908, 0.902, 0.893, 0.879, 0.856, 0.823, 0.764, 0.65, 0.349]
HRNet_visible = [0.922, 0.919, 0.912, 0.903, 0.89, 0.869, 0.841, 0.784, 0.684, 0.389]
DEKR_visible = [0.902, 0.896, 0.887, 0.876, 0.862, 0.839, 0.796, 0.73, 0.625, 0.348]
DarkPose_visible = [0.923, 0.92, 0.912, 0.903, 0.891, 0.872, 0.845, 0.787, 0.691, 0.426]

ResNet_occluded = [0.803, 0.772, 0.744, 0.698, 0.637, 0.577, 0.506, 0.408, 0.279, 0.114]
HRNet_occluded = [0.825, 0.798, 0.763, 0.718, 0.67, 0.611, 0.535, 0.43, 0.294, 0.118]
DEKR_occluded = [0.782, 0.751, 0.712, 0.669, 0.615, 0.553, 0.483, 0.384, 0.249, 0.097]
DarkPose_occluded = [0.833, 0.805, 0.773, 0.734, 0.681, 0.624, 0.55, 0.447, 0.314, 0.126]

ResNet_fd = [0.895, 0.885, 0.873, 0.856, 0.828, 0.791, 0.73, 0.628, 0.466, 0.161]
ResNet_no_fd = [0.895, 0.886, 0.871, 0.852, 0.828, 0.786, 0.725, 0.626, 0.464, 0.158]
DEKR_fd = [0.888, 0.878, 0.864, 0.843, 0.816, 0.781, 0.724, 0.634, 0.487, 0.222]
HRNet_fd = [0.909, 0.903, 0.893, 0.878, 0.857, 0.826, 0.774, 0.688, 0.542, 0.223]

HRNet_no_fd = [0.909, 0.902, 0.893, 0.876, 0.855, 0.825, 0.773, 0.688, 0.543, 0.227]
DarkPose_fd = [0.909, 0.904, 0.894, 0.883, 0.863, 0.833, 0.786, 0.707, 0.57, 0.295]
DEKR_no_fd = [0.884, 0.873, 0.858, 0.837, 0.812, 0.77, 0.714, 0.624, 0.477, 0.219]
DarkPose_no_fd = [0.91, 0.906, 0.896, 0.884, 0.863, 0.834, 0.786, 0.706, 0.575, 0.287]

ResNet_no_trunc = [0.923, 0.919, 0.912, 0.899, 0.88, 0.85, 0.804, 0.726, 0.591, 0.286]
HRNet_no_trunc = [0.929, 0.926, 0.918, 0.912, 0.891, 0.86, 0.825, 0.751, 0.616, 0.321]
DEKR_no_trunc = [0.917, 0.91, 0.901, 0.889, 0.866, 0.828, 0.785, 0.708, 0.571, 0.289]
DarkPose_no_trunc = [0.929, 0.926, 0.92, 0.91, 0.894, 0.868, 0.825, 0.758, 0.631, 0.352]

ResNet_trunc = [0.783, 0.769, 0.758, 0.742, 0.719, 0.683, 0.635, 0.559, 0.447, 0.192]
HRNet_trunc = [0.809, 0.799, 0.785, 0.771, 0.748, 0.717, 0.671, 0.595, 0.477, 0.226]
DEKR_trunc = [0.715, 0.704, 0.689, 0.673, 0.649, 0.617, 0.56, 0.48, 0.367, 0.159]
DarkPose_trunc = [0.808, 0.8, 0.787, 0.774, 0.749, 0.718, 0.678, 0.596, 0.488, 0.253]

ResNet_self = [0.829, 0.809, 0.794, 0.767, 0.721, 0.671, 0.599, 0.503, 0.367, 0.153]
HRNet_self = [0.849, 0.834, 0.802, 0.77, 0.736, 0.69, 0.627, 0.521, 0.372, 0.158]
DEKR_self = [0.811, 0.787, 0.759, 0.733, 0.688, 0.642, 0.577, 0.478, 0.333, 0.13]
DarkPose_self = [0.849, 0.834, 0.807, 0.782, 0.745, 0.7, 0.637, 0.535, 0.393, 0.164]

ResNet_p = [0.619, 0.585, 0.553, 0.507, 0.466, 0.403, 0.352, 0.278, 0.194, 0.088]
HRNet_p = [0.639, 0.607, 0.574, 0.531, 0.476, 0.424, 0.363, 0.289, 0.202, 0.094]
DEKR_p = [0.599, 0.568, 0.532, 0.488, 0.431, 0.374, 0.313, 0.243, 0.174, 0.075]
DarkPose_p = [0.65, 0.621, 0.576, 0.535, 0.488, 0.44, 0.376, 0.304, 0.202, 0.093]

ResNet_e = [0.742, 0.704, 0.67, 0.625, 0.571, 0.515, 0.455, 0.372, 0.249, 0.11]
HRNet_e = [0.771, 0.741, 0.707, 0.665, 0.626, 0.566, 0.5, 0.403, 0.277, 0.124]
DEKR_e = [0.75, 0.72, 0.685, 0.644, 0.596, 0.523, 0.45, 0.368, 0.247, 0.116]
DarkPose_e = [0.775, 0.748, 0.72, 0.685, 0.637, 0.581, 0.505, 0.422, 0.289, 0.136]

'''r = 1
self_np = np.array([ResNet_e[r], HRNet_e[r], DEKR_e[r], DarkPose_e[r]])
person_np = np.array([ResNet_p[r], HRNet_p[r], DEKR_p[r], DarkPose_p[r]])

r = -1
self_np = np.array([ResNet_e[r], HRNet_e[r], DEKR_e[r], DarkPose_e[r]])
person_np = np.array([ResNet_p[r], HRNet_p[r], DEKR_p[r], DarkPose_p[r]])'''


def make_ap_graphs(type_ap):
    if type_ap == 0:
        title = ["(a): AP at .0.5 : .05 : .7,", "(b): AP at .0.75 : .05 : .95"]
        data = {"ResNet-152": ResNet,
                "HRNet-W48": HRNet,
                "DEKR-W48": DEKR,
                "DarkPose-W48": DarkPose}
        create_ap_graph(data, Thresholds, type_ap, title)
    elif type_ap == 1:
        title = ["(a) visible", "(b) occluded"]
        data_vis = {"ResNet-152": ResNet_visible,
                    "HRNet-W48": HRNet_visible,
                    "DEKR-W48": DEKR_visible,
                    "DarkPose-W48": DarkPose_visible}
        data_occl = {"ResNet-152": ResNet_occluded,
                     "HRNet-W48": HRNet_occluded,
                     "DEKR-W48": DEKR_occluded,
                     "DarkPose-W48": DarkPose_occluded}
        data = [data_vis, data_occl]
        create_ap_graph(data, Thresholds, type_ap, title)
    elif type_ap == 2:
        title = ["(a) filtered data", "(b) original data"]
        data_fd = {"ResNet-152": ResNet_fd,
                   "HRNet-W48": HRNet_fd,
                   "DEKR-W48": DEKR_fd,
                   "DarkPose-W48": DarkPose_fd}
        data_no_fd = {"ResNet-152": ResNet_no_fd,
                      "HRNet-W48": HRNet_no_fd,
                      "DEKR-W48": DEKR_no_fd,
                      "DarkPose-W48": DarkPose_no_fd}
        data = [data_fd, data_no_fd]
        # print(data)
        create_ap_graph(data, Thresholds, type_ap, title)
    elif type_ap == 3:
        title = ["(a) without truncation", "(b) with truncation"]
        data_no_trunc = {"ResNet-152": ResNet_no_trunc,
                         "HRNet-W48": HRNet_no_trunc,
                         "DEKR-W48": DEKR_no_trunc,
                         "DarkPose-W48": DarkPose_no_trunc}
        data_trunc = {"ResNet-152": ResNet_trunc,
                      "HRNet-W48": HRNet_trunc,
                      "DEKR-W48": DEKR_trunc,
                      "DarkPose-W48": DarkPose_trunc}
        data = [data_no_trunc, data_trunc]
        # print(data)
        create_ap_graph(data, Thresholds, type_ap, title)
    elif type_ap == 4:
        title = ["(a) self", "(b) other person", "(c) environment"]
        self = {"ResNet-152": ResNet_self,
                "HRNet-W48": HRNet_self,
                "DEKR-W48": DEKR_self,
                "DarkPose-W48": DarkPose_self}
        person = {"ResNet-152": ResNet_p,
                  "HRNet-W48": HRNet_p,
                  "DEKR-W48": DEKR_p,
                  "DarkPose-W48": DarkPose_p}
        environment = {"ResNet-152": ResNet_e,
                       "HRNet-W48": HRNet_e,
                       "DEKR-W48": DEKR_e,
                       "DarkPose-W48": DarkPose_e}
        data = [self, person, environment]
        # print(data)
        create_ap_graph(data, Thresholds, type_ap, title)
