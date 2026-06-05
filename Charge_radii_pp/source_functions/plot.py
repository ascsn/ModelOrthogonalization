import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

from config import alpha_train, alpha_validation, alpha_test, color_train, color_validation, color_test, marker_train, marker_validation, marker_test, colors


def plot_filtered_supermodel(supermodel_predictions_range, filtered_models_output_list, Z, unit, property, Constraint, title):
    plt.rc("xtick", labelsize=30)
    plt.rc("ytick", labelsize=30)

    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    
    lower, median, upper = supermodel_predictions_range

    filtered_models_output= filtered_models_output_list[0]
    filtered_models_output_train= filtered_models_output_list[1]
    filtered_models_output_validation= filtered_models_output_list[2]
    filtered_models_output_test = filtered_models_output_list[3]
    filtered_models_output_stable = filtered_models_output_list[4]
    
    fig, ax = plt.subplots(figsize=(10,8), dpi=150)
    
    plt.plot(filtered_models_output["N"], median, color="darkkhaki", label=f'$f^\dagger$({Constraint})',linewidth=3)
    
    plt.plot(filtered_models_output["N"], lower, color="darkkhaki",linestyle="dashed",linewidth=2,alpha=0.5)
    plt.plot(filtered_models_output["N"], upper, color="darkkhaki",linestyle="dashed",linewidth=2,alpha=0.5)
    
    plt.fill_between(filtered_models_output["N"], lower, upper, color="darkkhaki",alpha=0.3)
    
    color_local= "khaki"
    
    
    # plt.plot(filtered_models_output["N"], -median_simplex_local/filtered_models_output['A'], color='purple', label='$f^\dagger$(Simplex Local)',linewidth=3)
    
    # # plt.plot(filtered_models_output["N"], lower, color=color_simplex,linestyle="dashed",linewidth=3,alpha=0.8)
    # # plt.plot(filtered_models_output["N"], upper, color=color_simplex,linestyle="dashed",linewidth=3,alpha=0.8)
    
    
    # plt.plot(filtered_models_output["N"], -lower_simplex_local/filtered_models_output['A'], color="purple",linestyle="dashed",linewidth=2,alpha=0.8)
    # plt.plot(filtered_models_output["N"], -upper_simplex_local/filtered_models_output['A'], color="purple",linestyle="dashed",linewidth=2,alpha=0.8)
    
    # plt.fill_between(filtered_models_output["N"], -lower_simplex_local/filtered_models_output['A'], -upper_simplex_local/filtered_models_output['A'], color='purple',alpha=0.3)
    
    ax.scatter(x = filtered_models_output_train["N"], y = filtered_models_output_train['truth'], label = "$\mathcal{X}_0^{tr}$",  alpha = alpha_train,color=color_train,s=100,marker=marker_train,zorder=2)
    
    ax.scatter(x = filtered_models_output_validation["N"], y = filtered_models_output_validation['truth'], label = "$\mathcal{X}_0^{va}$", alpha=alpha_validation ,color=color_validation,s=100,marker=marker_validation,zorder=2)

    ax.scatter(x = filtered_models_output_test["N"], y = filtered_models_output_test['truth'], label = "$\mathcal{X}_0^{te}$", alpha = alpha_test,color=color_test,s=100,marker=marker_test,zorder=2)
    
    ax.scatter(x = filtered_models_output_stable["N"], y = filtered_models_output_stable['truth'], label = "Stable", alpha = 0.9,color='k',s=80,marker="s",zorder=2)
    
    
    
    
    
    plt.xlabel("Neutrons",fontsize=35)
    plt.ylabel(f"(Z= {Z}) $ {{\\cal {property}}}$ [{unit}]", fontsize=33)
    # plt.ylabel(Selected_element_name+ " BE/A MeV",fontsize=25)
     
    plt.legend(fontsize=20,markerscale=1,ncol=2,columnspacing=0.5)

    plt.title(f'Radius Calibration with {title} component(s)', fontsize = 25)
    # plt.savefig(f'{save_fig}')
    # plt.show()

def plot_filtered_supermodel_simplex(supermodel_predictions_range, filtered_models_output_list, Z, unit, property, Constraint, title):
    plt.rc("xtick", labelsize=30)
    plt.rc("ytick", labelsize=30)

    plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    
    lower, median, upper = supermodel_predictions_range

    filtered_models_output= filtered_models_output_list[0]
    filtered_models_output_train= filtered_models_output_list[1]
    filtered_models_output_validation= filtered_models_output_list[2]
    filtered_models_output_test = filtered_models_output_list[3]
    filtered_models_output_stable = filtered_models_output_list[4]
    
    fig, ax = plt.subplots(figsize=(10,8), dpi=150)
    
    plt.plot(filtered_models_output["N"], median, color="purple", label = r'$f^\dagger(\mathbf{x}, \mathbf{b})$',linewidth=3)
    
    plt.plot(filtered_models_output["N"], lower, color="purple",linestyle="dashed",linewidth=2,alpha=0.5)
    plt.plot(filtered_models_output["N"], upper, color="purple",linestyle="dashed",linewidth=2,alpha=0.5)
    
    plt.fill_between(filtered_models_output["N"], lower, upper, color="purple",alpha=0.3)
    
    color_local= "khaki"
    
    
    # plt.plot(filtered_models_output["N"], -median_simplex_local/filtered_models_output['A'], color='purple', label='$f^\dagger$(Simplex Local)',linewidth=3)
    
    # # plt.plot(filtered_models_output["N"], lower, color=color_simplex,linestyle="dashed",linewidth=3,alpha=0.8)
    # # plt.plot(filtered_models_output["N"], upper, color=color_simplex,linestyle="dashed",linewidth=3,alpha=0.8)
    
    
    # plt.plot(filtered_models_output["N"], -lower_simplex_local/filtered_models_output['A'], color="purple",linestyle="dashed",linewidth=2,alpha=0.8)
    # plt.plot(filtered_models_output["N"], -upper_simplex_local/filtered_models_output['A'], color="purple",linestyle="dashed",linewidth=2,alpha=0.8)
    
    # plt.fill_between(filtered_models_output["N"], -lower_simplex_local/filtered_models_output['A'], -upper_simplex_local/filtered_models_output['A'], color='purple',alpha=0.3)
    
    ax.scatter(x = filtered_models_output_train["N"], y = filtered_models_output_train['truth'], label = "$\mathcal{X}_0^{tr}$",  alpha = alpha_train,color=color_train,s=100,marker=marker_train,zorder=2)
    
    ax.scatter(x = filtered_models_output_validation["N"], y = filtered_models_output_validation['truth'], label = "$\mathcal{X}_0^{va}$", alpha=alpha_validation ,color=color_validation,s=100,marker=marker_validation,zorder=2)

    ax.scatter(x = filtered_models_output_test["N"], y = filtered_models_output_test['truth'], label = "$\mathcal{X}_0^{te}$", alpha = alpha_test,color=color_test,s=100,marker=marker_test,zorder=2)
    
    ax.scatter(x = filtered_models_output_stable["N"], y = filtered_models_output_stable['truth'], label = "Stable", alpha = 0.9,color='k',s=80,marker="s",zorder=2)
    
    
    
    
    
    plt.xlabel("Neutrons",fontsize=35)
    plt.ylabel(f"(Z= {Z}) $ {{\\cal {property}}}$ [{unit}]", fontsize=33)
    # plt.ylabel(Selected_element_name+ " BE/A MeV",fontsize=25)
     
    plt.legend(fontsize=25,markerscale=1,ncol=2,columnspacing=0.5)

    # plt.title(f'Radius Calibration with {title} component(s)', fontsize = 25)
    # plt.savefig(f'{save_fig}')
    # plt.show()

def plot_singular_values(S, title):
    """ 
    Plots the singular values S in log scale, normalized by the largest singular value S[0].
    """
    plt.rc("xtick", labelsize = 25)
    plt.rc("ytick", labelsize = 25)
    plt.rcParams['text.usetex'] = False
    fig, ax = plt.subplots(figsize = (6,5), dpi = 100)

    ax.scatter(np.arange(1, S.size ), S[:-1]/S[0], color = "purple", alpha = 0.8, s = 90, marker = 'o')
    ax.plot(np.arange(1, S.size ), S[:-1]/S[0], color = "purple", linewidth = 3)

    ax.set_xticks([0,2,4,6,8,10])
    plt.xlabel("Component $j$", fontsize = 25)
    plt.ylabel("Singular Value $S_j/S_1$", fontsize = 25)
    plt.yscale("log")
    plt.title(title, fontsize = 25)
    # plt.savefig('Plots/Radii Data with Bailey/Singular values/Singular Values with FyDr and FyIVP.png')
    plt.show()

def plot_model_predictions(filtered_models_output_list, models_selected, Selected_element_name):
    plt.rc("xtick", labelsize=30)
    plt.rc("ytick", labelsize=30)

    # plt.rcParams['text.usetex'] = False
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    filtered_models_output= filtered_models_output_list[0]
    filtered_models_output_train= filtered_models_output_list[1]
    filtered_models_output_validation= filtered_models_output_list[2]
    filtered_models_output_test = filtered_models_output_list[3]
    filtered_models_output_stable = filtered_models_output_list[4]
    
    fig, ax = plt.subplots(figsize=(10,8), dpi=150)
    i = 0
    for model in models_selected:
        plt.plot(filtered_models_output['N'], filtered_models_output[model],label = model, color=colors[i], alpha = 0.9,linewidth=5, zorder = 1)
        i+=1
    ax.scatter(x = filtered_models_output_train["N"], y = filtered_models_output_train['truth'], label = "Truth-Train",  alpha = 1,color=color_train,s=200,marker=marker_train,zorder=2)

    ax.scatter(x = filtered_models_output_validation["N"], y = filtered_models_output_validation['truth'], label = "Truth-Validation", alpha = 1,color=color_validation,s=350,marker=marker_validation,zorder=2)
    
    ax.scatter(x = filtered_models_output_test["N"], y = filtered_models_output_test['truth'], label = "Truth-Test", alpha = 1,color=color_test,s=350,marker=marker_test,zorder=2)
    
    ax.scatter(x = filtered_models_output_stable["N"], y = filtered_models_output_stable['truth'], label = "Truth-Stable", alpha = 1,color='k',s=80,marker="s",zorder=2)
    
    
    
    plt.xlabel("N",fontsize=35)
    plt.ylabel(Selected_element_name+" R(fm)",fontsize=35)
    # plt.legend(fontsize=18,markerscale=1 )
    plt.legend(fontsize=15,markerscale=1,ncol=2  )
    # plt.savefig(f'{savefig}')
    # plt.show()

""" 
Need to plot RMSE as a function of the number of components kept in the supermodel.
"""
def plot_pc_rmse_curve(
    rmse_df,
    title='RMSE vs PCs kept',
    figsize=(10, 8),
    dpi=100,
    fontsize=25
):
    """ 
    rmse_df should have columns: 'n_components', 'train', 'validation', 'test' and will be extracted from metrics.py
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')

    ax.plot(
        rmse_df['n_components'],
        rmse_df['train'],
        marker='o',
        linewidth=3,
        markersize=12,
        label=r'$f^\dagger(\mathcal{X}_0^{tr})$',
        color = color_train
    )

    ax.plot(
        rmse_df['n_components'],
        rmse_df['validation'],
        marker='s',
        linewidth=3,
        markersize=12,
        label=r'$f^\dagger(\mathcal{X}_0^{va})$',
        color = color_validation
    )

    ax.plot(
        rmse_df['n_components'],
        rmse_df['test'],
        marker='^',
        linewidth=3,
        markersize=12,
        label=r'$f^\dagger(\mathcal{X}_0^{te})$',
        color = color_test
    )

    ax.set_xlabel(r'PCs kept $p$', fontsize=fontsize)
    ax.set_ylabel(r'RMSE [fm]', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    ax.tick_params(axis='both', labelsize=fontsize)
    ax.legend(fontsize=fontsize )

    ax.grid(alpha=0.25)
    plt.tight_layout()



"""
Need to plot the model weights
"""
def plot_model_weights(model_weights, models_selected, colors, components_kept):
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rcParams['text.usetex'] = True

    plt.figure(figsize=(10, 6), dpi=150)

    for i, model in enumerate(models_selected):
        plt.bar(
            model,
            np.mean(model_weights.T[i]),
            yerr=np.std(model_weights.T[i]),
            color=colors[i],
            capsize=5
        )

    plt.minorticks_off()

    plt.xlabel('Models', fontsize=15)
    plt.ylabel(r'$\omega_k$', fontsize=35)
    plt.xticks(fontsize=15, rotation='vertical')
    plt.yticks(fontsize=25)
    plt.grid(axis='y')

    plt.title(f'Models weights of unconstrained BMM with {components_kept} PCs',
        fontsize=20
    )

    plt.tight_layout()


""" 
Need to plot PCs projections
"""
def plot_model_pc_projections(Vt_hat_normalized, key_list, colors):
    plt.rc("xtick", labelsize=35)
    plt.rc("ytick", labelsize=35)

    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    texts = []

    for i, model in enumerate(key_list):
        x_end = Vt_hat_normalized.T[i][0]
        y_end = Vt_hat_normalized.T[i][1]

        ax.arrow(
            0, 0, x_end, y_end,
            head_width=0.02,
            head_length=0.05,
            fc=colors[i],
            ec=colors[i],
            width=0.007
        )

        text = ax.text(
            x_end, y_end, model,
            fontsize=20,
            ha='center',
            va='center',
            color='black'
        )
        texts.append(text)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))

    plt.xlabel(r'Projection on $\phi_1$', fontsize=35)
    plt.ylabel(r'Projection on $\phi_2$', fontsize=35)

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)

    plt.axis('equal')
    plt.tight_layout()

"""
Need to plot coverage
"""
def plot_coverage(
    percentiles,
    coverage_train,
    coverage_validation,
    coverage_test,
    color_train,
    color_validation,
    color_test,
    coverage_simplex_train=None,
    coverage_simplex_validation=None,
    coverage_simplex_test=None
):
    plt.rc("xtick", labelsize=22)
    plt.rc("ytick", labelsize=22)
    plt.rcParams['text.usetex'] = True

    plt.figure(figsize=(6, 6), dpi=200)

    width_line = 2

    # main curves
    plt.plot(percentiles, coverage_train, label='Train', color=color_train, linewidth=3)
    plt.plot(percentiles, coverage_validation, label='Validation', color=color_validation, linewidth=3)
    plt.plot(percentiles, coverage_test, label='Test', color=color_test, linewidth=3)

    # optional simplex curves
    if coverage_simplex_train is not None:
        plt.plot(percentiles, coverage_simplex_train, linestyle='dashed', color=color_train, linewidth=width_line)
        plt.plot(percentiles, coverage_simplex_validation, linestyle='dashed', color=color_validation, linewidth=width_line)
        plt.plot(percentiles, coverage_simplex_test, linestyle='dashed', color=color_test, linewidth=width_line)

    # reference line
    plt.plot(percentiles, percentiles, label='Reference', color='k', linewidth=3)

    plt.xlabel('Credible Interval (\%)', fontsize=22)
    plt.ylabel('Coverage (\%)', fontsize=22)
    plt.xticks(ticks=[0, 20, 40, 60, 80, 100])

    plt.legend(fontsize=20)
    plt.tight_layout()

"""
Need a function to plot heatmaps of BMC errors at each nuclei
"""
def plot_bmc_error_heatmap_scatter(error_df, threshold=0.05, cmap='coolwarm', absolute_error=False):
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    plt.rc("xtick", labelsize=25)
    plt.rc("ytick", labelsize=15)

    # Scatter plot of absolute error
    sc = ax.scatter(
        error_df['N'],
        error_df['Z'],
        c=error_df['abs_error'],
        cmap=cmap,
        vmin=0,
        vmax=error_df['abs_error'].max(),
        marker='s',          # square markers
        s=170,               # size of each square
        edgecolors='white',  # spacing/separation
        linewidths=0.6
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label('BMC absolute error (fm)', fontsize=25)

    # Outliers
    if absolute_error:
        outliers = error_df[error_df['abs_error'] > threshold]
        label = f'error $> {threshold:.3f}$ fm'
    else:
        outliers = error_df[error_df['abs_error'] > error_df['interval_half_width']]
        label = fr'nuclei outside $90\%$ interval'

    outlier_Z_vals = np.sort(outliers['Z'].unique()) if not outliers.empty else np.array([])

    if not outliers.empty:
        ax.scatter(
            outliers['N'],
            outliers['Z'],
            marker='*',
            s=60,
            c='yellow',
            edgecolors='k',
            linewidths=1.4,
            label= label,
            zorder=3
        )
        ax.legend(loc='upper left', fontsize=25)

    # Axis limits in physical coordinates
    ax.set_xlim(0, error_df['N'].max() + 5)
    ax.set_ylim(0, error_df['Z'].max() + 5)

    # Base ticks
    ax.set_xticks(np.arange(0, error_df['N'].max() + 6, 20))

    base_yticks = np.arange(0, error_df['Z'].max() + 6, 20)
    all_yticks = np.unique(np.concatenate([base_yticks, outlier_Z_vals]))
    ax.set_yticks(all_yticks)

    ax.set_xlabel('')

    ax.annotate('Neutrons', xy=(0.35, 0.1), xycoords='axes fraction',
             ha='center', va='top', fontsize=45) 
    ax.set_ylabel('')

    ax.annotate('Protons', xy=(0.05,0.7), xycoords='axes fraction',
             ha='center', va='top', fontsize=45,rotation =90) 
    ax.set_title('Nuclear Chart: BMC Absolute Prediction Error', fontsize=20)

    ax.grid(False)
    plt.tight_layout()


