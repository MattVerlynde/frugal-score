import skfuzzy as fuzz
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

n_points = 21
plotly_use = False

df = pd.read_csv("data/output_change_detection.csv")

df = df[['Energy (plug)', 'Energy (CodeCarbon)', 'AUC', 'Duration', 'Window size','Threads','Number images','Method']]
df_mean = df.groupby(['Window size','Threads','Number images','Method']).mean().reset_index()
df_std = df.groupby(['Window size','Threads','Number images','Method']).std().reset_index()

plt.figure(figsize=(8,6))
plt.hist(df_mean['AUC'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of AUC across Change Detection Methods', fontsize=16)
plt.xlabel('AUC', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.hist(df_mean['Duration'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Duration across Change Detection Methods', fontsize=16)
plt.xlabel('Duration (s)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.hist(df_mean['Energy (plug)']/(3600*1000), bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Energy (plug) across Change Detection Methods', fontsize=16)
plt.xlabel('Energy (plug) (kWh)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.hist(df_mean['Energy (CodeCarbon)']/(3600*1000), bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Energy (CodeCarbon) across Change Detection Methods', fontsize=16)
plt.xlabel('Energy (CodeCarbon) (kWh)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

# # empty dataframe to store results
# df = pd.DataFrame(columns=['Accuracy', 'Energy_Train', 'Energy_Test', 'Score', 'Defuzzification_Mode'])
# df_aggregated = pd.DataFrame(columns=['Accuracy', 'Energy_Train', 'Energy_Test', 'Score_Membership', 'Score_Value'])

energy_universe = np.arange(0, 3025000, 1000)
auc_universe = np.arange(0, 1.01, 0.01)
score_universe = np.arange(0, 101, 1)

auc_low_mf = fuzz.trimf(auc_universe, (0.25, 0.5, 0.75))
auc_mid_mf = fuzz.trimf(auc_universe, (0, 0.25, 0.5)) + fuzz.trimf(auc_universe, (0.5, 0.75, 1))
auc_hig_mf = fuzz.trimf(auc_universe, (0, 0, 0.25)) + fuzz.trimf(auc_universe, (0.75, 1, 1))

energy_low_mf = fuzz.trimf(energy_universe, (0, 0, 12600))
energy_mid_mf = fuzz.trimf(energy_universe, (0, 12600, 63000))
energy_hig_mf = fuzz.trapmf(energy_universe, (12600, 63000, 3024000, 3024000))

score_low_mf = fuzz.trimf(score_universe, (0, 0, 50))
score_mid_mf = fuzz.trimf(score_universe, (0, 50, 100))
score_hig_mf = fuzz.trimf(score_universe, (50, 100, 100))

results = pd.DataFrame(columns=['Energy (plug)', 'Energy (CodeCarbon)', 'AUC', 'Duration', 'Window size','Threads','Number images','Method', 'Accuracy', 'Energy_Train', 'Energy_Test', 'Score', 'Defuzzification_Mode'])
df = df.reset_index(drop=True)

type_energy = 'codecarbon'  # 'plug' or 'codecarbon'

for i in range(len(df)):
    auc_i = df.loc[i]['AUC']
    energy_plug_i = df.loc[i]['Energy (plug)']
    energy_codecarbon_i = df.loc[i]['Energy (CodeCarbon)']
    if type_energy == 'plug':
        energy_i = energy_plug_i
    else:
        energy_i = energy_codecarbon_i
    # Fuzzification
    auc_low = fuzz.interp_membership(auc_universe, auc_low_mf, auc_i)
    auc_mid = fuzz.interp_membership(auc_universe, auc_mid_mf, auc_i)
    auc_hig = fuzz.interp_membership(auc_universe, auc_hig_mf, auc_i)

    energy_plug_low = fuzz.interp_membership(energy_universe, energy_low_mf, energy_i)
    energy_plug_mid = fuzz.interp_membership(energy_universe, energy_mid_mf, energy_i)
    energy_plug_hig = fuzz.interp_membership(energy_universe, energy_hig_mf, energy_i)
    
    energy_codecarbon_low = fuzz.interp_membership(energy_universe, energy_low_mf, energy_codecarbon_i)
    energy_codecarbon_mid = fuzz.interp_membership(energy_universe, energy_mid_mf, energy_codecarbon_i)
    energy_codecarbon_hig = fuzz.interp_membership(energy_universe, energy_hig_mf, energy_codecarbon_i)

    score_hig = np.minimum(score_hig_mf,min(auc_hig, energy_plug_low))
    
    score_mid = np.minimum(score_mid_mf,max(
                min(auc_mid, energy_plug_mid),
                max(
                min(auc_hig, energy_plug_mid),
                min(auc_mid, energy_plug_low)
                )))
        
    score_low = np.minimum(score_low_mf,max(energy_plug_hig, auc_low))
    
    aggregated = np.maximum(score_hig, np.maximum(score_mid, score_low))
    # print(aggregated)
    aggregated = np.fmax(aggregated, 0)
    # print(aggregated)
    for mode in ['centroid', 'bisector', 'mom', 'som', 'lom']:
        if np.unique(aggregated).size == 1:
            if aggregated[0] == 0:
                defuzzified = None
            defuzzified = None
        else:
            defuzzified = fuzz.defuzz(score_universe, aggregated, mode)
        results = pd.concat([results, pd.DataFrame({
            'Energy (plug)': [energy_plug_i],
            'Energy (CodeCarbon)': [energy_codecarbon_i],
            'AUC': [auc_i],
            'Duration': [df.loc[i]['Duration']],
            'Window size': [df.loc[i]['Window size']],
            'Threads': [df.loc[i]['Threads']],
            'Number images': [df.loc[i]['Number images']],
            'Method': [df.loc[i]['Method']],
            'Accuracy': [auc_i],
            'Energy_Train': [energy_plug_i],
            'Energy_Test': [energy_codecarbon_i],
            'Score': [defuzzified],
            'Defuzzification_Mode': [mode]
        })], ignore_index=True)
        
results.to_csv('data/output_change_detection_scores.csv', index=False)

# Plotting using Matplotlib
results = pd.read_csv('data/output_change_detection_scores.csv')
# keep rows where thread is 12 or method is 2
results = results[(results['Threads'] == 12) | (results['Method'] == 2)]


fig, axs = plt.subplots(3, 1, figsize=(4, 6), sharey=True)
axs[0].plot(auc_universe, auc_low_mf, label='Low')
axs[0].plot(auc_universe, auc_mid_mf, label='Medium')
axs[0].plot(auc_universe, auc_hig_mf, label='High')
# axs[0].set_title('Accuracy Membership Functions')
axs[0].set_xlabel('AUC')
axs[0].set_ylabel('Membership Degree')
axs[0].set_ylim(0, 1)
# axs[0].legend(loc='right')
axs[1].plot(energy_universe, energy_low_mf, label='Low')
axs[1].plot(energy_universe, energy_mid_mf, label='Medium')
axs[1].plot(energy_universe, energy_hig_mf, label='High')
# axs[1].set_title('Energy Train Membership Functions')
axs[1].set_xlabel('Energy (J)')
axs[1].set_ylabel('Membership Degree')
# axs[1].legend(loc='right')
axs[2].plot(score_universe, score_low_mf, label='Low')
axs[2].plot(score_universe, score_mid_mf, label='Medium')
axs[2].plot(score_universe, score_hig_mf, label='High')
# axs[2].set_title('Energy Test Membership Functions')
axs[2].set_xlabel('Frugality Score')
axs[2].set_ylabel('Membership Degree')
axs[2].legend(loc='right')
plt.tight_layout()
plt.savefig('output/changedetection/fuzzy_logic_membership_functions.pdf', dpi=300, format='pdf')
plt.show()

results_all = results.copy()
results_all.sort_values(by=['Method', 'Window size', 'Threads', 'Number images'], inplace=True)
for n_images in [2, 4, 17]:
    for mode in results_all['Defuzzification_Mode'].unique():
        results = results_all[(results_all['Number images'] == n_images) & (results_all['Defuzzification_Mode'] == mode)]
        # same but only one y label on the left
        fig, axs = plt.subplots(3, 1, figsize=(5, 5))
        for ax in axs:
            ax.label_outer()
        # make histogram of Energy train for all classifiers
        results_grouped = results.groupby(['Window size','Threads','Number images','Method'])[["AUC", "Energy (plug)", "Energy (CodeCarbon)", "Score"]].mean().reset_index()
        results_grouped_95confidence = results.groupby(['Window size','Threads','Number images','Method'])[["AUC", "Energy (plug)", "Energy (CodeCarbon)", "Score"]].agg(lambda x: np.percentile(x, 95)).reset_index()
        results_grouped["Classifier"] = results_grouped["Method"].astype(str) + results_grouped["Window size"].astype(str) + results_grouped["Threads"].astype(str) + results_grouped["Number images"].astype(str)
        results_grouped_95confidence["Classifier"] = results_grouped_95confidence["Method"].astype(str) + results_grouped_95confidence["Window size"].astype(str) + results_grouped_95confidence["Threads"].astype(str) + results_grouped_95confidence["Number images"].astype(str)
        results_grouped.sort_values(by=['Method', 'Window size'], inplace=True)
        results_grouped_95confidence.sort_values(by=['Method', 'Window size'], inplace=True)
        def replace_classif_name(name):
            if name.startswith('2'):
                return 'LogDiff'
            elif name.startswith('1'):
                return 'NG-GLRT' + name[3:].split('.')[0]
            else:
                return 'G-GLRT' + name[3:].split('.')[0]
        results_grouped["Classifier"] = results_grouped["Classifier"].apply(replace_classif_name)
        results_grouped_95confidence["Classifier"] = results_grouped_95confidence["Classifier"].apply(replace_classif_name) 

        
        cm = plt.get_cmap('viridis')
        axs[0].bar(results_grouped["Classifier"], results_grouped["AUC"], yerr=abs(results_grouped_95confidence["AUC"] - results_grouped["AUC"]), capsize=5, alpha=0.7, edgecolor="black")
        axs[0].set_ylabel('AUC')
        axs[0].set_ylim(0, 1)
        for i, v in enumerate(results_grouped["AUC"]):
            if v < max(results_grouped["AUC"])*0.25:
                axs[0].text(i, v + max(results_grouped["AUC"])*0.1, f"{v:.0f}", ha='center', fontsize=8)
            else:
                axs[0].text(i, v - max(results_grouped["AUC"])*0.1, f"{v:.2f}", ha='center', fontsize=8)

        # axs[0].set_xlabel('Classifier')
        # make histogram of Energy test for all classifiers
        axs[1].bar(results_grouped["Classifier"], results_grouped["Energy (CodeCarbon)"], yerr=abs(results_grouped_95confidence["Energy (CodeCarbon)"] - results_grouped["Energy (CodeCarbon)"]), capsize=5, alpha=0.7, edgecolor="black")
        axs[1].set_ylabel('Energy (J)')
        for i, v in enumerate(results_grouped["Energy (CodeCarbon)"]):
            if v < max(results_grouped["Energy (CodeCarbon)"])*0.25:
                axs[1].text(i, v + max(results_grouped["Energy (CodeCarbon)"])*0.1, f"{v:.0f}", ha='center', fontsize=8)
            else:
                axs[1].text(i, v - max(results_grouped["Energy (CodeCarbon)"])*0.1, f"{v:.0f}", ha='center', fontsize=8)
        # axs[1].set_xlabel('Classifier')
        # make histogram of accuracy for all classifiers
        axs[2].bar(results_grouped["Classifier"], results_grouped["Score"], yerr=abs(results_grouped_95confidence["Score"] - results_grouped["Score"]), capsize=5, alpha=0.7, edgecolor="black", color=cm(results_grouped["Score"]/100))
        axs[2].set_ylabel(f'Frugality Score\n(mode {mode.upper()})')
        axs[2].set_xlabel('Change Detection Method')
        axs[2].set_ylim(0, 100)
        for i, v in enumerate(results_grouped["Score"]):
            if v < max(results_grouped["Score"])*0.25:
                axs[2].text(i, v + 100*0.1, f"{v:.2f}", ha='center', fontsize=8)
            else:
                axs[2].text(i, v - 100*0.1, f"{v:.2f}", ha='center', fontsize=8)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        fig.align_ylabels(axs)
        plt.savefig(f"output/changedetection/classifier_bar_comparison_{n_images}images_{mode}.pdf", dpi=300, format='pdf')
        plt.show()
   
# PLot 3D heatmap of results
# df["Energy (plug)"] /= (3600*1000)
# for mode in ['centroid', 'bisector', 'mom', 'som', 'lom']:
#     fig, axs = plt.subplots(1, 1, figsize=(6, 5), subplot_kw={"projection": "3d"})
#     df = pd.read_csv('data/output_change_detection_scores.csv')
#     sc = axs.scatter(
#             data = df[df['Defuzzification_Mode'] == mode],
#             xs='Energy (plug)', 
#             ys='AUC', 
#             c='Score', 
#             cmap='viridis', 
#             s=10,
#             alpha=1,
#             marker='o')
#     sc.set_clim(0, 100)
#     plt.colorbar(sc, label=f'Frugality Score (Mode {mode.upper()})', pad=0.1, shrink=0.8)
#     axs.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
#     axs.set_xlabel('Energy consumption (J)')
#     axs.set_ylabel('AUC')
#     plt.tight_layout()
#     plt.savefig(f'output/changedetection/fuzzy_logic_score_distribution_{mode}.pdf', dpi=300, format='pdf')
#     plt.show()
