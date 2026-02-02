import skfuzzy as fuzz
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

n_points = 21
plotly_use = False

df = pd.read_csv("../../import_lappusmb/summaries/summary_results.csv")

df = df[(df["Run"]==6) | (df["Run"]==7) | (df["Run"]==8)][['Pretrained','Model','Energy train','Energy test','Accuracy train','Accuracy val','Accuracy test','Loss train','Loss val','Loss test']]
df = df[(df["Model"] != 'resnet50') & (df["Model"] != 'densenet121')]
df_mean = df.groupby(['Pretrained','Model']).mean().reset_index()
df_std = df.groupby(['Pretrained','Model']).std().reset_index()

print(df_mean)

plt.figure(figsize=(8,6))
plt.hist(df_mean['Accuracy train'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of train accuracies across Models', fontsize=16)
plt.xlabel('Accuracy Train', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.hist(df_mean['Accuracy val'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of val accuracies across Models', fontsize=16)
plt.xlabel('Accuracy Val', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.hist(df_mean['Accuracy test'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of test accuracies across Models', fontsize=16)
plt.xlabel('Accuracy Test', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.hist(df_mean['Energy train']*3600*1000, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of energy consumptions in training across models', fontsize=16)
plt.xlabel('Energy train (J)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.hist(df_mean['Energy test']*3600*1000, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of energy consumptions in testing across models', fontsize=16)
plt.xlabel('Energy test (J)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df_mean['Energy train']*3600*1000, df_mean['Accuracy test'], c=df_mean['Pretrained'])
plt.title('Distribution of energy consumptions in testing across models', fontsize=16)
plt.xlabel('Energy train (J)', fontsize=14)
plt.ylabel('Accuracy Test', fontsize=14)
plt.tight_layout()
plt.show()

# # empty dataframe to store results
# df = pd.DataFrame(columns=['Accuracy', 'Energy_Train', 'Energy_Test', 'Score', 'Defuzzification_Mode'])
# df_aggregated = pd.DataFrame(columns=['Accuracy', 'Energy_Train', 'Energy_Test', 'Score_Membership', 'Score_Value'])

df['Energy train'] *= 3600*1000  # convert to Joules
df['Energy test'] *= 3600*1000   # convert to Joules

energy_train_universe = np.arange(0, 1837000, 1000)
energy_test_universe = np.arange(0, 10300, 100)
accuracy_universe = np.arange(0, 1.01, 0.01)
score_universe = np.arange(0, 101, 1)

accuracy_low_mf = fuzz.trimf(accuracy_universe, (0, 0, 0.5))
accuracy_mid_mf = fuzz.trimf(accuracy_universe, (0, 0.5, 1))
accuracy_hig_mf = fuzz.trimf(accuracy_universe, (0.5, 1, 1))

energy_train_low_mf = fuzz.trimf(energy_train_universe, (0, 0, 51000)) # 0s-5min
energy_train_mid_mf = fuzz.trimf(energy_train_universe, (0, 51000, 612000)) # 5min-1h 
energy_train_hig_mf = fuzz.trapmf(energy_train_universe, (51000, 612000, 1836000, 1836000)) # >1h

energy_test_low_mf = fuzz.trimf(energy_test_universe, (0, 0, 5100)) # 0s-30s
energy_test_mid_mf = fuzz.trimf(energy_test_universe, (0, 5100, 10200)) # 30s-60s
energy_test_hig_mf = fuzz.trapmf(energy_test_universe, (5100, 10200, 20400, 20400)) # >60s

score_1_mf = fuzz.trimf(score_universe, (0, 0, 25))
score_2_mf = fuzz.trimf(score_universe, (0, 25, 50))
score_3_mf = fuzz.trimf(score_universe, (25, 50, 75))
score_4_mf = fuzz.trimf(score_universe, (50, 75, 100))
score_5_mf = fuzz.trimf(score_universe, (75, 100, 100))

results = pd.DataFrame(columns=['Energy train', 'Energy test', 'Accuracy', 'Model', 'Pretrained', 'Score', 'Defuzzification_Mode'])
df = df.reset_index(drop=True)

for i in range(len(df)):
    accuracy_i = df.loc[i]['Accuracy test']
    energy_train_i = df.loc[i]['Energy train']
    energy_test_i = df.loc[i]['Energy test']
    # Fuzzification
    accuracy_low = fuzz.interp_membership(accuracy_universe, accuracy_low_mf, accuracy_i)
    accuracy_mid = fuzz.interp_membership(accuracy_universe, accuracy_mid_mf, accuracy_i)
    accuracy_hig = fuzz.interp_membership(accuracy_universe, accuracy_hig_mf, accuracy_i)

    energy_train_low = fuzz.interp_membership(energy_train_universe, energy_train_low_mf, energy_train_i)
    energy_train_mid = fuzz.interp_membership(energy_train_universe, energy_train_mid_mf, energy_train_i)
    energy_train_hig = fuzz.interp_membership(energy_train_universe, energy_train_hig_mf, energy_train_i)

    energy_test_low = fuzz.interp_membership(energy_test_universe, energy_test_low_mf, energy_test_i)
    energy_test_mid = fuzz.interp_membership(energy_test_universe, energy_test_mid_mf, energy_test_i)
    energy_test_hig = fuzz.interp_membership(energy_test_universe, energy_test_hig_mf, energy_test_i)

    score_5 = np.minimum(score_5_mf,min(accuracy_hig, energy_train_low, energy_test_low))
    
    score_4 = np.minimum(score_4_mf,max(
                min(accuracy_hig, energy_train_low, energy_test_mid),
                max(
                min(accuracy_hig, energy_train_mid, energy_test_low),
                min(accuracy_mid, energy_train_low, energy_test_low)
                )))
        
    score_3 = np.minimum(score_3_mf,max(
                min(accuracy_hig, energy_train_low, energy_test_hig),
                max(
                min(accuracy_hig, energy_train_mid, energy_test_mid),
                max(
                min(accuracy_hig, energy_train_hig, energy_test_mid),
                max(
                min(accuracy_mid, energy_train_mid, energy_test_low),
                max(
                min(accuracy_mid, energy_train_low, energy_test_mid),
                min(accuracy_low, energy_train_low, energy_test_low)
                ))))))
        
    score_2 = np.minimum(score_2_mf,max(
                min(accuracy_hig, energy_train_mid, energy_test_hig),
                max(
                min(accuracy_hig, energy_train_hig, energy_test_mid),
                max(
                min(accuracy_mid, energy_train_low, energy_test_hig),
                max(
                min(accuracy_mid, energy_train_mid, energy_test_mid),
                max(
                min(accuracy_mid, energy_train_hig, energy_test_low),
                max(
                min(accuracy_low, energy_train_low, energy_test_mid),
                min(accuracy_low, energy_train_mid, energy_test_low)
                )))))))
        
    score_1 = np.minimum(score_1_mf,max(
                min(accuracy_hig, energy_train_hig, energy_test_hig),
                max(
                min(accuracy_mid, energy_train_hig, energy_test_hig),
                max(
                min(accuracy_mid, energy_train_hig, energy_test_mid),
                max(
                min(accuracy_mid, energy_train_mid, energy_test_hig),
                max(
                min(accuracy_low, energy_train_hig, energy_test_hig),
                max(
                min(accuracy_low, energy_train_hig, energy_test_mid),
                max(
                min(accuracy_low, energy_train_mid, energy_test_hig),
                max(
                min(accuracy_low, energy_train_low, energy_test_hig),
                max(
                min(accuracy_low, energy_train_mid, energy_test_mid),
                min(accuracy_low, energy_train_hig, energy_test_low)
                ))))))))))

    aggregated = np.maximum(score_1, np.maximum(score_2, np.maximum(score_3, np.maximum(score_4, score_5))))
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
            'Energy train': [energy_train_i],
            'Energy test': [energy_test_i],
            'Accuracy': [accuracy_i],
            'Model': [df.loc[i]['Model']],
            'Pretrained': [df.loc[i]['Pretrained']],
            'Score': [defuzzified],
            'Defuzzification_Mode': [mode]
        })], ignore_index=True)
        
results.to_csv('output/cifar100/output_cifar_scores.csv', index=False)

# Plotting using Matplotlib
results = pd.read_csv('output/cifar100/output_cifar_scores.csv')

results["Model"] = results["Model"].replace({
    "resnet18": "ResNet18",
    "vgg16": "VGG16",
    "mobilenet_v2": "MobileNet V2",
    "efficientnet_b0": "EfficientNet B0",
    "shufflenet_v2_x0_5": "ShuffleNet V2",
    "squeezenet1_0": "SqueezeNet V1"
})

fig, axs = plt.subplots(4, 1, figsize=(4, 6), sharey=True)
axs[0].plot(accuracy_universe*100, accuracy_low_mf, label='Low')
axs[0].plot(accuracy_universe*100, accuracy_mid_mf, label='Medium')
axs[0].plot(accuracy_universe*100, accuracy_hig_mf, label='High')
# axs[0].set_title('Accuracy Membership Functions')
axs[0].set_xlabel('Accuracy (%)')
axs[0].set_ylabel('Membership Degree')
# axs[0].legend(loc='right')
axs[1].plot(energy_train_universe, energy_train_low_mf, label='Low')
axs[1].plot(energy_train_universe, energy_train_mid_mf, label='Medium')
axs[1].plot(energy_train_universe, energy_train_hig_mf, label='High')
# axs[1].set_title('Energy Train Membership Functions')
axs[1].set_xlabel('Energy during training (J)')
axs[1].set_ylabel('Membership Degree')
# axs[1].legend(loc='right')
axs[2].plot(energy_test_universe, energy_test_low_mf, label='Low')
axs[2].plot(energy_test_universe, energy_test_mid_mf, label='Medium')
axs[2].plot(energy_test_universe, energy_test_hig_mf, label='High')
# axs[1].set_title('Energy Train Membership Functions')
axs[2].set_xlabel('Energy during testing (J)')
axs[2].set_ylabel('Membership Degree')
# axs[1].legend(loc='right')
axs[3].plot(score_universe, score_1_mf, label='1')
axs[3].plot(score_universe, score_2_mf, label='2')
axs[3].plot(score_universe, score_3_mf, label='3')
axs[3].plot(score_universe, score_4_mf, label='4')
axs[3].plot(score_universe, score_5_mf, label='5')
# axs[2].set_title('Energy Test Membership Functions')
axs[3].set_xlabel('Frugality Score')
axs[3].set_ylabel('Membership Degree')
axs[3].legend(loc='right')
plt.tight_layout()
plt.savefig('output/cifar100/fuzzy_logic_membership_functions.pdf', dpi=300, format='pdf')
plt.show()

results_all = results.copy()
results_all.sort_values(by=['Pretrained', 'Model'], inplace=True)
for pretrained in results_all['Pretrained'].unique():
    for mode in results_all['Defuzzification_Mode'].unique():
        results = results_all[(results_all['Pretrained'] == pretrained) & (results_all['Defuzzification_Mode'] == mode)]
        # same but only one y label on the left
        fig, axs = plt.subplots(4, 1, figsize=(5, 6))
        for ax in axs:
            ax.label_outer()
        # make histogram of Energy train for all classifiers
        results_grouped = results.groupby(['Model'])[["Accuracy", "Energy train", "Energy test", "Score"]].mean().reset_index()
        results_grouped_95confidence = results.groupby(['Model'])[["Accuracy", "Energy train", "Energy test", "Score"]].agg(lambda x: np.percentile(x, 95)).reset_index()
        
        cm = plt.get_cmap('viridis')
        axs[0].bar(results_grouped["Model"], results_grouped["Accuracy"]*100, yerr=abs(results_grouped_95confidence["Accuracy"] - results_grouped["Accuracy"]), capsize=5, alpha=0.7, edgecolor="black")
        axs[0].set_ylabel('Accuracy (%)')
        axs[0].set_ylim(0, 100)
        for i, v in enumerate(results_grouped["Accuracy"]*100):
            if v < 100*0.5:
                axs[0].text(i, v + max(results_grouped["Accuracy"]*100)*0.1, f"{v:.2f}", ha='center', fontsize=8)
            else:
                axs[0].text(i, v - max(results_grouped["Accuracy"]*100)*0.2, f"{v:.2f}", ha='center', fontsize=8)
        # axs[0].set_xlabel('Classifier')
        # make histogram of Energy test for all classifiers
        axs[1].bar(results_grouped["Model"], results_grouped["Energy train"], yerr=abs(results_grouped_95confidence["Energy train"] - results_grouped["Energy train"]), capsize=5, alpha=0.7, edgecolor="black")
        axs[1].set_ylabel('Energy in\ntraining (J)')
        for i, v in enumerate(results_grouped["Energy train"]):
            if v < max(results_grouped["Energy train"])*0.5:
                axs[1].text(i, v + max(results_grouped["Energy train"])*0.1, f"{v:.1e}", ha='center', fontsize=8)
            else:
                axs[1].text(i, v - max(results_grouped["Energy train"])*0.3, f"{v:.1e}", ha='center', fontsize=8)
        # axs[1].set_xlabel('Classifier')
        # make histogram of Energy test for all classifiers
        axs[2].bar(results_grouped["Model"], results_grouped["Energy test"], yerr=abs(results_grouped_95confidence["Energy test"] - results_grouped["Energy test"]), capsize=5, alpha=0.7, edgecolor="black")
        axs[2].set_ylabel('Energy in\ntesting (J)')
        for i, v in enumerate(results_grouped["Energy test"]):
            if v < max(results_grouped["Energy test"])*0.5:
                axs[2].text(i, v + max(results_grouped["Energy test"])*0.1, f"{v:.0f}", ha='center', fontsize=8)
            else:
                axs[2].text(i, v - max(results_grouped["Energy test"])*0.3, f"{v:.0f}", ha='center', fontsize=8)
        # axs[1].set_xlabel('Classifier')
        # make histogram of accuracy for all classifiers
        axs[3].bar(results_grouped["Model"], results_grouped["Score"], yerr=abs(results_grouped_95confidence["Score"] - results_grouped["Score"]), capsize=5, alpha=0.7, edgecolor="black", color=cm(results_grouped["Score"]/100))
        axs[3].set_ylabel(f'Frugality Score\n(mode {mode.lower()})')
        axs[3].set_xlabel('Classifier')
        axs[3].set_ylim(0, 100)
        for i, v in enumerate(results_grouped["Score"]):
            if v < 100*0.5:
                axs[3].text(i, v + 100*0.1, f"{v:.2f}", ha='center', fontsize=8)
            else:
                axs[3].text(i, v - 100*0.2, f"{v:.2f}", ha='center', fontsize=8)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        fig.align_ylabels(axs)
        plt.savefig(f"output/cifar100/classifier_bar_comparison_pretrained{pretrained}_{mode}.pdf", dpi=300, format='pdf')
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
