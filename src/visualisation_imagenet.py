import skfuzzy as fuzz
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# df = pd.read_csv("data/output_imagenet_2101.csv")

# for column in ['CPU', 'Memory', 'Energy (plug)', 'Temperature', 'Reads', 'Energy (plug not filtered)']:
#     df.loc[(df['Model'] == "vgg16") & (df['Accuracy_train'].notnull()), column] *= (1110273/97338)

# df.to_csv("data/output_imagenet_2101_fixed.csv", index=False)

df = pd.read_csv("data/output_imagenet_2101_fixed.csv")

df = df[['Energy (plug)','Model','Accuracy_train','Accuracy_val','Accuracy_test','Loss_train','Loss_val','Loss_test', 'Duration']]

df_mean = pd.DataFrame()
df_std = pd.DataFrame()
for model in df['Model'].unique():
    df_model_train = df[(df['Model'] == model) & (df['Accuracy_train'].notnull())]
    df_model_test = df[(df['Model'] == model) & (df['Accuracy_test'].notnull())]
    results_dict_mean = {'Model': model, 
                    'Accuracy train': df_model_train['Accuracy_train'].mean(), 
                    'Accuracy val': df_model_train['Accuracy_val'].mean(), 
                    'Accuracy test': df_model_test['Accuracy_test'].mean(), 
                    'Loss train': df_model_train['Loss_train'].mean(), 
                    'Loss val': df_model_train['Loss_val'].mean(), 
                    'Loss test': df_model_test['Loss_test'].mean(),
                    'Energy train': df_model_train['Energy (plug)'].mean(),
                    'Energy test': df_model_test['Energy (plug)'].mean() if df_model_test['Energy (plug)'].mean() != 0 else df_model_train['Energy (plug)'].mean()*df_model_test['Duration'].mean()/df_model_train['Duration'].mean(),
                    'Pretrained': True if 'pretrained' in model.lower() else False}
    results_dict_std = {'Model': model, 
                    'Accuracy train': df_model_train['Accuracy_train'].std(), 
                    'Accuracy val': df_model_train['Accuracy_val'].std(), 
                    'Accuracy test': df_model_test['Accuracy_test'].std(), 
                    'Loss train': df_model_train['Loss_train'].std(), 
                    'Loss val': df_model_train['Loss_val'].std(), 
                    'Loss test': df_model_test['Loss_test'].std(),
                    'Energy train': df_model_train['Energy (plug)'].std(),
                    'Energy test': df_model_test['Energy (plug)'].std() if df_model_test['Energy (plug)'].std() != 0 else df_model_train['Energy (plug)'].std()*df_model_test['Duration'].std()/df_model_train['Duration'].std(),
                    'Pretrained': True if 'pretrained' in model.lower() else False}
    df_mean = pd.concat([df_mean, pd.DataFrame(results_dict_mean, index=[0])], ignore_index=True)
    df_std = pd.concat([df_std, pd.DataFrame(results_dict_std, index=[0])], ignore_index=True)



print(df_mean)
df = df_mean.copy()

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

# # empty dataframe to store results
# df = pd.DataFrame(columns=['Accuracy', 'Energy_Train', 'Energy_Test', 'Score', 'Defuzzification_Mode'])
# df_aggregated = pd.DataFrame(columns=['Accuracy', 'Energy_Train', 'Energy_Test', 'Score_Membership', 'Score_Value'])

energy_train_universe = np.arange(0, 381025000, 1000)
energy_test_universe = np.arange(0, 25300, 100)
accuracy_universe = np.arange(0, 1.01, 0.01)
score_universe = np.arange(0, 101, 1)

accuracy_low_mf = fuzz.trimf(accuracy_universe, (0, 0, 0.5))
accuracy_mid_mf = fuzz.trimf(accuracy_universe, (0, 0.5, 0.7))
accuracy_hig_mf = fuzz.trapmf(accuracy_universe, (0.5, 0.7, 1, 1))

energy_train_low_mf = fuzz.trimf(energy_train_universe, (0, 0, 1.8144e7)) # 0-24h
energy_train_mid_mf = fuzz.trimf(energy_train_universe, (0, 1.8144e7, 5.4432e7)) # 24h-72h 
energy_train_hig_mf = fuzz.trapmf(energy_train_universe, (1.8144e7, 5.4432e7, 3.81024e8, 3.81024e8)) # >72h

energy_test_low_mf = fuzz.trimf(energy_test_universe, (0, 0, 2100)) # 0s-10s
energy_test_mid_mf = fuzz.trimf(energy_test_universe, (0, 2100, 6300)) # 10s-30s
energy_test_hig_mf = fuzz.trapmf(energy_test_universe, (2100, 6300, 25200, 25200)) # >30s

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
        
results.to_csv('output/imagenet/output_imagenet_scores.csv', index=False)

# Plotting using Matplotlib
results = pd.read_csv('output/imagenet/output_imagenet_scores.csv')

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
plt.savefig('output/imagenet/fuzzy_logic_membership_functions.pdf', dpi=300, format='pdf')
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
        axs[0].bar(results_grouped["Model"], results_grouped["Accuracy"]*100, alpha=0.7, edgecolor="black")
        axs[0].set_ylabel('Accuracy (%)')
        axs[0].set_ylim(0, 100)
        for i, v in enumerate(results_grouped["Accuracy"]*100):
            if v < max(results_grouped["Accuracy"]*100)*0.5:
                axs[0].text(i, v + max(results_grouped["Accuracy"]*100)*0.1, f"{v:.2f}", ha='center', fontsize=8)
            else:
                axs[0].text(i, v - max(results_grouped["Accuracy"]*100)*0.2, f"{v:.2f}", ha='center', fontsize=8)
        # axs[0].set_xlabel('Classifier')
        # make histogram of Energy test for all classifiers
        axs[1].bar(results_grouped["Model"], results_grouped["Energy train"], alpha=0.7, edgecolor="black")
        axs[1].set_ylabel('Energy in\ntraining (J)')
        for i, v in enumerate(results_grouped["Energy train"]):
            if v < max(results_grouped["Energy train"])*0.5:
                axs[1].text(i, v + max(results_grouped["Energy train"])*0.1, f"{v:.1e}", ha='center', fontsize=8)
            else:
                axs[1].text(i, v - max(results_grouped["Energy train"])*0.2, f"{v:.1e}", ha='center', fontsize=8)
        # axs[1].set_xlabel('Classifier')
        # make histogram of Energy test for all classifiers
        axs[2].bar(results_grouped["Model"], results_grouped["Energy test"], alpha=0.7, edgecolor="black")
        axs[2].set_ylabel('Energy in\ntesting (J)')
        for i, v in enumerate(results_grouped["Energy test"]):
            if v < max(results_grouped["Energy test"])*0.5:
                axs[2].text(i, v + max(results_grouped["Energy test"])*0.1, f"{v:.0f}", ha='center', fontsize=8)
            else:
                axs[2].text(i, v - max(results_grouped["Energy test"])*0.2, f"{v:.0f}", ha='center', fontsize=8)
        # axs[1].set_xlabel('Classifier')
        # make histogram of accuracy for all classifiers
        axs[3].bar(results_grouped["Model"], results_grouped["Score"], alpha=0.7, edgecolor="black", color=cm(results_grouped["Score"]/100))
        axs[3].set_ylabel(f'Frugality Score\n(mode {mode.lower()})')
        axs[3].set_xlabel('Classifier')
        axs[3].set_ylim(0, 100)
        for i, v in enumerate(results_grouped["Score"]):
            if v < 100*0.5:
                axs[3].text(i, v + 100*0.1, f"{v:.2f}", ha='center', fontsize=8)
            else:
                axs[3].text(i, v - 100*0.1, f"{v:.2f}", ha='center', fontsize=8)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        fig.align_ylabels(axs)
        plt.savefig(f"output/imagenet/classifier_bar_comparison_pretrained{pretrained}_{mode}.pdf", dpi=300, format='pdf')
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
