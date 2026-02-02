import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from keras.datasets import mnist
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from codecarbon import OfflineEmissionsTracker

import pandas as pd

mode = "centroid"  # defuzzification mode

def main(repeat=30):

    names = [
        "Decision Tree",
        "Random Forest",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
        "Nearest Neighbors",
        "Linear SVM",
    ]

    classifiers = [      
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        AdaBoostClassifier(random_state=42),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=42),
    ]
        # "Gaussian Process",
        # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        # "RBF SVM",
        # SVC(gamma=2, C=1, random_state=42),
        # "Neural Net",
        # MLPClassifier(alpha=1, max_iter=1000, random_state=42),


    datasets = [
        mnist.load_data(),
    ]

    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    # datasets = [
    #     make_moons(noise=0.3, random_state=0),
    #     make_circles(noise=0.2, factor=0.5, random_state=1),
    #     linearly_separable,
    # ]

    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    results = pd.DataFrame(columns=["Classifier", "Dataset", "Score", "Energy train", "Energy test", "Time"])
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        # X, y = ds
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.4, random_state=42
        # )
        
        # x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        # y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        (X_train, y_train), (X_test, y_test) = ds
        print(X_train.shape)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        # # Flatten the images
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # just plot the dataset first
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k")
        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.6, edgecolors="k"
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
        i_dataset = i

        # iterate over classifiers
        for name, classifier in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

            for _ in range(repeat):
                clf = make_pipeline(StandardScaler(), classifier)
                tracker = OfflineEmissionsTracker(project_name="Classification train", country_iso_code="FRA", save_to_file=True, output_dir="output")
                tracker.start()
                t0 = time.time()
                clf.fit(X_train, y_train)
                t1 = time.time()
                emissions = tracker.stop()
                energy_train = emissions*3600000/0.034 # J
                tracker = OfflineEmissionsTracker(project_name="Classification test", country_iso_code="FRA", save_to_file=True, output_dir="output")
                tracker.start()
                score = clf.score(X_test, y_test)
                emissions = tracker.stop()
                energy_test = emissions*3600000/0.034 # J

                results = pd.concat([results, pd.DataFrame({"Classifier": [name], "Dataset": [ds_cnt], "Score": [score], "Energy train": [energy_train], "Energy test": [energy_test], "Time": [t1 - t0]})], ignore_index=True)

            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k"
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                edgecolors="k",
                alpha=0.6,
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(
                x_max - 0.3,
                y_min + 0.3,
                ("P = %.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            ax.text(
                x_max - 0.3,
                y_max - 0.3,
                ("E = %.2f J" % energy_train).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )

            i += 1
            print(f"Classifier {i-i_dataset}: ", name)
            print("Score: ", score)
            print("Time: %.2fs" % (t1 - t0))
            print("Energy train: ", energy_train)

    plt.tight_layout()
    plt.show()

    results["Iteration"] = [c for _, c in zip(range(len(results)), cycle(range(repeat)))]


    emissions_total = pd.read_csv("output/emissions.csv")
    emissions_total["step"] = [c for _, c in zip(range(len(emissions_total)), cycle(['train','test']))]

    emissions_info = pd.DataFrame(columns=["Classifier", "Dataset"])
    for i in range(len(results)):
        for j in range(2):
            emissions_info = pd.concat([emissions_info, pd.DataFrame({"Classifier": [results.iloc[i]["Classifier"]], "Dataset": [results.iloc[i]["Dataset"]]})], ignore_index=True)
    emissions_total["Classifier"] = emissions_info["Classifier"]
    emissions_total["Dataset"] = emissions_info["Dataset"]

    return results

# Save the results to a CSV file
# main().to_csv("output/results_classif_mnist.csv", index=False)

results = pd.read_csv("output/results_classif_mnist.csv")
# Display the results
print(results.head())

# emissions_total = pd.read_csv("output/emissions.csv")
# emissions_total["step"] = [c for _, c in zip(range(len(emissions_total)), cycle(['train','test']))]

# emissions_info = pd.DataFrame(columns=["Classifier", "Dataset", "Iteration"])
# for i in range(len(results)):
#     for j in range(2):
#         emissions_info = pd.concat([emissions_info, pd.DataFrame({"Classifier": [results.iloc[i]["Classifier"]], "Dataset": [results.iloc[i]["Dataset"]]})], ignore_index=True)
# emissions_total["Classifier"] = emissions_info["Classifier"]
# emissions_total["Dataset"] = emissions_info["Dataset"]
# emissions_total.to_csv("output/emissions_total.csv", index=False)

# for i in range(len(results)):
#     results.iloc[i]["Energy train"] = emissions_total.iloc[i*2]["energy_consumed"]*3600000
#     results.iloc[i]["Energy test"] = emissions_total.iloc[i*2+1]["energy_consumed"]*3600000

# Calculate ACP between the classifiers
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_scaled = StandardScaler().fit_transform(results[["Score", "Energy train", "Energy test", "Time"]])
# Perform PCA
pca = PCA(n_components=2)
acp_results = pca.fit_transform(data_scaled)

# Concat classifer names and dataset numbers
results["Setup"] = results["Classifier"] + " - " + results["Dataset"].astype(str)

acp_results = pd.DataFrame(acp_results, columns=["PC1", "PC2"])
acp_results["Setup"] = results["Setup"].values
acp_results["Classifier"] = results["Classifier"].values
acp_results["Dataset"] = results["Dataset"].values
acp_results["Score"] = results["Score"].values
acp_results["Energy train"] = results["Energy train"].values
acp_results["Energy test"] = results["Energy test"].values
acp_results["Time"] = results["Time"].values

# Plot the results
# plt.figure(figsize=(10, 10))
# for clf, marker in zip(results["Classifier"].unique(), cycle(("o", "s", "D", "^", "v", "<", ">", "p", "*"))):
#     plt.scatter(
#         acp_results[acp_results["Classifier"] == clf]["PC1"],
#         acp_results[acp_results["Classifier"] == clf]["PC2"],
#         c=acp_results[acp_results["Classifier"] == clf]["Dataset"],
#         marker=marker,
#         label=clf,
#         alpha=0.5,
#     )
# plt.title("PCA of Classifiers")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend()
# plt.show()

# Plot the results
# plt.figure(figsize=(10, 10))
# for clf, marker in zip(results["Classifier"].unique(), cycle(("o", "s", "D", "^", "v", "<", ">", "p", "*"))):
#     plt.scatter(
#         acp_results[acp_results["Classifier"] == clf]["Score"],
#         acp_results[acp_results["Classifier"] == clf]["Energy train"],
#         c=acp_results[acp_results["Classifier"] == clf]["Dataset"],
#         marker=marker,
#         label=clf,
#         alpha=0.5,
#     )
# plt.title("Score vs Energy")
# plt.xlabel("Score")
# plt.ylabel("Energy train")
# plt.legend()
# plt.show()

# Calculate the correlation matrix
correlation_matrix = results[["Score", "Energy train", "Time"]].corr()
# Display the correlation matrix
print(correlation_matrix)
# Plot the correlation matrix
import seaborn as sns
# plt.figure(figsize=(10, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
# plt.title("Correlation Matrix")
# plt.show()

# Plot the results
# results.groupby(["Classifier", "Dataset"]).agg({"Energy train": ["mean", "std"]}).reset_index().plot(
#     x="Classifier",
#     y="Energy train",
#     yerr=results.groupby(["Classifier", "Dataset"]).agg({"Energy train": "std"}).reset_index()["Energy train"],
#     kind="bar",
#     figsize=(15, 7),
#     title="Average Score per Classifier and Dataset",
#     xlabel="Classifier",
#     ylabel="Energy train",
#     legend=True,
#     alpha=0.7,
#     edgecolor="black",
#     linewidth=1,
#     grid=True)
# plt.show()
# results.groupby(["Classifier", "Dataset"]).agg({"Score": ["mean", "std"]}).reset_index().plot(
#     x="Classifier",
#     y="Score",
#     yerr=results.groupby(["Classifier", "Dataset"]).agg({"Score": "std"}).reset_index()["Score"],
#     kind="bar",
#     figsize=(15, 7),
#     title="Average Score per Classifier and Dataset",
#     xlabel="Classifier",
#     ylabel="Score",
#     legend=True,
#     alpha=0.7,
#     edgecolor="black",
#     linewidth=1,
#     grid=True)
# plt.show()

from sklearn.preprocessing import OneHotEncoder

# One-hot encode the dataset and classifier columns
encoder = OneHotEncoder()
results_encoded = encoder.fit_transform(results[["Dataset", "Classifier"]]).toarray()
# Get the feature names
feature_names = encoder.get_feature_names_out(["Dataset", "Classifier"])
# Create a DataFrame with the encoded features
results_encoded_df = pd.DataFrame(results_encoded, columns=feature_names)
# Concatenate the encoded features with the original DataFrame
results_final = pd.concat([results[["Score", "Energy train", "Energy test", "Time", "Iteration"]].reset_index(drop=True), results_encoded_df], axis=1)
# Display the final DataFrame
print(results_final)
# Calculate the correlation matrix
correlation_matrix_final = results_final.corr()
# Display the correlation matrix
print(correlation_matrix_final)
# Plot the correlation matrix
# plt.figure(figsize=(10, 10))
# sns.heatmap(correlation_matrix_final, annot=True, cmap="coolwarm", square=True)
# plt.title("Correlation Matrix")
# plt.show()

import skfuzzy as fuzz

scores = []
for i in range(len(results)):
    x_energy_train = results.iloc[i]["Energy train"]
    x_energy_test = results.iloc[i]["Energy test"]
    x_perf = results.iloc[i]["Score"]

    x_score_universe = np.arange(0, 101, 1)
    # Generate fuzzy membership functions
    score_low = fuzz.trimf(x_score_universe, (0, 0, 50))
    score_hig = fuzz.trimf(x_score_universe, (50, 100, 100))
    score_mid = 1- (score_low + score_hig)

    # score_1 = fuzz.trimf(x_score, (0, 0, 0.25))
    # score_2 = fuzz.trimf(x_score, (0, 0.25, 0.5))
    # score_3 = fuzz.trimf(x_score, (0.25, 0.5, 0.75))
    # score_4 = fuzz.trimf(x_score, (0.5, 0.75, 1))
    # score_5 = fuzz.trimf(x_score, (0.75, 1, 1))
    x_energy_train_universe = np.arange(0, 37900, 100)
    x_energy_test_universe = np.arange(0, 12700, 100)
    x_perf_universe = np.arange(0, 1.1, 0.1)

    perf_low_mf = fuzz.trapmf(x_perf_universe, (0, 0, 0.5, 0.8))
    perf_low = fuzz.interp_membership(x_perf_universe, perf_low_mf, x_perf)
    perf_mid_mf = fuzz.trimf(x_perf_universe, (0.5, 0.8, 1))
    perf_mid = fuzz.interp_membership(x_perf_universe, perf_mid_mf, x_perf)
    perf_hig_mf = fuzz.trimf(x_perf_universe, (0.8, 1, 1))
    perf_hig = fuzz.interp_membership(x_perf_universe, perf_hig_mf, x_perf)
    
    energy_train_low_mf = fuzz.trimf(x_energy_train_universe, (0, 0, 210))
    energy_train_low = fuzz.interp_membership(x_energy_train_universe, energy_train_low_mf, x_energy_train)
    energy_train_mid_mf = fuzz.trimf(x_energy_train_universe, (0, 210, 1050))
    energy_train_mid = fuzz.interp_membership(x_energy_train_universe, energy_train_mid_mf, x_energy_train)
    energy_train_hig_mf = fuzz.trimf(x_energy_train_universe, (1050, 37800, 37800))
    energy_train_hig = fuzz.interp_membership(x_energy_train_universe, energy_train_hig_mf, x_energy_train)
    
    energy_test_low_mf = fuzz.trimf(x_energy_test_universe, (0, 0, 105))
    energy_test_low = fuzz.interp_membership(x_energy_test_universe, energy_test_low_mf, x_energy_test)
    energy_test_mid_mf = fuzz.trimf(x_energy_test_universe, (0, 105, 210))
    energy_test_mid = fuzz.interp_membership(x_energy_test_universe, energy_test_mid_mf, x_energy_test)
    energy_test_hig_mf = fuzz.trapmf(x_energy_test_universe, (105, 210, 12600, 12600))
    energy_test_hig = fuzz.interp_membership(x_energy_test_universe, energy_test_hig_mf, x_energy_test)
    
    energy_hig = np.maximum(energy_train_hig, energy_test_hig)
    energy_mid = np.maximum(energy_train_mid, energy_test_mid)
    energy_low = np.minimum(energy_train_low, energy_test_low)

    rule1 = np.minimum(score_hig,min(perf_hig, energy_low))
    rule2 = np.minimum(score_mid,max(min(perf_mid, energy_low), min(perf_low, energy_mid)))
    rule3 = np.minimum(score_low,max(perf_low, energy_hig))

    # rule1 = np.minimum(score_1,min(perf_low, energy_hig))
    # rule2 = np.minimum(score_2,min(perf_mid, energy_hig))
    # rule3 = np.minimum(score_3,max(min(perf_hig, energy_hig),min(perf_low, energy_low)))
    # rule4 = np.minimum(score_4,min(perf_mid, energy_low))
    # rule5 = np.minimum(score_5,max(perf_hig, energy_low))

    # Aggregate the rules
    aggregated = np.maximum(rule1, np.maximum(rule2, rule3))
    # aggregated = np.maximum(rule1, np.maximum(rule2, np.maximum(rule3, np.maximum(rule4, rule5))))
    # print(aggregated)
    # Compute the defuzzified output
    if np.unique(aggregated).size == 1:
        if aggregated[0] == 0:
            defuzzified = None
        defuzzified = None
    else:
        defuzzified = fuzz.defuzz(x_score_universe, aggregated, mode)

    scores.append(defuzzified)

energy = results["Energy train"]+results["Energy test"]
energy_n = (energy- min(energy)) / (max(energy) - min(energy))
score_n = (results["Score"]- min(results["Score"])) / (max(results["Score"]) - min(results["Score"]))
results["Frugality score"] = scores
results["Evchenko score"] = results["Score"] - 1/(1+1/(results["Energy train"]+results["Energy test"]))
results["Score WS"] = 0.5*(score_n + (1-energy_n))
results["Score HM"] = 2*(score_n*(1-energy_n))/(4*score_n + (1-energy_n))
# Display the results
print(results)
results.to_csv("output/results_classif_mnist_score.csv", index=False)

# Plot the results
# plt.figure(figsize=(10, 10))
# for clf, marker in zip(results["Classifier"].unique(), cycle(("o", "s", "D", "^", "v", "<", ">", "p", "*"))):
#     plt.scatter(
#         results[results["Classifier"] == clf]["Frugality score"],
#         results[results["Classifier"] == clf]["Energy train"],
#         c=results[results["Classifier"] == clf]["Dataset"],
#         marker=marker,
#         label=clf,
#         alpha=0.5,
#     )
# plt.title("Frugality score vs Energy")
# plt.xlabel("Frugality score")
# plt.ylabel("Energy train")
# plt.legend()
# plt.show()

# results.groupby(["Classifier", "Dataset"]).agg({"Frugality score": ["mean", "std"]}).reset_index().plot(
#     x="Classifier",
#     y="Frugality score",
#     yerr=results.groupby(["Classifier", "Dataset"]).agg({"Frugality score": "std"}).reset_index()["Frugality score"],
#     kind="bar",
#     figsize=(15, 7),
#     title="Average Frugality Score per Classifier and Dataset",
#     xlabel="Classifier",
#     ylabel="Frugality score",
#     legend=True,
#     alpha=0.7,
#     edgecolor="black",
#     linewidth=1,
#     grid=True)
# plt.show()

fig, ax = plt.subplots(figsize=(3.5, 3))
cm = plt.cm.get_cmap('viridis')
scatter = plt.scatter(
    data=results,
    x="Energy train", 
    y="Score", 
    c="Frugality score", 
    cmap=cm, 
    s=50,
    alpha=0.5,
    marker="."
)
scatter.set_clim(0, 100)
plt.colorbar(scatter, label='Score')
plt.xlabel("Energy consumption during training (J)")
plt.ylabel("Accuracy (%)")
plt.show()

# same but only one y label on the left
fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for ax in axs:
    ax.label_outer()
axs[0].plot(x_energy_train_universe, energy_train_low_mf, label='Low')
axs[0].plot(x_energy_train_universe, energy_train_mid_mf, label='Medium')
axs[0].plot(x_energy_train_universe, energy_train_hig_mf, label='High')
axs[0].set_xlabel('Energy during training (J)')
axs[0].set_ylabel('Membership Degree')
axs[1].plot(x_energy_test_universe, energy_test_low_mf, label='Low')
axs[1].plot(x_energy_test_universe, energy_test_mid_mf, label='Medium')
axs[1].plot(x_energy_test_universe, energy_test_hig_mf, label='High')
axs[1].set_xlabel('Energy during testing (J)')
# axs[1].set_ylabel('Membership Degree')
axs[2].plot(x_perf_universe, perf_low_mf, label='Low')
axs[2].plot(x_perf_universe, perf_mid_mf, label='Medium')
axs[2].plot(x_perf_universe, perf_hig_mf, label='High')
axs[2].set_xlabel('Accuracy (%)')
# axs[2].set_ylabel('Membership Degree')
axs[3].plot(x_score_universe, score_low, label='Low')
axs[3].plot(x_score_universe, score_mid, label='Medium')
axs[3].plot(x_score_universe, score_hig, label='High')
axs[3].set_xlabel('Frugality Score')
# axs[3].set_ylabel('Membership Degree')
axs[3].legend(loc='right')
plt.tight_layout()
plt.savefig("output/classification/classifier_mf_comparison.pdf", dpi=300, format='pdf')
plt.show()


cm = plt.get_cmap("viridis")

# make histogram of frugality score for all classifiers
results_grouped = results.groupby("Classifier")["Frugality score"].mean().reset_index()
results_grouped_95confidence = results.groupby("Classifier")["Frugality score"].agg(lambda x: np.percentile(x, 95)).reset_index()
plt.figure(figsize=(10, 5))
ax = plt.bar(results_grouped["Classifier"], results_grouped["Frugality score"], yerr=results_grouped_95confidence["Frugality score"] - results_grouped["Frugality score"], capsize=5, alpha=0.7, edgecolor="black", color=cm(results_grouped["Frugality score"]/100))
# plt.ylim(0, 100)
# plt.legend()
plt.title("Histogram of Frugality Score")
plt.xlabel("Classifier")
plt.ylabel("Frugality Score")
plt.show()

# make histogram of Energy train for all classifiers
results_grouped = results.groupby("Classifier")["Energy train"].mean().reset_index()
results_grouped_95confidence = results.groupby("Classifier")["Energy train"].agg(lambda x: np.percentile(x, 95)).reset_index()
plt.figure(figsize=(10, 5))
plt.bar(results_grouped["Classifier"], results_grouped["Energy train"], yerr=results_grouped_95confidence["Energy train"] - results_grouped["Energy train"], capsize=5, alpha=0.7, edgecolor="black")
# plt.legend()
plt.title("Histogram of Energy train")
plt.xlabel("Classifier")
plt.ylabel("Energy train")
plt.show()

# make histogram of Energy test for all classifiers
results_grouped = results.groupby("Classifier")["Energy test"].mean().reset_index()
results_grouped_95confidence = results.groupby("Classifier")["Energy test"].agg(lambda x: np.percentile(x, 95)).reset_index()
plt.figure(figsize=(10, 5))
plt.bar(results_grouped["Classifier"], results_grouped["Energy test"], yerr=results_grouped_95confidence["Energy test"] - results_grouped["Energy test"], capsize=5, alpha=0.7, edgecolor="black")
# plt.legend()
plt.title("Histogram of Energy test")
plt.xlabel("Classifier")
plt.ylabel("Energy test")
plt.show()

# make histogram of accuracy for all classifiers
results_grouped = results.groupby("Classifier")["Score"].mean().reset_index()
results_grouped_95confidence = results.groupby("Classifier")["Score"].agg(lambda x: np.percentile(x, 95)).reset_index()
plt.figure(figsize=(10, 5))
plt.bar(results_grouped["Classifier"], results_grouped["Score"], yerr=results_grouped_95confidence["Score"] - results_grouped["Score"], capsize=5, alpha=0.7, edgecolor="black")
plt.legend()
plt.title("Histogram of Score")
plt.xlabel("Classifier")
plt.ylabel("Score")
plt.show()


# same but only one y label on the left]:
fig, axs = plt.subplots(4, 1, figsize=(5, 6))
for ax in axs:
    ax.label_outer()
# make histogram of accuracy for all classifiers
results_grouped = results.groupby("Classifier")["Score"].mean().reset_index()
results_grouped_95confidence = results.groupby("Classifier")["Score"].agg(lambda x: np.percentile(x, 95)).reset_index()
axs[0].bar(results_grouped["Classifier"], results_grouped["Score"]*100, yerr=results_grouped_95confidence["Score"] - results_grouped["Score"], capsize=5, alpha=0.7, edgecolor="black")
axs[0].set_ylabel('Accuracy (%)')
axs[0].set_ylim(0, 100)
for i, v in enumerate(results_grouped["Score"]*100):
    if v < max(results_grouped["Score"]*100)*0.5:
        axs[0].text(i, v + max(results_grouped["Score"]*100)*0.1, f"{v:.2f}", ha='center', fontsize=8)
    else:
        axs[0].text(i, v - max(results_grouped["Score"]*100)*0.2, f"{v:.2f}", ha='center', fontsize=8)
# axs[0].set_xlabel('Classifier')
# make histogram of Energy train for all classifiers
results_grouped = results.groupby("Classifier")["Energy train"].mean().reset_index()
results_grouped_95confidence = results.groupby("Classifier")["Energy train"].agg(lambda x: np.percentile(x, 95)).reset_index()
axs[1].bar(results_grouped["Classifier"], results_grouped["Energy train"], yerr=results_grouped_95confidence["Energy train"] - results_grouped["Energy train"], capsize=5, alpha=0.7, edgecolor="black")
axs[1].set_ylabel('Energy in\ntraining (J)')
for i, v in enumerate(results_grouped["Energy train"]):
    if v < max(results_grouped["Energy train"])*0.5:
        axs[1].text(i, v + max(results_grouped["Energy train"])*0.1, f"{v:.0f}", ha='center', fontsize=8)
    else:
        axs[1].text(i, v - max(results_grouped["Energy train"])*0.2, f"{v:.0f}", ha='center', fontsize=8)
# axs[1].set_xlabel('Classifier')
# make histogram of Energy test for all classifiers
results_grouped = results.groupby("Classifier")["Energy test"].mean().reset_index()
results_grouped_95confidence = results.groupby("Classifier")["Energy test"].agg(lambda x: np.percentile(x, 95)).reset_index()
axs[2].bar(results_grouped["Classifier"], results_grouped["Energy test"], yerr=results_grouped_95confidence["Energy test"] - results_grouped["Energy test"], capsize=5, alpha=0.7, edgecolor="black")
axs[2].set_ylabel('Energy in\ntesting (J)')
for i, v in enumerate(results_grouped["Energy test"]):
    if v < max(results_grouped["Energy test"])*0.5:
        axs[2].text(i, v + max(results_grouped["Energy test"])*0.1, f"{v:.0f}", ha='center', fontsize=8)
    else:
        axs[2].text(i, v - max(results_grouped["Energy test"])*0.2, f"{v:.0f}", ha='center', fontsize=8)
# axs[2].set_xlabel('Classifier')
# make histogram of frugality score for all classifiers
results_grouped = results.groupby("Classifier")["Frugality score"].mean().reset_index()
results_grouped_95confidence = results.groupby("Classifier")["Frugality score"].agg(lambda x: np.percentile(x, 95)).reset_index()
axs[3].bar(results_grouped["Classifier"], results_grouped["Frugality score"], yerr=results_grouped_95confidence["Frugality score"] - results_grouped["Frugality score"], capsize=5, alpha=0.7, edgecolor="black", color=cm(results_grouped["Frugality score"]/100))
axs[3].set_ylabel(f'Frugality Score\n(mode {mode})')
axs[3].set_xlabel('Classifier')
axs[3].set_ylim(0, 100)
for i, v in enumerate(results_grouped["Frugality score"]):
    if v < max(results_grouped["Frugality score"])*0.5:
        axs[3].text(i, v + 100*0.1, f"{v:.2f}", ha='center', fontsize=8)
    else:
        axs[3].text(i, v - 100*0.1, f"{v:.2f}", ha='center', fontsize=8)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
fig.align_ylabels(axs)
plt.savefig(f"output/classification/classifier_bar_comparison_{mode}.pdf", dpi=300, format='pdf')
plt.show()


# Make 3d plot
import plotly.express as px

fig = px.scatter_3d(results, x="Energy test", y="Energy train", z="Score", color="Classifier", symbol="Dataset", hover_name="Setup", size="Frugality score", size_max=10, opacity=0.7)
fig.update_traces(marker=dict(size=5))
fig.update_layout(title="Frugality score vs Energy train vs Score", scene=dict(
                    xaxis_title='Energy test',
                    yaxis_title='Energy train',
                    zaxis_title='Score'))
fig.show()


print(results[["Classifier", "Dataset", "Frugality score"]].groupby(["Classifier", "Dataset"]).min().unstack())
print(results[["Classifier", "Dataset", "Frugality score"]].groupby(["Classifier", "Dataset"]).quantile(q=0.25).unstack())
print(results[["Classifier", "Dataset", "Frugality score"]].groupby(["Classifier", "Dataset"]).quantile(q=0.5).unstack())
print(results[["Classifier", "Dataset", "Frugality score"]].groupby(["Classifier", "Dataset"]).quantile(q=0.75).unstack())
print(results[["Classifier", "Dataset", "Frugality score"]].groupby(["Classifier", "Dataset"]).max().unstack())


# score_mean = results[["Classifier", "Score"]].groupby(["Classifier"]).mean().unstack()
# energy_train_mean = results[["Classifier", "Energy train"]].groupby(["Classifier"]).mean().unstack()
# energy_test_mean = results[["Classifier", "Energy test"]].groupby(["Classifier"]).mean().unstack()
results["Energy total"] = results["Energy train"] + results["Energy test"]
mean_results = results[["Classifier", "Score", "Energy train", "Energy test", "Energy total"]].groupby(["Classifier"]).mean().unstack()
# mean_results['Energy total'] = mean_results["Energy train"] + mean_results["Energy test"]

mean_results_norm = mean_results.copy()
mean_results_norm["Energy train"] = (mean_results["Energy train"] - mean_results["Energy train"].min()) / (mean_results["Energy train"].max() - mean_results["Energy train"].min())
mean_results_norm["Energy test"] = (mean_results["Energy test"] - mean_results["Energy test"].min()) / (mean_results["Energy test"].max() - mean_results["Energy test"].min())
mean_results_norm["Score"] = (mean_results["Score"] - mean_results["Score"].min()) / (mean_results["Score"].max() - mean_results["Score"].min())
mean_results_norm["Energy total"] = (mean_results["Energy total"]) / (mean_results["Energy total"].max())

print("Score WS")

for epsilon in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    print(f"Epsilon: {epsilon}")
    print((epsilon*(1- mean_results_norm["Energy total"]) + (1-epsilon)*mean_results_norm["Score"]).transpose())

print("Score HM")

for kappa in [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    print(f"Kappa: {kappa}")
    print(((1 + kappa**2)*(1- mean_results_norm["Energy total"])*mean_results_norm["Score"]/((1- mean_results_norm["Energy total"]) + (kappa**2)*mean_results_norm["Score"])).transpose())

print("Evchenko score")
for w in [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    print(f"w: {w}")
    print((mean_results["Score"] - w/(1+1/(mean_results["Energy total"]/(3.6*1e6)))).transpose())

print(mean_results_norm["Energy total"])
print(mean_results_norm["Score"])