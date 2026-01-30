import skfuzzy as fuzz
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

n_points = 21
plotly_use = False

# # empty dataframe to store results
# df = pd.DataFrame(columns=['Accuracy', 'Energy_Train', 'Energy_Test', 'Score', 'Defuzzification_Mode'])
# df_aggregated = pd.DataFrame(columns=['Accuracy', 'Energy_Train', 'Energy_Test', 'Score_Membership', 'Score_Value'])

# energy_train = np.linspace(0, 15, n_points)
# energy_test = np.linspace(0, 6e-4, n_points)
# accuracy = np.linspace(0, 100, n_points)
# score = np.linspace(0, 100, 100)

# accuracy_low_mf = fuzz.trimf(accuracy, (0, 0, 50))
# accuracy_mid_mf = fuzz.trimf(accuracy, (0, 50, 100))
# accuracy_hig_mf = fuzz.trimf(accuracy, (50, 100, 100))

# energy_train_low_mf = fuzz.trimf(energy_train, (0, 0, 5))
# energy_train_mid_mf = fuzz.trimf(energy_train, (0, 5, 15))
# energy_train_hig_mf = fuzz.trimf(energy_train, (5, 15, 15))

# energy_test_low_mf = fuzz.trimf(energy_test, (0, 0, 3e-4))
# energy_test_mid_mf = fuzz.trimf(energy_test, (0, 3e-4, 6e-4))
# energy_test_hig_mf = fuzz.trimf(energy_test, (3e-4, 6e-4, 6e-4))

# score_1_mf = fuzz.trimf(score, (0, 0, 25))
# score_2_mf = fuzz.trimf(score, (0, 25, 50))
# score_3_mf = fuzz.trimf(score, (25, 50, 75))
# score_4_mf = fuzz.trimf(score, (50, 75, 100))
# score_5_mf = fuzz.trimf(score, (75, 100, 100))

# for i_acc in tqdm(range(1,len(accuracy)), position=0, leave=False):
#     for i_et in tqdm(range(1,len(energy_train)), position=1, leave=False):
#         for i_ee in tqdm(range(1,len(energy_test)), position=2, leave=False):
#             accuracy_i = accuracy[i_acc]
#             energy_train_i = energy_train[i_et]
#             energy_test_i = energy_test[i_ee]

#             # Fuzzification
#             accuracy_low = fuzz.interp_membership(accuracy, accuracy_low_mf, accuracy_i)
#             accuracy_mid = fuzz.interp_membership(accuracy, accuracy_mid_mf, accuracy_i)
#             accuracy_hig = fuzz.interp_membership(accuracy, accuracy_hig_mf, accuracy_i)

#             energy_train_low = fuzz.interp_membership(energy_train, energy_train_low_mf, energy_train_i)
#             energy_train_mid = fuzz.interp_membership(energy_train, energy_train_mid_mf, energy_train_i)
#             energy_train_hig = fuzz.interp_membership(energy_train, energy_train_hig_mf, energy_train_i)

#             energy_test_low = fuzz.interp_membership(energy_test, energy_test_low_mf, energy_test_i)
#             energy_test_mid = fuzz.interp_membership(energy_test, energy_test_mid_mf, energy_test_i)
#             energy_test_hig = fuzz.interp_membership(energy_test, energy_test_hig_mf, energy_test_i)

#             # print(f"Accuracy: {accuracy_i}, Energy Train: {energy_train_i}, Energy Test: {energy_test_i}")
#             # print(f"Memberships - Acc Low: {accuracy_low}, Acc Mid: {accuracy_mid}, Acc High: {accuracy_hig}")
#             # print(f"Memberships - Energy Train Low: {energy_train_low}, Mid: {energy_train_mid}, High: {energy_train_hig}")
#             # print(f"Memberships - Energy Test Low: {energy_test_low}, Mid: {energy_test_mid}, High: {energy_test_hig}")

#             rule1 = np.minimum(score_1_mf,max(
#                 min(accuracy_low, energy_train_hig, energy_test_hig),
#                 max(
#                 min(accuracy_low, energy_train_hig, energy_test_mid),
#                 max(
#                 min(accuracy_low, energy_train_hig, energy_test_low),
#                 max(
#                 min(accuracy_low, energy_train_mid, energy_test_hig),
#                 max(
#                 min(accuracy_low, energy_train_mid, energy_test_mid),
#                 max(
#                 min(accuracy_low, energy_train_low, energy_test_hig),
#                 max(
#                 min(accuracy_mid, energy_train_hig, energy_test_hig),
#                 max(
#                 min(accuracy_mid, energy_train_hig, energy_test_mid),
#                 max(
#                 min(accuracy_mid, energy_train_mid, energy_test_hig),
#                 min(accuracy_hig, energy_train_hig, energy_test_hig)
#                 ))))))))
#                 ))
#             rule2 = np.minimum(score_2_mf,max(
#                 min(accuracy_low, energy_train_mid, energy_test_low),
#                 max(
#                 min(accuracy_low, energy_train_low, energy_test_mid),
#                 max(
#                 min(accuracy_mid, energy_train_hig, energy_test_low),
#                 max(
#                 min(accuracy_mid, energy_train_mid, energy_test_mid),
#                 max(
#                 min(accuracy_mid, energy_train_low, energy_test_hig),
#                 max(
#                 min(accuracy_hig, energy_train_mid, energy_test_low),
#                 min(accuracy_hig, energy_train_low, energy_test_mid)
#                 )))))
#                 ))
#             rule3 = np.minimum(score_3_mf,max(
#                 min(accuracy_hig, energy_train_hig, energy_test_low),
#                 max(
#                 min(accuracy_mid, energy_train_mid, energy_test_low),
#                 max(
#                 min(accuracy_hig, energy_train_mid, energy_test_mid),
#                 max(
#                 min(accuracy_low, energy_train_low, energy_test_low),
#                 max(
#                 min(accuracy_mid, energy_train_low, energy_test_mid),
#                 min(accuracy_hig, energy_train_low, energy_test_hig)
#                 ))))
#                 ))
#             rule4 = np.minimum(score_4_mf,max(
#                 min(accuracy_hig, energy_train_low, energy_test_mid),
#                 max(
#                 min(accuracy_hig, energy_train_mid, energy_test_low),
#                 min(accuracy_mid, energy_train_low, energy_test_low)
#                 )
#                 ))
#             rule5 = np.minimum(score_5_mf,min(accuracy_hig, energy_train_low, energy_test_low))

#             aggregated = np.maximum(rule1, np.maximum(rule2, np.maximum(rule3, np.maximum(rule4, rule5))))
#             # print(aggregated)
#             aggregated = np.fmax(aggregated, 0)
#             # print(aggregated)
#             if len(np.unique(aggregated)) != 1:
#                 for j in range(len(score)):
#                     df_aggregated = pd.concat([df_aggregated, pd.DataFrame({'Accuracy': [accuracy_i], 'Energy_Train': [energy_train_i], 'Energy_Test': [energy_test_i], 'Score_Membership': [aggregated[j]], 'Score_Value': [score[j]]})], ignore_index=True)
#                 # Defuzzification
#                 for mode in ['centroid', 'bisector', 'mom', 'som', 'lom']:
#                     score_i = fuzz.defuzz(score, aggregated, mode)
#                     df = pd.concat([df, pd.DataFrame({'Accuracy': [accuracy_i], 'Energy_Train': [energy_train_i], 'Energy_Test': [energy_test_i], 'Score': [score_i], 'Defuzzification_Mode': [mode]})], ignore_index=True)

# df.to_csv('data/theory/fuzzy_logic_results.csv', index=False)
# df_aggregated.to_csv('data/theory/fuzzy_logic_aggregated.csv', index=False)

if plotly_use == True:
    df = pd.read_csv('data/theory/fuzzy_logic_results.csv')
    df_aggregated = pd.read_csv('data/theory/fuzzy_logic_aggregated.csv')
    # 3D Scatter plot using Plotly
    import plotly.graph_objects as go
    # fig = px.scatter_3d(df, x='Accuracy', y='Energy_Train', z='Energy_Test', color='Score', title='Fuzzy Logic Score Visualization', color_continuous_scale='Viridis')
    # fig.update_layout(scene = dict(
    #                     xaxis_title='Accuracy (%)',
    #                     yaxis_title='Energy Train (kWh)',
    #                     zaxis_title='Energy Test (kWh)'),
    #                   )
    # fig.show()



    # Interactive plot with Defuzzification Mode filter
    fig = go.Figure()
    for mode in df['Defuzzification_Mode'].unique():
        df_mode = df[df['Defuzzification_Mode'] == mode]
        fig.add_trace(go.Scatter3d(
            x=df_mode['Accuracy'],
            y=df_mode['Energy_Train'],
            z=df_mode['Energy_Test'],
            mode='markers',
            marker=dict(
                size=int(50/n_points),
                color=df_mode['Score'],
                colorscale='ylorrd_r',
                colorbar=dict(title='Score'),
                opacity=1,
            ),
            hovertext=df_mode.apply(lambda row: f"Score: {row['Score']}", axis=1),
            name=mode
        ))
    fig.update_layout(
        # hovermode='x unified',
        title='Fuzzy Logic Score Visualization by Defuzzification Mode',
        scene=dict(
            xaxis_title='Accuracy (%)',
            yaxis_title='Energy Train (kWh)',
            zaxis_title='Energy Test (kWh)'
        ),
        updatemenus=[
            dict(
                type='dropdown',
                active=0,
                buttons=[
                    dict(
                        label=mode,
                        method='update',
                        args=[{'visible': [trace.name == mode for trace in fig.data]},
                            {'title': f'Fuzzy Logic Score Visualization - {mode}'}]
                    ) for mode in df['Defuzzification_Mode'].unique()
                ],
                direction='down',
                showactive=True,
            )
        ]
    )
    fig.show()


    fig = go.Figure()
    for mode in df['Defuzzification_Mode'].unique():
        df_mode = df[df['Defuzzification_Mode'] == mode]
        fig.add_trace(go.Volume(
            x=df_mode['Accuracy'].values.flatten(),
            y=df_mode['Energy_Train'].values.flatten(),
            z=df_mode['Energy_Test'].values.flatten(),
            value=df_mode['Score'].values.flatten(),
            isomin=0,
            isomax=100,
            opacity=0.5, # needs to be small to see through all surfaces
            surface_count=100, # needs to be a large number for good volume rendering
            name=mode))
        # fig.show()
    fig.update_layout(
        # hovermode='x unified',
        title='Fuzzy Logic Score Visualization by Defuzzification Mode',
        scene=dict(
            xaxis_title='Accuracy (%)',
            yaxis_title='Energy Train (kWh)',
            zaxis_title='Energy Test (kWh)'
        ),
        updatemenus=[
            dict(
                type='dropdown',
                active=0,
                buttons=[
                    dict(
                        label=mode,
                        method='update',
                        args=[{'visible': [trace.name == mode for trace in fig.data]},
                            {'title': f'Fuzzy Logic Score Visualization - {mode}'}]
                    ) for mode in df['Defuzzification_Mode'].unique()
                ],
                direction='down',
                showactive=True,
            )
        ]
    )
    fig.show()



    # Create figure
    fig = go.Figure()
    # Interactive plot with Score Membership filter
    for accuracy in df_aggregated['Accuracy'].unique():
        for energy_train in df_aggregated['Energy_Train'].unique():
            for energy_test in df_aggregated['Energy_Test'].unique():
                df_subset = df_aggregated[(df_aggregated['Accuracy'] == accuracy) &
                                        (df_aggregated['Energy_Train'] == energy_train) &
                                        (df_aggregated['Energy_Test'] == energy_test)]
                fig.add_trace(go.Scatter(
                    x=df_subset['Score_Value'],
                    y=df_subset['Score_Membership'],
                    mode='lines',
                    name=f'Acc: {accuracy:2f}, E_Train: {energy_train:2f}, E_Test: {energy_test:2f}'
                )
                )
    # Make only the first trace visible
    for i, trace in enumerate(fig.data):
        trace.visible = (i == 0)
    # Create dropdown menu for filtering on accuracy, energy_train, energy_test
    acc_buttons = []
    for accuracy in df_aggregated['Accuracy'].unique():
        label = f'Acc: {accuracy:2f}'
        visibility = [label in trace.name for trace in fig.data]
        acc_buttons.append(
            dict(
                label=label,
                method='update',
                args=[{'visible': visibility},
                    {'title': f'Score Membership - {label}'}]
                )
        )
    energy_train_buttons = []
    for energy_train in df_aggregated['Energy_Train'].unique():
        label = f'E_Train: {energy_train:2f}'
        visibility = [label in trace.name for trace in fig.data]
        energy_train_buttons.append(
            dict(
                label=label,
                method='update',
                args=[{'visible': visibility},
                    {'title': f'Score Membership - {label}'}]
                )
        )
    energy_test_buttons = []
    for energy_test in df_aggregated['Energy_Test'].unique():
        label = f'E_Test: {energy_test:2f}'
        visibility = [label in trace.name for trace in fig.data]
        energy_test_buttons.append(
            dict(
                label=label,
                method='update',
                args=[{'visible': visibility},
                    {'title': f'Score Membership - {label}'}]
                )
        )
    fig.update_layout(
        title='Fuzzy Logic Score Membership Visualization',
        xaxis_title='Score Value',
        yaxis_title='Score Membership',
        updatemenus=[
            dict(
                type='dropdown',
                active=0,
                buttons=acc_buttons,
                direction='down',
                showactive=True,
                pad={"r": 10, "t": 10},
                x=0.1,
                xanchor="left",
                y=1.08,
                yanchor="bottom"
            ),
            dict(
                type='dropdown',
                active=0,
                buttons=energy_train_buttons,
                direction='down',
                showactive=True,
                pad={"r": 10, "t": 10},
                x=0.5,
                xanchor="left",
                y=1.08,
                yanchor="bottom"
            ),
            dict(
                type='dropdown',
                active=0,
                buttons=energy_test_buttons,
                direction='down',
                showactive=True,
                pad={"r": 10, "t": 10},
                x=0.9,
                xanchor="left",
                y=1.08,
                yanchor="bottom"
            )
        ]
    )
    fig.show()

else:
    # Plotting using Matplotlib
    n_points = 1000

    # Plot membership functions
    energy_train = np.linspace(0, 15, n_points)
    energy_test = np.linspace(0, 6e-4, n_points)
    accuracy = np.linspace(0, 100, n_points)
    score = np.linspace(0, 100, n_points)

    accuracy_low_mf = fuzz.trimf(accuracy, (0, 0, 50))
    accuracy_mid_mf = fuzz.trimf(accuracy, (0, 50, 100))
    accuracy_hig_mf = fuzz.trimf(accuracy, (50, 100, 100))

    energy_train_low_mf = fuzz.trimf(energy_train, (0, 0, 5))
    energy_train_mid_mf = fuzz.trimf(energy_train, (0, 5, 15))
    energy_train_hig_mf = fuzz.trimf(energy_train, (5, 15, 15))

    energy_test_low_mf = fuzz.trimf(energy_test, (0, 0, 3e-4))
    energy_test_mid_mf = fuzz.trimf(energy_test, (0, 3e-4, 6e-4))
    energy_test_hig_mf = fuzz.trimf(energy_test, (3e-4, 6e-4, 6e-4))

    score_1_mf = fuzz.trimf(score, (0, 0, 25))
    score_2_mf = fuzz.trimf(score, (0, 25, 50))
    score_3_mf = fuzz.trimf(score, (25, 50, 75))
    score_4_mf = fuzz.trimf(score, (50, 75, 100))
    score_5_mf = fuzz.trimf(score, (75, 100, 100))

    fig, axs = plt.subplots(4, 1, figsize=(5, 6), sharey=True)
    axs[0].plot(accuracy, accuracy_low_mf, label='Low')
    axs[0].plot(accuracy, accuracy_mid_mf, label='Medium')
    axs[0].plot(accuracy, accuracy_hig_mf, label='High')
    # axs[0].set_title('Accuracy Membership Functions')
    axs[0].set_xlabel('Accuracy (%)')
    # axs[0].set_ylabel('Membership Degree')
    # axs[0].legend(loc='right')
    axs[1].plot(energy_train, energy_train_low_mf, label='Low')
    axs[1].plot(energy_train, energy_train_mid_mf, label='Medium')
    axs[1].plot(energy_train, energy_train_hig_mf, label='High')
    # axs[1].set_title('Energy Train Membership Functions')
    axs[1].set_xlabel('Energy in training (kWh)')
    # axs[1].set_ylabel('Membership Degree')
    # axs[1].legend(loc='right')
    axs[2].plot(energy_test, energy_test_low_mf, label='Low')
    axs[2].plot(energy_test, energy_test_mid_mf, label='Medium')
    axs[2].plot(energy_test, energy_test_hig_mf, label='High')
    # axs[2].set_title('Energy Test Membership Functions')
    axs[2].set_xlabel('Energy in testing (kWh)')
    # axs[2].set_ylabel('Membership Degree')
    axs[2].ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    axs[2].legend(loc='right')
    axs[3].plot(score, score_1_mf, label='1')
    axs[3].plot(score, score_2_mf, label='2')
    axs[3].plot(score, score_3_mf, label='3')
    axs[3].plot(score, score_4_mf, label='4')
    axs[3].plot(score, score_5_mf, label='5')
    # axs[3].set_title('Score Membership Functions')
    axs[3].set_xlabel('Frugality Score')
    # axs[3].set_ylabel('Membership Degree')
    axs[3].legend(loc='right')
    fig.supylabel('Membership Degree')
    plt.tight_layout()
    plt.savefig('output/fuzzy_logic_membership_functions.pdf', dpi=300, format='pdf')
    plt.show()

    # same but only one y label on the left
    fig, axs = plt.subplots(3, 1, figsize=(4, 3), sharey=True)
    for ax in axs:
        ax.label_outer()
    axs[0].plot(energy_train, energy_train_low_mf, label='Low')
    axs[0].plot(energy_train, energy_train_mid_mf, label='Medium')
    axs[0].plot(energy_train, energy_train_hig_mf, label='High')
    axs[0].set_xlabel('Energy in training (kWh)')
    axs[0].set_ylabel('Membership Degree')
    axs[1].plot(energy_test, energy_test_low_mf, label='Low')
    axs[1].plot(energy_test, energy_test_mid_mf, label='Medium')
    axs[1].plot(energy_test, energy_test_hig_mf, label='High')
    axs[1].set_xlabel('Energy in testing (kWh)')
    axs[1].set_ylabel('Membership Degree')
    axs[2].plot(accuracy, accuracy_low_mf, label='Low')
    axs[2].plot(accuracy, accuracy_mid_mf, label='Medium')
    axs[2].plot(accuracy, accuracy_hig_mf, label='High')
    axs[2].set_xlabel('Accuracy (%)')
    axs[2].set_ylabel('Membership Degree')
    axs[2].legend(loc='right')
    plt.tight_layout()
    plt.show()
   
    # PLot 3D heatmap of results
    for mode in ['centroid', 'bisector', 'mom', 'som', 'lom']:
        fig, axs = plt.subplots(1, 1, figsize=(6, 5), subplot_kw={"projection": "3d"})
        df = pd.read_csv('data/theory/fuzzy_logic_results.csv')
        sc = axs.scatter(
            data = df[df['Defuzzification_Mode'] == mode],
            ys='Energy_Train', 
            xs='Energy_Test', 
            zs='Accuracy', 
            c='Score', 
            cmap='viridis', 
            s=10,
            alpha=1,
            marker='o')
        sc.set_clim(0, 100)
        plt.colorbar(sc, label=f'Frugality Score (Mode {mode.upper()})', pad=0.1, shrink=0.8)
        axs.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
        axs.set_xlabel('Energy in testing (kWh)')
        axs.set_ylabel('Energy in training (kWh)')
        axs.set_zlabel('Accuracy (%)', labelpad=5)
        plt.tight_layout()
        plt.savefig(f'output/fuzzy_logic_score_distribution_{mode}.pdf', dpi=300, format='pdf')
        plt.show()
