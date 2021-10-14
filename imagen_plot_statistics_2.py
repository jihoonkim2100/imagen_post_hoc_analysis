#################################################################################
""" IMAGEN Instrument Summary Statistic """
# Author: JiHoon Kim, <jihoon.kim@fu-berlin.de>, 12th October 2021
#
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, bartlett
from statannot import add_stat_annotation
import warnings
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")

def ml_plot(train, test, col):
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(3*len(col), 18))
    # 0,0. Training set
    ax0 = sns.countplot(data=train, x='Model',
                       hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                       ax=axes[0,0], palette='Set2')
    axes[0,0].set_title(f'Training Set (n = {len(train["Session"].tolist())//4}) by MODEL')
        
    axes[0,0].legend(title='Model Prediction',loc='lower center')
    
    for p in ax0.patches:
        ax0.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 0,1. Model Prediction
    ax1 = sns.violinplot(data=train, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                         ax = axes[0,1], palette="Set3")
        
    axes[0,1].set_title(f'{col}, Training Set')
    
    axes[0,1].legend(title='Model Prediction',loc='lower center')
        
    add_stat_annotation(ax1, data=train, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                        box_pairs=[(("GB","TP & FP"),("GB","TN & FN")),
                                   (("LR","TP & FP"),("LR","TN & FN")),
                                   (("SVM-lin","TP & FP"),("SVM-lin","TN & FN")),
                                   (("SVM-rbf","TP & FP"),("SVM-rbf","TN & FN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 0,2. Prediction TF
    ax2 = sns.violinplot(data=train, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Predict TF', hue_order=['TP & TN', 'FP & FN'],
                         ax = axes[0,2], palette="Set1")
            
    axes[0,2].set_title(f'{col}, Training Set')
    
    axes[0,2].legend(title='Prediction TF',loc='lower center')
        
    add_stat_annotation(ax2, data=train, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Predict TF', hue_order=['TP & TN', 'FP & FN'],
                        box_pairs=[(("GB","TP & TN"),("GB","FP & FN")),
                                   (("LR","TP & TN"),("LR","FP & FN")),
                                   (("SVM-lin","TP & TN"),("SVM-lin","FP & FN")),
                                   (("SVM-rbf","TP & TN"),("SVM-rbf","FP & FN"))],
                        loc='inside', verbose=2, line_height=0.1)
        
    # 0,3. Holdout set
    ax3 = sns.countplot(data=test, x='Model',
                        hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                        ax=axes[0,3], palette='Set2')
    axes[0,3].set_title(f'Holdout Set (n = {len(test["Session"].tolist())//4}) by MODEL')
    
    axes[0,3].legend(title='Model Prediction',loc='lower center')
    
    for p in ax3.patches:
        ax3.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 0,4. Model Prediction
    ax4 = sns.violinplot(data=test, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                         ax = axes[0,4], palette="Set3")
        
    axes[0,4].set_title(f'{col}, Holdout Set')
    
    axes[0,4].legend(title='Model Prediction',loc='lower center')
        
    add_stat_annotation(ax4, data=test, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                        box_pairs=[(("GB","TP & FP"),("GB","TN & FN")),
                                   (("LR","TP & FP"),("LR","TN & FN")),
                                   (("SVM-lin","TP & FP"),("SVM-lin","TN & FN")),
                                   (("SVM-rbf","TP & FP"),("SVM-rbf","TN & FN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 0,5. Prediction TF
    ax5 = sns.violinplot(data=test, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Sex', hue_order=['Male', 'Female'],
                         ax = axes[0,5], palette="Set1")
        
    axes[0,5].set_title(f'{col}, Holdout Set')
    
    axes[0,5].legend(title='Prediction TF',loc='lower center')
        
    add_stat_annotation(ax5, data=test, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Predict TF', hue_order=['TP & TN', 'FP & FN'],
                        box_pairs=[(("GB","TP & TN"),("GB","FP & FN")),
                                   (("LR","TP & TN"),("LR","FP & FN")),
                                   (("SVM-lin","TP & TN"),("SVM-lin","FP & FN")),
                                   (("SVM-rbf","TP & TN"),("SVM-rbf","FP & FN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 1,0. Training set
    ax0 = sns.countplot(data=train, x='Model',
                        hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
                        ax=axes[1,0], palette='Set2')
    axes[1,0].set_title(f'Training Set (n = {len(train["Session"].tolist())//4}) by MODEL')
        
    axes[1,0].legend(title='Model Prediction',loc='lower center')
    
    for p in ax0.patches:
        ax0.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 1,1. Model Prediction P
    train_P = train[train['Model PN']=='TP & FP']
    ax1 = sns.violinplot(data=train_P, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TP', 'FP'],
                         ax = axes[1,1], palette="Set3")
        
    axes[1,1].set_title(f'{col}, Training Set')
    
    axes[1,1].legend(title='Prediction Positive',loc='lower center')
        
    add_stat_annotation(ax1, data=train_P, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TP', 'FP'],
                        box_pairs=[(("GB","TP"),("GB","FP")),
                                   (("LR","TP"),("LR","FP")),
                                   (("SVM-lin","TP"),("SVM-lin","FP")),
                                   (("SVM-rbf","TP"),("SVM-rbf","FP"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 1,2. Model Prediction N
    train_N = train[train['Model PN']=='TN & FN']
    ax2 = sns.violinplot(data=train_N, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TN', 'FN'],
                         ax = axes[1,2], palette="Set1")
            
    axes[1,2].set_title(f'{col}, Training Set')
    
    axes[1,2].legend(title='Prediction Negative',loc='lower center')
        
    add_stat_annotation(ax2, data=train_N, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TN', 'FN'],
                        box_pairs=[(("GB","TN"),("GB","FN")),
                                   (("LR","TN"),("LR","FN")),
                                   (("SVM-lin","TN"),("SVM-lin","FN")),
                                   (("SVM-rbf","TN"),("SVM-rbf","FN"))],
                        loc='inside', verbose=2, line_height=0.1)
        
    # 1,3. Holdout set
    ax3 = sns.countplot(data=test, x='Model',
                        hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
                        ax=axes[1,3], palette='Set2')
    axes[1,3].set_title(f'Holdout Set (n = {len(test["Session"].tolist())//4}) by MODEL')
    
    axes[1,3].legend(title='Model Prediction',loc='lower center')
    
    for p in ax3.patches:
        ax3.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 1,4. Model Prediction P
    test_P = test[test['Model PN']=='TP & FP']
    ax4 = sns.violinplot(data=test_P, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TP', 'FP'],
                         ax = axes[1,4], palette="Set3")
        
    axes[1,4].set_title(f'{col}, Holdout Set')
    
    axes[1,4].legend(title='Prediction Positive',loc='lower center')
        
    add_stat_annotation(ax4, data=test_P, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TP', 'FP'],
                        box_pairs=[(("GB","TP"),("GB","FP")),
                                   (("LR","TP"),("LR","FP")),
                                   (("SVM-lin","TP"),("SVM-lin","FP")),
                                   (("SVM-rbf","TP"),("SVM-rbf","FP"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 1,5. Model Prediction N
    test_N = test[test['Model PN']=='TN & FN']
    ax5 = sns.violinplot(data=test_N, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TN', 'FN'],
                         ax = axes[1,5], palette="Set1")
        
    axes[1,5].set_title(f'{col}, Holdout Set')
    
    axes[1,5].legend(title='Prediction Negative',loc='lower center')
        
    add_stat_annotation(ax5, data=test_N, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TN', 'FN'],
                        box_pairs=[(("GB","TN"),("GB","FN")),
                                   (("LR","TN"),("LR","FN")),
                                   (("SVM-lin","TN"),("SVM-lin","FN")),
                                   (("SVM-rbf","TN"),("SVM-rbf","FN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 2,0. Training set
    ax0 = sns.countplot(data=train, x='Model',
                        hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
                        ax=axes[2,0], palette='Set2')
    axes[2,0].set_title(f'Training Set (n = {len(train["Session"].tolist())//4}) by MODEL')
        
    axes[2,0].legend(title='Model Prediction',loc='lower center')
    
    for p in ax0.patches:
        ax0.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 2,1. Prediction T
    train_T = train[train['Predict TF']=='TP & TN']
    ax1 = sns.violinplot(data=train_T, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TP', 'TN'],
                         ax = axes[2,1], palette="Set3")
        
    axes[2,1].set_title(f'{col}, Training Set')
    
    axes[2,1].legend(title='Prediction True',loc='lower center')
        
    add_stat_annotation(ax1, data=train_T, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TP', 'TN'],
                        box_pairs=[(("GB","TP"),("GB","TN")),
                                   (("LR","TP"),("LR","TN")),
                                   (("SVM-lin","TP"),("SVM-lin","TN")),
                                   (("SVM-rbf","TP"),("SVM-rbf","TN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 2,2. Prediction F
    train_F = train[train['Predict TF']=='FP & FN']
    ax2 = sns.violinplot(data=train_F, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['FP', 'FN'],
                         ax = axes[2,2], palette="Set1")
            
    axes[2,2].set_title(f'{col}, Training Set')
    
    axes[2,2].legend(title='Prediction False',loc='lower center')
        
    add_stat_annotation(ax2, data=train_F, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['FP', 'FN'],
                        box_pairs=[(("GB","FP"),("GB","FN")),
                                   (("LR","FP"),("LR","FN")),
                                   (("SVM-lin","FP"),("SVM-lin","FN")),
                                   (("SVM-rbf","FP"),("SVM-rbf","FN"))],
                        loc='inside', verbose=2, line_height=0.1)
        
    # 2,3. Holdout set
    ax3 = sns.countplot(data=test, x='Model',
                        hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
                        ax=axes[2,3], palette='Set2')
    axes[2,3].set_title(f'Holdout Set (n = {len(test["Session"].tolist())//4}) by MODEL')
    
    axes[2,3].legend(title='Model Prediction',loc='lower center')
    
    for p in ax3.patches:
        ax3.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 2,4. Prediction T
    test_T = test[test['Predict TF']=='TP & TN']
    ax4 = sns.violinplot(data=test_T, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TP', 'TN'],
                         ax = axes[2,4], palette="Set3")
        
    axes[2,4].set_title(f'{col}, Holdout Set')
    
    axes[2,4].legend(title='Prediction True',loc='lower center')
        
    add_stat_annotation(ax4, data=test_T, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TP', 'TN'],
                        box_pairs=[(("GB","TP"),("GB","TN")),
                                   (("LR","TP"),("LR","TN")),
                                   (("SVM-lin","TP"),("SVM-lin","TN")),
                                   (("SVM-rbf","TP"),("SVM-rbf","TN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 2,5. Prediction F
    test_F = test[test['Predict TF']=='FP & FN']
    ax5 = sns.violinplot(data=test_F, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['FP', 'FN'],
                         ax = axes[2,5], palette="Set1")
        
    axes[2,5].set_title(f'{col}, Holdout Set')
    
    axes[2,5].legend(title='Prediction False',loc='lower center')
        
    add_stat_annotation(ax5, data=test_F, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['FP', 'FN'],
                        box_pairs=[(("GB","FP"),("GB","FN")),
                                   (("LR","FP"),("LR","FN")),
                                   (("SVM-lin","FP"),("SVM-lin","FN")),
                                   (("SVM-rbf","FP"),("SVM-rbf","FN"))],
                        loc='inside', verbose=2, line_height=0.1)
        
    
#################################################################################
# fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(2*len(col), 5))
# # 0. Training set
# ax0 = sns.countplot(data=train, x='Model',
#                    hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
#                    ax=axes[0], palette='Set2')
# axes[0].set_title(f'Training Set (n = {len(train["Session"].tolist())//4}) by MODEL')
    
# axes[0].legend(title='Model Prediction',loc='lower center')

# for p in ax0.patches:
#     ax0.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
    
# # 1. Model Prediction
# ax1 = sns.violinplot(data=train, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
#                      ax = axes[1], palette="Set3")
    
# axes[1].set_title(f'{col}, Training Set')

# axes[1].legend(title='Model Prediction',loc='lower center')
    
# add_stat_annotation(ax1, data=train, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
#                     box_pairs=[(("GB","TP & FP"),("GB","TN & FN")),
#                                (("LR","TP & FP"),("LR","TN & FN")),
#                                (("SVM-lin","TP & FP"),("SVM-lin","TN & FN")),
#                                (("SVM-rbf","TP & FP"),("SVM-rbf","TN & FN"))],
#                     loc='inside', verbose=2, line_height=0.1)

# # 2. Prediction TF
# ax2 = sns.violinplot(data=train, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Predict TF', hue_order=['TP & TN', 'FP & FN'],
#                      ax = axes[2], palette="Set1")
        
# axes[2].set_title(f'{col}, Training Set')

# axes[2].legend(title='Prediction TF',loc='lower center')
    
# add_stat_annotation(ax2, data=train, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Predict TF', hue_order=['TP & TN', 'FP & FN'],
#                     box_pairs=[(("GB","TP & TN"),("GB","FP & FN")),
#                                (("LR","TP & TN"),("LR","FP & FN")),
#                                (("SVM-lin","TP & TN"),("SVM-lin","FP & FN")),
#                                (("SVM-rbf","TP & TN"),("SVM-rbf","FP & FN"))],
#                     loc='inside', verbose=2, line_height=0.1)
    
# # 3. Holdout set
# ax3 = sns.countplot(data=test, x='Model',
#                     hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
#                     ax=axes[3], palette='Set2')
# axes[3].set_title(f'Holdout Set (n = {len(test["Session"].tolist())//4}) by MODEL')

# axes[3].legend(title='Model Prediction',loc='lower center')

# for p in ax3.patches:
#     ax3.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
    
# # 4. Model Prediction
# ax4 = sns.violinplot(data=test, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
#                      ax = axes[4], palette="Set3")
    
# axes[4].set_title(f'{col}, Holdout Set')

# axes[4].legend(title='Model Prediction',loc='lower center')
    
# add_stat_annotation(ax4, data=test, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
#                     box_pairs=[(("GB","TP & FP"),("GB","TN & FN")),
#                                (("LR","TP & FP"),("LR","TN & FN")),
#                                (("SVM-lin","TP & FP"),("SVM-lin","TN & FN")),
#                                (("SVM-rbf","TP & FP"),("SVM-rbf","TN & FN"))],
#                     loc='inside', verbose=2, line_height=0.1)

# # 5. Prediction TF
# ax5 = sns.violinplot(data=test, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Sex', hue_order=['Male', 'Female'],
#                      ax = axes[5], palette="Set1")
    
# axes[5].set_title(f'{col}, Holdout Set')

# axes[5].legend(title='Model Prediction',loc='lower center')
    
# add_stat_annotation(ax5, data=test, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Sex', hue_order=['Male', 'Female'],
#                     box_pairs=[(("GB","Male"),("GB","Female")),
#                                (("LR","Male"),("LR","Female")),
#                                (("SVM-lin","Male"),("SVM-lin","Female")),
#                                (("SVM-rbf","Male"),("SVM-rbf","Female"))],
#                     loc='inside', verbose=2, line_height=0.1)

#################################################################################
# fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(2*len(col), 6))
# # 0. Training set
# ax0 = sns.countplot(data=train, x='Model',
#                     hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
#                     ax=axes[0], palette='Set2')
# axes[0].set_title(f'Training Set (n = {len(train["Session"].tolist())//4}) by MODEL')
    
# axes[0].legend(title='Model Prediction',loc='lower center')

# for p in ax0.patches:
#     ax0.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
    
# # 1. Model Prediction P
# train_P = train[train['Model PN']=='TP & FP']
# ax1 = sns.violinplot(data=train_P, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Prob', hue_order=['TP', 'FP'],
#                      ax = axes[1], palette="Set3")
    
# axes[1].set_title(f'{col}, Training Set')

# axes[1].legend(title='Prediction Positive',loc='lower center')
    
# add_stat_annotation(ax1, data=train_P, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Prob', hue_order=['TP', 'FP'],
#                     box_pairs=[(("GB","TP"),("GB","FP")),
#                                (("LR","TP"),("LR","FP")),
#                                (("SVM-lin","TP"),("SVM-lin","FP")),
#                                (("SVM-rbf","TP"),("SVM-rbf","FP"))],
#                     loc='inside', verbose=2, line_height=0.1)

# # 2. Model Prediction N
# train_N = train[train['Model PN']=='TN & FN']
# ax2 = sns.violinplot(data=train_N, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Prob', hue_order=['TN', 'FN'],
#                      ax = axes[2], palette="Set1")
        
# axes[2].set_title(f'{col}, Training Set')

# axes[2].legend(title='Prediction Negative',loc='lower center')
    
# add_stat_annotation(ax2, data=train_N, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Prob', hue_order=['TN', 'FN'],
#                     box_pairs=[(("GB","TN"),("GB","FN")),
#                                (("LR","TN"),("LR","FN")),
#                                (("SVM-lin","TN"),("SVM-lin","FN")),
#                                (("SVM-rbf","TN"),("SVM-rbf","FN"))],
#                     loc='inside', verbose=2, line_height=0.1)
    
# # 3. Holdout set
# ax3 = sns.countplot(data=test, x='Model',
#                     hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
#                     ax=axes[3], palette='Set2')
# axes[3].set_title(f'Holdout Set (n = {len(test["Session"].tolist())//4}) by MODEL')

# axes[3].legend(title='Model Prediction',loc='lower center')

# for p in ax3.patches:
#     ax3.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
    
# # 4. Model Prediction P
# test_P = test[test['Model PN']=='TP & FP']
# ax4 = sns.violinplot(data=test_P, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Prob', hue_order=['TP', 'FP'],
#                      ax = axes[4], palette="Set3")
    
# axes[4].set_title(f'{col}, Holdout Set')

# axes[4].legend(title='Prediction Positive',loc='lower center')
    
# add_stat_annotation(ax4, data=test_P, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Prob', hue_order=['TP', 'FP'],
#                     box_pairs=[(("GB","TP"),("GB","FP")),
#                                (("LR","TP"),("LR","FP")),
#                                (("SVM-lin","TP"),("SVM-lin","FP")),
#                                (("SVM-rbf","TP"),("SVM-rbf","FP"))],
#                     loc='inside', verbose=2, line_height=0.1)

# # 5. Model Prediction N
# test_N = test[test['Model PN']=='TN & FN']
# ax5 = sns.violinplot(data=test_N, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Prob', hue_order=['TN', 'FN'],
#                      ax = axes[5], palette="Set1")
    
# axes[5].set_title(f'{col}, Holdout Set')

# axes[5].legend(title='Prediction Negative',loc='lower center')
    
# add_stat_annotation(ax5, data=test_N, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Prob', hue_order=['TN', 'FN'],
#                     box_pairs=[(("GB","TN"),("GB","FN")),
#                                (("LR","TN"),("LR","FN")),
#                                (("SVM-lin","TN"),("SVM-lin","FN")),
#                                (("SVM-rbf","TN"),("SVM-rbf","FN"))],
#                     loc='inside', verbose=2, line_height=0.1)

#################################################################################
# fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(2*len(col), 6))
# # 0. Training set
# ax0 = sns.countplot(data=train, x='Model',
#                     hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
#                     ax=axes[0], palette='Set2')
# axes[0].set_title(f'Training Set (n = {len(train["Session"].tolist())//4}) by MODEL')
    
# axes[0].legend(title='Model Prediction',loc='lower center')

# for p in ax0.patches:
#     ax0.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
    
# # 1. Prediction T
# train_T = train[train['Predict TF']=='TP & TN']
# ax1 = sns.violinplot(data=train_T, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Prob', hue_order=['TP', 'TN'],
#                      ax = axes[1], palette="Set3")
    
# axes[1].set_title(f'{col}, Training Set')

# axes[1].legend(title='Prediction True',loc='lower center')
    
# add_stat_annotation(ax1, data=train_T, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Prob', hue_order=['TP', 'TN'],
#                     box_pairs=[(("GB","TP"),("GB","TN")),
#                                (("LR","TP"),("LR","TN")),
#                                (("SVM-lin","TP"),("SVM-lin","TN")),
#                                (("SVM-rbf","TP"),("SVM-rbf","TN"))],
#                     loc='inside', verbose=2, line_height=0.1)

# # 2. Prediction F
# train_F = train[train['Predict TF']=='FP & FN']
# ax2 = sns.violinplot(data=train_F, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Prob', hue_order=['FP', 'FN'],
#                      ax = axes[2], palette="Set1")
        
# axes[2].set_title(f'{col}, Training Set')

# axes[2].legend(title='Prediction False',loc='lower center')
    
# add_stat_annotation(ax2, data=train_F, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Prob', hue_order=['FP', 'FN'],
#                     box_pairs=[(("GB","FP"),("GB","FN")),
#                                (("LR","FP"),("LR","FN")),
#                                (("SVM-lin","FP"),("SVM-lin","FN")),
#                                (("SVM-rbf","FP"),("SVM-rbf","FN"))],
#                     loc='inside', verbose=2, line_height=0.1)
    
# # 3. Holdout set
# ax3 = sns.countplot(data=test, x='Model',
#                     hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
#                     ax=axes[3], palette='Set2')
# axes[3].set_title(f'Holdout Set (n = {len(test["Session"].tolist())//4}) by MODEL')

# axes[3].legend(title='Model Prediction',loc='lower center')

# for p in ax3.patches:
#     ax3.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
    
# # 4. Prediction T
# test_T = test[test['Predict TF']=='TP & TN']
# ax4 = sns.violinplot(data=test_T, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Prob', hue_order=['TP', 'TN'],
#                      ax = axes[4], palette="Set3")
    
# axes[4].set_title(f'{col}, Holdout Set')

# axes[4].legend(title='Prediction True',loc='lower center')
    
# add_stat_annotation(ax4, data=test_T, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Prob', hue_order=['TP', 'TN'],
#                     box_pairs=[(("GB","TP"),("GB","TN")),
#                                (("LR","TP"),("LR","TN")),
#                                (("SVM-lin","TP"),("SVM-lin","TN")),
#                                (("SVM-rbf","TP"),("SVM-rbf","TN"))],
#                     loc='inside', verbose=2, line_height=0.1)

# # 5. Prediction F
# test_F = test[test['Predict TF']=='FP & FN']
# ax5 = sns.violinplot(data=test_F, x="Model", y=col,
#                      inner="quartile", split=True,
#                      hue='Prob', hue_order=['FP', 'FN'],
#                      ax = axes[5], palette="Set1")
    
# axes[5].set_title(f'{col}, Holdout Set')

# axes[5].legend(title='Prediction False',loc='lower center')
    
# add_stat_annotation(ax5, data=test_F, x='Model', y=col, 
#                     test='t-test_ind',
#                     hue='Prob', hue_order=['FP', 'FN'],
#                     box_pairs=[(("GB","FP"),("GB","FN")),
#                                (("LR","FP"),("LR","FN")),
#                                (("SVM-lin","FP"),("SVM-lin","FN")),
#                                (("SVM-rbf","FP"),("SVM-rbf","FN"))],
#                     loc='inside', verbose=2, line_height=0.1)