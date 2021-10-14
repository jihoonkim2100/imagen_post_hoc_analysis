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

def ml_Prob_plot(IN, train, test, col):
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(2*len(col), 16))
    
    data = train
    data_M = data[data['Sex']!='Female']
    data_F = data[data['Sex']=='Female']
    
    for i, data in enumerate([data, data_M, data_F]):
        # Model count
        ax = sns.countplot(data=data, x="Model", 
                           hue='Prob', #hue_order=['TP & FP', 'TN & FN'],
                           ax=axes[i,0], palette="Set2")
        if i == 0:
            axes[i,0].set_title(f'All: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
            
        elif i==1:
            axes[i,0].set_title(f'Male: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
        else:
            axes[i,0].set_title(f'Female: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
    
        ax.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        for p in ax.patches:
            ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()))
            
        axes[i,0].legend(title='Model',loc='lower left')

        # Model prediction = True
        data2 = data[data['Model PN']=='TP & FP']
        ax3 = sns.violinplot(data=data2, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,1], palette="Set3")
    
        ax3.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,1].set_title(f'{IN} - {col}, Validation - Positive')

        add_stat_annotation(ax3, data=data2, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("GB","HC"),("GB","AAM")),
                                       (("LR","HC"),("LR","AAM")),
                                       (("SVM-lin","HC"),("SVM-lin","AAM")),
                                       (("SVM-rbf","HC"),("SVM-rbf","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)
        
#         add_stat_annotation(ax3, data=data2, x='Model',
#                             y=col, test='t-test_ind',
#                             hue="Class", hue_order = ['HC', 'AAM'],
#                             box_pairs=[(("('X', 'Binge', 'cb', 'GB')","HC"),
#                                         ("('X', 'Binge', 'cb', 'GB')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'LR')","HC"),
#                                         ("('X', 'Binge', 'cb', 'LR')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'SVM-lin')","HC"),
#                                         ("('X', 'Binge', 'cb', 'SVM-lin')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'SVM-rbf')","HC"),
#                                         ("('X', 'Binge', 'cb', 'SVM-rbf')","AAM"))],
#                             loc='inside', verbose=2, line_height=0.1)
    
        # Model prediction = False    
        data3 = data[data['Model PN']=='TN & FN']
        ax2 = sns.violinplot(data=data3, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,2], palette="Set3")
    
        ax2.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,2].set_title(f'{IN} - {col}, Validation - Negative')
        add_stat_annotation(ax2, data=data2, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("GB","HC"),("GB","AAM")),
                                       (("LR","HC"),("LR","AAM")),
                                       (("SVM-lin","HC"),("SVM-lin","AAM")),
                                       (("SVM-rbf","HC"),("SVM-rbf","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)    
#         add_stat_annotation(ax2, data=data3, x='Model',
#                             y=col, test='t-test_ind',
#                             hue="Prob", hue_order = ['HC', 'AAM'],
#                             box_pairs=[(("('X', 'Binge', 'cb', 'GB')","HC"),
#                                         ("('X', 'Binge', 'cb', 'GB')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'LR')","HC"),
#                                         ("('X', 'Binge', 'cb', 'LR')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'SVM-lin')","HC"),
#                                         ("('X', 'Binge', 'cb', 'SVM-lin')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'SVM-rbf')","HC"),
#                                         ("('X', 'Binge', 'cb', 'SVM-rbf')","AAM"))],
#                             loc='inside', verbose=2, line_height=0.1)
    
    data = test
    data_M = data[data['Sex']!='Female']
    data_F = data[data['Sex']=='Female']
    
    for i, data in enumerate([data, data_M, data_F]):
        # Model count
        ax = sns.countplot(data=data, x="Model", 
                           hue='Prob', #hue_order=['TP & FP', 'TN & FN'],
                           ax=axes[i,3], palette="Set2")
        if i == 0:
            axes[i,3].set_title(f'All: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
            
        elif i==1:
            axes[i,3].set_title(f'Male: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
        else:
            axes[i,3].set_title(f'Female: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
    
        ax.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        for p in ax.patches:
            ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()))
            
        axes[i,3].legend(title='Model',loc='lower left')

        # Model prediction = True
        data2 = data[data['Model PN']=='TP & FP']
        ax3 = sns.violinplot(data=data2, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,4], palette="Set3")
    
        ax3.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,4].set_title(f'{IN} - {col}, Test - Positive')
        add_stat_annotation(ax3, data=data2, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("GB","HC"),("GB","AAM")),
                                       (("LR","HC"),("LR","AAM")),
                                       (("SVM-lin","HC"),("SVM-lin","AAM")),
                                       (("SVM-rbf","HC"),("SVM-rbf","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)    
#         add_stat_annotation(ax3, data=data2, x='Model',
#                             y=col, test='t-test_ind',
#                             hue="Class", hue_order = ['HC', 'AAM'],
#                             box_pairs=[(("('X-Binge', 'cb', 'GB')","HC"),
#                                         ("('X-Binge', 'cb', 'GB')","AAM")),
#                                        (("('X-Binge', 'cb', 'LR')","HC"),
#                                         ("('X-Binge', 'cb', 'LR')","AAM")),
#                                        (("('X-Binge', 'cb', 'SVM-lin')","HC"),
#                                         ("('X-Binge', 'cb', 'SVM-lin')","AAM")),
#                                        (("('X-Binge', 'cb', 'SVM-rbf')","HC"),
#                                         ("('X-Binge', 'cb', 'SVM-rbf')","AAM"))],
#                             loc='inside', verbose=2, line_height=0.1)
    
        # Model prediction = False    
        data3 = data[data['Model PN']=='TN & FN']
        ax2 = sns.violinplot(data=data3, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,5], palette="Set3")
    
        ax2.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,5].set_title(f'{IN} - {col}, Test - Negative')
        add_stat_annotation(ax2, data=data2, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("GB","HC"),("GB","AAM")),
                                       (("LR","HC"),("LR","AAM")),
                                       (("SVM-lin","HC"),("SVM-lin","AAM")),
                                       (("SVM-rbf","HC"),("SVM-rbf","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)    
#         add_stat_annotation(ax2, data=data3, x='Model',
#                             y=col, test='t-test_ind',
#                             hue="Class", hue_order = ['HC', 'AAM'],
#                             box_pairs=[(("('X-Binge', 'cb', 'GB')","HC"),
#                                         ("('X-Binge', 'cb', 'GB')","AAM")),
#                                        (("('X-Binge', 'cb', 'LR')","HC"),
#                                         ("('X-Binge', 'cb', 'LR')","AAM")),
#                                        (("('X-Binge', 'cb', 'SVM-lin')","HC"),
#                                         ("('X-Binge', 'cb', 'SVM-lin')","AAM")),
#                                        (("('X-Binge', 'cb', 'SVM-rbf')","HC"),
#                                         ("('X-Binge', 'cb', 'SVM-rbf')","AAM"))],
#                             loc='inside', verbose=2, line_height=0.1)        
    
    return [data.groupby(['Model PN','Model','Class'])[col].mean(),
            data.groupby(['Model PN','Model','Class','Sex'])[col].mean()]

class IMAGEN_descriptive:
    """ Plot the demographic statistics """
    def __init__(self, DF, COL):
        """ Set up the Dataframe and Columns
        
        Parameters
        ----------
        DF : pandas.dataframe
            The Instrument dataframe
            
        COL : string
            Instruments columns: ROI Columns, Sex, Site, Class
        
        """
        self.DF = DF
        self.Columns = list(COL[:-6])
        self.Target = list(COL[-1:])

    def histogram(self, bins=False, save=False):
        """ Plot the histogram
        
        Parameters
        ----------
        bins : Boolean, optional
            True : default (10), False : Sturge's Rule
        
        save : Boolean, optional
            If it is True then save as file (*.png)
        
        Notes
        -------
        Plot the subplot of the Histogram
        
        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO,
        >>>                                       col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.histogram(bins=True, save=False)

        """
        # Set the columns
        columns = self.Columns
        # Compute the bins based on Sturge's Rule
        b = 1 + (3.3*math.log(len(self.DF)))
        k = 10 if bins == False else math.ceil(b)
        # Plot the histogram
        self.DF[columns].hist(bins = k, figsize=(20, 14))
        
        if save == True:
            # PDF print version function needed
            pass
    
    def pairplot(self, save=False):
        """ Plot the pairplot

        Parameters
        ----------
        save : Boolean, optional
            If it is True then save as file (*.png)

        Notes
        -----
        Plot the subplot of the pairplot

        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO,
        >>>                                       col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.pairplot(save=False)
        
        """
        columns = self.Columns
        # Plot the pairplot
        sns.pairplot(data=self.DF, vars=columns, hue='Class',
                     plot_kws={'alpha': 0.2}, height=3,
                     diag_kind='kde', palette="Set1")
        
        if save == True:
            # PDF print version function needed
            pass
    
    def violinplot(self, save=False):
        """ Plot the violinplot

        Parameters
        ----------
        save : Boolean, optional
            If it is True then save as file (*.png)
        
        Notes
        -----
        Plot the subplot of the violinplot
        
        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO,
        >>>                                       col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.violinplot()
        
        """
        columns  = self.Columns       
        title = self.Columns
        
        # violin plot
        sns.set(style="whitegrid", font_scale=1.5)
        fig, axes = plt.subplots(nrows=1, ncols=len(columns)+1,
                                 figsize=((len(columns*2)+1)**2, len(columns*2)+1))

        sns.countplot(x="Class", hue='Sex', order=['HC', 'AAM'],
                      data = self.DF, ax = axes[0], palette="Set2")

        for i, j in enumerate(columns):
            axes[i+1].set_title(title[i])
            sns.violinplot(x="Class", y=j, data=self.DF, order=['HC', 'AAM'],
                           inner="quartile", ax = axes[i+1], palette="Set1")
            add_stat_annotation(ax = axes[i+1], data=self.DF, x="Class", y=j,
                                box_pairs = [("HC","AAM")], order=["HC","AAM"], test='t-test_ind',
                                text_format='star', loc='inside')
                
        # violin plot
        sns.set(style="whitegrid", font_scale=1.5)
        fig, axes = plt.subplots(nrows=1, ncols=len(columns)+1,
                                     figsize=((len(columns*2)+1)**2, len(columns*2)+1))

        sns.countplot(x="Class", hue='Sex', data = self.DF, order=['HC', 'AAM'],
                          ax = axes[0], palette="Set2")

        for i, j in enumerate(columns):
            axes[i+1].set_title(title[i])
            sns.violinplot(x="Class", y=j, hue='Sex', data=self.DF,
                           order=['HC', 'AAM'], inner="quartile",
                           ax = axes[i+1], split=True, palette="Set2")

    def catplot(self, save=False):
        """ Plot the catplot

        Parameters
        ----------
        save : Boolean, optional
            If it is True then save as file (*.png)
        
        Notes
        -----
        Plot the subplot of the catplot

        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO,
        >>>                                       col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.catplot(save=False)
        
        """
        columns = self.Columns
        
        # catplot
        sns.set(style="whitegrid", font_scale=1.5)

        for i, j in enumerate(columns):
            sns.catplot(x='Sex', y=j, hue = 'Class', col='Site', 
                        inner="quartile", data = self.DF, kind='violin',
                        split=True, height=4, aspect=.7, palette="Set2")
            
    def categorical_plot(self, save=False):
        """ Plot the barplot # change it lineplot
        
        Parameters
        ----------
        save : Boolean, optional
            If it is True then save as file (*.png)
        
        Notes
        -----
        Plot the subplot of the barplot        
        
        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO,
        >>>                                       col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.barplot(save=False)
        
        """
        columns = self.Columns
        # Table of nicotin dependence
        nd_class = pd.crosstab(index=self.DF["Class"], 
                               columns=self.DF["Nicotine dependence"],
                               margins=True)   # Include row and column totals

        nd_class.columns = ["highly dependent", "less dependent","moderately dependent", "coltotal"]
        nd_class.index= ["AAM","HC","rowtotal"]
        freq_nd_class = nd_class/nd_class.loc["rowtotal","coltotal"]

        print(f"{nd_class} \n \n {freq_nd_class} \n")
        
        ax = sns.countplot(y=columns[0], hue="Class", data=self.DF, palette="Set2")
        
        s = sns.catplot(y=columns[0], hue="Class", col="Sex", palette="Set2",
                        data=self.DF, kind="count", height=4, aspect=.7);
        
        c = sns.catplot(y=columns[0], hue="Class", col="Site", palette="Set2",
                        data=self.DF, kind="count", height=4, aspect=.7);
        
        
    def to_pdf(self):
        pass
    
    def demographic_plot(self, bins=False, save=False, viz=False):
        """ Plot the Summary Statistics
        
        Parameters
        ----------
        bins : Boolean, optional
            True : default (10), False : Sturge's Rule
        
        save : Boolean, optional
            If it is True then save as file (*.png)
        
        Notes
        -------
        Plot the Summary Statistics.
        Later divided into one Categorical way, the other numerical

        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO, col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.demographic_plot()

        """
        # Plot the demographic
        self.histogram(bins, save)
        self.pairplot(save)
        self.violinplot(save)
        self.catplot(save)
#         self.barplot(save)
        
        if viz == True:
            print(f"{'-'*83} \n{self.__str__()} \n{'-'*83}")
    
    def __str__(self):
        """ Print the Summary statistics """
        return 'Plot 1. histogram: '+ str(self.Columns) \
               +"\n"+'Plot 2. pariplot: '+ str(self.Columns) \
               +'\n'+'Plot 3. violinplot: '+ str(self.Columns) \
               +'\n'+'Plot 4. catplot: '+ str(self.Columns) \
               +'\n'+'Plot 5. barplot: '+ str(self.Columns)
    
class IMAGEN_inference(IMAGEN_descriptive):
    """ Compute the inference statistics """
    def __init__(self, DF, COL):
        """ Set up the Dataframe and Columns
        
        Parameters
        ----------
        DF : pandas.dataframe
            The Instrument dataframe
            
        COL : string
            Instruments columns: ROI Columns, Sex, Site, Class
        
        """
        self.DF = DF
        self.Columns = list(COL[:-6])
        self.Target = list(COL[-1:])
        
    def inference_statistics(self):
        for mean in self.Columns:
            print("-"*10, mean)
            myAAM = list(self.DF[self.DF['Class'] == 'AAM'][mean].values)
            AAM = [x for x in myAAM if pd.isnull(x) == False]
            myHC = list(self.DF[self.DF['Class'] == 'HC'][mean].values)    
            HC = [x for x in myHC if pd.isnull(x) == False]
            
            # Shapiro-Wilks
            normal1 = shapiro(AAM)
            normal2 = shapiro(HC)
            # Levene test
            normal3 = levene(AAM,HC)
            # bartlett test
            variance = bartlett(AAM, HC)
            # ttest
            ttest1 = ttest_ind(AAM, HC)
            ttest2 = ttest_ind(AAM, HC, equal_var=False)
            print(f'Shapiro-Wilks AAM: {normal1} \n'
                  f'Shapiro-Wilks HHC: {normal2} \n'
                  f'Levene test:       {normal3} \n'
                  f'Bartlett test:     {variance} \n'
                  f'T test:            {ttest1} \n'
                  f'T test:            {ttest2} \n')
            
    def ANOVA(self):
        pass

    def chi_squared(self):
        pass
    
    def __str__(self):
        """ Print the Inference statistics"""
        return 'Compute 1. normality check: '+ str(self.Columns) \
               +"\n"+'compute 2. t-test: '+ str(self.Columns)# \
               #+'\n'+'Compute 3. ANOVA: '+ str(self.Columns) \
               #+'\n'+'Compute 4. chi_squared: '+ str(self.Columns)
    
class IMAGEN_statistics(IMAGEN_inference):
    """ Summary of Descriptive, Inference Statistics """
    def __init__(self, DF, COL):
        """ Set up the Dataframe and Columns
        
        Parameters
        ----------
        DF : pandas.dataframe
            The Instrument dataframe
            
        COL : string
            Instruments columns: ROI Columns, Sex, Site, Class
        
        """
        self.DF = DF
        self.Columns = list(COL[:-6])
        self.Target = list(COL[-1:])
        
    def to_statistics(self):
        pass
    
def sc_plot(IN, data, col):
    fig, axes = plt.subplots(nrows=4, ncols=len(col)+1, figsize=(6*len(col), 7*4))
    # By class    
    ax = sns.countplot(data=data, x="Class", order=['HC', 'AAM'],
                       ax=axes[0,0], palette="Set1")
    
    axes[0,0].set_title(f'Session {data["Session"].values[0]}' +
                        f' (n = {len(data["Session"].tolist())}) by CLASS')

    for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()+2))

    for i, roi in enumerate(col):
        ax2 = sns.violinplot(data=data, x="Class", order=['HC', 'AAM'],
                             y=roi, inner="quartile", split=True,
                             ax = axes[0,i+1], palette="Set1")
#         ax2.set(ylim=(0, None))
        
        axes[0,i+1].set_title(f'{IN} - {" ".join(roi.split(" ")[:-1])}' +
                              f' (n = {str(len(data[roi].dropna()))})')
        
        add_stat_annotation(ax2, data=data, x='Class', order=['HC', 'AAM'],
                            y=roi, test='t-test_ind',
                            box_pairs=[(("HC"), ("AAM"))],
                            loc='inside', verbose=2, line_height=0.06)

    # By sex and class
    ax = sns.countplot(data=data, x="Sex", order=['Male', 'Female'],
                       hue='Class',hue_order=['HC', 'AAM'],
                       ax=axes[1,0], palette="Set2")
    
    axes[1,0].set_title(f'Session {data["Session"].values[0]}' +
                        f' (n = {len(data["Session"].tolist())}) by CLASS|SEX')
    
    for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+2))
        
    for i, roi in enumerate(col):
        ax2 = sns.violinplot(data=data, x="Sex", order=['Male', 'Female'],
                             y=roi, hue='Class', hue_order=['HC', 'AAM'],
                             inner="quartile", split=True,
                             ax = axes[1,i+1], palette="Set2")
        
#         ax2.set(ylim=(0, None))
        
        axes[1,i+1].set_title(f'{IN} - {" ".join(roi.split(" ")[:-1])}' +
                              f' (n = {str(len(data[roi].dropna()))})')
        
        add_stat_annotation(ax2, data=data, x='Sex', order=['Male', 'Female'],
                            y=roi, test='t-test_ind',
                            hue='Class', hue_order = ['HC', 'AAM'],
                            box_pairs=[(("Male","HC"), ("Male","AAM")),
                                       (("Female","HC"), ("Female","AAM")),
                                       (("Male","HC"),("Female","HC")),
                                       (("Male","AAM"),("Female","AAM"))],
                            loc='inside', verbose=2, line_height=0.06)

    # By class and sex
    ax = sns.countplot(data=data, x="Class", order=['HC', 'AAM'],
                       hue='Sex',hue_order=['Male', 'Female'],
                       ax=axes[2,0])
    
    axes[2,0].set_title(f'Session {data["Session"].values[0]}' +
                        f' (n = {len(data["Session"].tolist())}) by SEX|CLASS')
    
    for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+2))
    
    for i, roi in enumerate(col):
        ax2 = sns.violinplot(data=data, x="Class", order=['HC', 'AAM'],
                             y=roi, hue='Sex', hue_order=['Male', 'Female'],
                             inner="quartile", split=True,
                             ax = axes[2,i+1])
        
#         ax2.set(ylim=(0, None))
        
        axes[2,i+1].set_title(f'{IN} - {" ".join(roi.split(" ")[:-1])}' +
                              f' (n = {str(len(data[roi].dropna()))})')
        
        add_stat_annotation(ax2, data=data, x='Class', order=['HC', 'AAM'],
                            y=roi, test='t-test_ind',
                            hue='Sex', hue_order = ['Male', 'Female'],
                            box_pairs=[(("HC","Male"), ("AAM","Male")),
                                       (("HC","Female"), ("AAM","Female")),
                                       (("HC","Male"),("HC","Female")),
                                       (("AAM","Male"),("AAM","Female"))],
                            loc='inside', verbose=2, line_height=0.06)
        
    # By sex
    ax = sns.countplot(data=data, x="Sex", order=['Male', 'Female'],
                       ax=axes[3,0], palette="Paired")

    axes[3,0].set_title(f'Session {data["Session"].values[0]}' +
                        f' (n = {len(data["Session"].tolist())}) by SEX')

    for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()+2))

    for i, roi in enumerate(col):
        ax2 = sns.violinplot(data=data, x="Sex", order=['Male', 'Female'], y=roi,
                             inner="quartile", split=True,
                             ax = axes[3,i+1], palette="Paired")
        
#         ax2.set(ylim=(0, None))
        
        axes[3,i+1].set_title(f'{IN} - {" ".join(roi.split(" ")[:-1])}' +
                              f' (n = {str(len(data[roi].dropna()))})')
        
        add_stat_annotation(ax2, data=data, x='Sex', 
                            y=roi, test='t-test_ind',
                            box_pairs=[(("Male"), ("Female"))],
                            loc='inside', verbose=2, line_height=0.06)
        
    return [data.groupby(['Session','Class'])[col].mean(),
            data.groupby(['Session','Sex','Class'])[col].mean(),
            data.groupby(['Session','Class','Sex'])[col].mean(),
            data.groupby(['Session','Sex'])[col].mean()]

def pd_TF_plot(IN, train, test, col):
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(2*len(col), 16))
    
    data = train
    data_M = data[data['Sex']!='Female']
    data_F = data[data['Sex']=='Female']
    
    for i, data in enumerate([data, data_M, data_F]):
        # Model count
        ax = sns.countplot(data=data, x="Model", 
                           hue='Predict TF', hue_order=['TP & TN', 'FP & FN'],
                           ax=axes[i,0], palette="Set2")
        if i == 0:
            axes[i,0].set_title(f'All: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
            
        elif i==1:
            axes[i,0].set_title(f'Male: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
        else:
            axes[i,0].set_title(f'Female: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
    
        ax.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        for p in ax.patches:
            ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()))
            
        axes[i,0].legend(title='Prediction',loc='lower left')

        # Model prediction = True
        data2 = data[data['Predict TF']=='TP & TN']
        ax3 = sns.violinplot(data=data2, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,1], palette="Set3")
    
        ax3.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,1].set_title(f'{IN} - {col}, Validation - True')
    
        add_stat_annotation(ax3, data=data2, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("('X', 'Binge', 'cb', 'GB')","HC"),
                                        ("('X', 'Binge', 'cb', 'GB')","AAM")),
                                       (("('X', 'Binge', 'cb', 'LR')","HC"),
                                        ("('X', 'Binge', 'cb', 'LR')","AAM")),
                                       (("('X', 'Binge', 'cb', 'SVM-lin')","HC"),
                                        ("('X', 'Binge', 'cb', 'SVM-lin')","AAM")),
                                       (("('X', 'Binge', 'cb', 'SVM-rbf')","HC"),
                                        ("('X', 'Binge', 'cb', 'SVM-rbf')","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)
    
        # Model prediction = False    
        data3 = data[data['Predict TF']=='FP & FN']
        ax2 = sns.violinplot(data=data3, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,2], palette="Set3")
    
        ax2.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,2].set_title(f'{IN} - {col}, Validation - False')
    
        add_stat_annotation(ax2, data=data3, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("('X', 'Binge', 'cb', 'GB')","HC"),
                                        ("('X', 'Binge', 'cb', 'GB')","AAM")),
                                       (("('X', 'Binge', 'cb', 'LR')","HC"),
                                        ("('X', 'Binge', 'cb', 'LR')","AAM")),
                                       (("('X', 'Binge', 'cb', 'SVM-lin')","HC"),
                                        ("('X', 'Binge', 'cb', 'SVM-lin')","AAM")),
                                       (("('X', 'Binge', 'cb', 'SVM-rbf')","HC"),
                                        ("('X', 'Binge', 'cb', 'SVM-rbf')","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)
    
    data = test
    data_M = data[data['Sex']!='Female']
    data_F = data[data['Sex']=='Female']
    
    for i, data in enumerate([data, data_M, data_F]):
        # Model count
        ax = sns.countplot(data=data, x="Model", 
                           hue='Predict TF', hue_order=['TP & TN', 'FP & FN'],
                           ax=axes[i,3], palette="Set2")
        if i == 0:
            axes[i,3].set_title(f'All: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
            
        elif i==1:
            axes[i,3].set_title(f'Male: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
        else:
            axes[i,3].set_title(f'Female: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
    
        ax.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        for p in ax.patches:
            ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()))
            
        axes[i,3].legend(title='Prediction',loc='lower left')

        # Model prediction = True
        data2 = data[data['Predict TF']=='TP & TN']
        ax3 = sns.violinplot(data=data2, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,4], palette="Set3")
    
        ax3.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,4].set_title(f'{IN} - {col}, Test - True')
    
        add_stat_annotation(ax3, data=data2, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("('X-Binge', 'cb', 'GB')","HC"),
                                        ("('X-Binge', 'cb', 'GB')","AAM")),
                                       (("('X-Binge', 'cb', 'LR')","HC"),
                                        ("('X-Binge', 'cb', 'LR')","AAM")),
                                       (("('X-Binge', 'cb', 'SVM-lin')","HC"),
                                        ("('X-Binge', 'cb', 'SVM-lin')","AAM")),
                                       (("('X-Binge', 'cb', 'SVM-rbf')","HC"),
                                        ("('X-Binge', 'cb', 'SVM-rbf')","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)
    
        # Model prediction = False    
        data3 = data[data['Predict TF']=='FP & FN']
        ax2 = sns.violinplot(data=data3, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,5], palette="Set3")
    
        ax2.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,5].set_title(f'{IN} - {col}, Test - False')
    
        add_stat_annotation(ax2, data=data3, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("('X-Binge', 'cb', 'GB')","HC"),
                                        ("('X-Binge', 'cb', 'GB')","AAM")),
                                       (("('X-Binge', 'cb', 'LR')","HC"),
                                        ("('X-Binge', 'cb', 'LR')","AAM")),
                                       (("('X-Binge', 'cb', 'SVM-lin')","HC"),
                                        ("('X-Binge', 'cb', 'SVM-lin')","AAM")),
                                       (("('X-Binge', 'cb', 'SVM-rbf')","HC"),
                                        ("('X-Binge', 'cb', 'SVM-rbf')","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)        
    
    return [data.groupby(['Predict TF','Model','Class'])[col].mean(),
            data.groupby(['Predict TF','Model','Class','Sex'])[col].mean()]

def ml_PN_plot(IN, train, test, col):
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(2*len(col), 16))
    
    data = train
    data_M = data[data['Sex']!='Female']
    data_F = data[data['Sex']=='Female']
    
    for i, data in enumerate([data, data_M, data_F]):
        # Model count
        ax = sns.countplot(data=data, x="Model", 
                           hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                           ax=axes[i,0], palette="Set2")
        if i == 0:
            axes[i,0].set_title(f'All: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
            
        elif i==1:
            axes[i,0].set_title(f'Male: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
        else:
            axes[i,0].set_title(f'Female: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
    
        ax.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        for p in ax.patches:
            ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()))
            
        axes[i,0].legend(title='Model',loc='lower left')

        # Model prediction = True
        data2 = data[data['Model PN']=='TP & FP']
        ax3 = sns.violinplot(data=data2, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,1], palette="Set3")
    
        ax3.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,1].set_title(f'{IN} - {col}, Validation - Positive')
    
        add_stat_annotation(ax3, data=data2, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("('X', 'Binge', 'cb', 'GB')","HC"),
                                        ("('X', 'Binge', 'cb', 'GB')","AAM")),
                                       (("('X', 'Binge', 'cb', 'LR')","HC"),
                                        ("('X', 'Binge', 'cb', 'LR')","AAM")),
                                       (("('X', 'Binge', 'cb', 'SVM-lin')","HC"),
                                        ("('X', 'Binge', 'cb', 'SVM-lin')","AAM")),
                                       (("('X', 'Binge', 'cb', 'SVM-rbf')","HC"),
                                        ("('X', 'Binge', 'cb', 'SVM-rbf')","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)
    
        # Model prediction = False    
        data3 = data[data['Model PN']=='TN & FN']
        ax2 = sns.violinplot(data=data3, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,2], palette="Set3")
    
        ax2.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,2].set_title(f'{IN} - {col}, Validation - Negative')
    
        add_stat_annotation(ax2, data=data3, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("('X', 'Binge', 'cb', 'GB')","HC"),
                                        ("('X', 'Binge', 'cb', 'GB')","AAM")),
                                       (("('X', 'Binge', 'cb', 'LR')","HC"),
                                        ("('X', 'Binge', 'cb', 'LR')","AAM")),
                                       (("('X', 'Binge', 'cb', 'SVM-lin')","HC"),
                                        ("('X', 'Binge', 'cb', 'SVM-lin')","AAM")),
                                       (("('X', 'Binge', 'cb', 'SVM-rbf')","HC"),
                                        ("('X', 'Binge', 'cb', 'SVM-rbf')","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)
    
    data = test
    data_M = data[data['Sex']!='Female']
    data_F = data[data['Sex']=='Female']
    
    for i, data in enumerate([data, data_M, data_F]):
        # Model count
        ax = sns.countplot(data=data, x="Model", 
                           hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                           ax=axes[i,3], palette="Set2")
        if i == 0:
            axes[i,3].set_title(f'All: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
            
        elif i==1:
            axes[i,3].set_title(f'Male: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
        else:
            axes[i,3].set_title(f'Female: Session {data["Session"].values[0]}' +
                                f' (n = {len(data["Session"].tolist())//4}) by MODEL')
    
        ax.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        for p in ax.patches:
            ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()))
            
        axes[i,3].legend(title='Model',loc='lower left')

        # Model prediction = True
        data2 = data[data['Model PN']=='TP & FP']
        ax3 = sns.violinplot(data=data2, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,4], palette="Set3")
    
        ax3.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,4].set_title(f'{IN} - {col}, Test - Positive')
    
        add_stat_annotation(ax3, data=data2, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("('X-Binge', 'cb', 'GB')","HC"),
                                        ("('X-Binge', 'cb', 'GB')","AAM")),
                                       (("('X-Binge', 'cb', 'LR')","HC"),
                                        ("('X-Binge', 'cb', 'LR')","AAM")),
                                       (("('X-Binge', 'cb', 'SVM-lin')","HC"),
                                        ("('X-Binge', 'cb', 'SVM-lin')","AAM")),
                                       (("('X-Binge', 'cb', 'SVM-rbf')","HC"),
                                        ("('X-Binge', 'cb', 'SVM-rbf')","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)
    
        # Model prediction = False    
        data3 = data[data['Model PN']=='TN & FN']
        ax2 = sns.violinplot(data=data3, x="Model",
                             y=col, hue="Class", hue_order=['HC','AAM'],
                             inner="quartile", split=True,
                             ax = axes[i,5], palette="Set3")
    
        ax2.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
        axes[i,5].set_title(f'{IN} - {col}, Test - Negative')
    
        add_stat_annotation(ax2, data=data3, x='Model',
                            y=col, test='t-test_ind',
                            hue="Class", hue_order = ['HC', 'AAM'],
                            box_pairs=[(("('X-Binge', 'cb', 'GB')","HC"),
                                        ("('X-Binge', 'cb', 'GB')","AAM")),
                                       (("('X-Binge', 'cb', 'LR')","HC"),
                                        ("('X-Binge', 'cb', 'LR')","AAM")),
                                       (("('X-Binge', 'cb', 'SVM-lin')","HC"),
                                        ("('X-Binge', 'cb', 'SVM-lin')","AAM")),
                                       (("('X-Binge', 'cb', 'SVM-rbf')","HC"),
                                        ("('X-Binge', 'cb', 'SVM-rbf')","AAM"))],
                            loc='inside', verbose=2, line_height=0.1)        
    
    return [data.groupby(['Model PN','Model','Class'])[col].mean(),
            data.groupby(['Model PN','Model','Class','Sex'])[col].mean()]

# def ml_plot(IN, data, col):
#     fig, axes = plt.subplots(nrows=3, ncols=2+1, figsize=(2*len(col), 16))
    
#     data_M = data[data['Sex']!='Female']
#     data_F = data[data['Sex']=='Female']
    
#     for i, data in enumerate([data, data_M, data_F]):
#         # Model count
#         ax = sns.countplot(data=data, x="Model", 
#                            hue='Predict', hue_order=[True,False],
#                            ax=axes[i,0], palette="Set2")
#         if i == 0:
#             axes[i,0].set_title(f'All: Session {data["Session"].values[0]}' +
#                                 f' (n = {len(data["Session"].tolist())//4}) by MODEL')
            
#         elif i==1:
#             axes[i,0].set_title(f'Male: Session {data["Session"].values[0]}' +
#                                 f' (n = {len(data["Session"].tolist())//4}) by MODEL')
#         else:
#             axes[i,0].set_title(f'Female: Session {data["Session"].values[0]}' +
#                                 f' (n = {len(data["Session"].tolist())//4}) by MODEL')
    
#         ax.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
#         for p in ax.patches:
#             ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()))
            
#         axes[i,0].legend(title='Prediction',loc='lower left')

#         # Model prediction = True
#         data2 = data[data['Predict']==True]
#         ax3 = sns.violinplot(data=data2, x="Model",
#                              y=col, hue="Class", hue_order=['HC','AAM'],
#                              inner="quartile", split=True,
#                              ax = axes[i,1], palette="Set3")
    
#         ax3.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
#         axes[i,1].set_title(f'{IN} - {col}, Validation - True')
    
#         add_stat_annotation(ax3, data=data2, x='Model',
#                             y=col, test='t-test_ind',
#                             hue="Class", hue_order = ['HC', 'AAM'],
#                             box_pairs=[(("('X', 'Binge', 'cb', 'GB')","HC"),
#                                         ("('X', 'Binge', 'cb', 'GB')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'LR')","HC"),
#                                         ("('X', 'Binge', 'cb', 'LR')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'SVM-lin')","HC"),
#                                         ("('X', 'Binge', 'cb', 'SVM-lin')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'SVM-rbf')","HC"),
#                                         ("('X', 'Binge', 'cb', 'SVM-rbf')","AAM"))],
#                             loc='inside', verbose=2, line_height=0.1)
    
#         # Model prediction = False    
#         data3 = data[data['Predict']==False]
#         ax2 = sns.violinplot(data=data3, x="Model",
#                              y=col, hue="Class", hue_order=['HC','AAM'],
#                              inner="quartile", split=True,
#                              ax = axes[i,2], palette="Set3")
    
#         ax2.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
#         axes[i,2].set_title(f'{IN} - {col}, Validation - False')
    
#         add_stat_annotation(ax2, data=data3, x='Model',
#                             y=col, test='t-test_ind',
#                             hue="Class", hue_order = ['HC', 'AAM'],
#                             box_pairs=[(("('X', 'Binge', 'cb', 'GB')","HC"),
#                                         ("('X', 'Binge', 'cb', 'GB')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'LR')","HC"),
#                                         ("('X', 'Binge', 'cb', 'LR')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'SVM-lin')","HC"),
#                                         ("('X', 'Binge', 'cb', 'SVM-lin')","AAM")),
#                                        (("('X', 'Binge', 'cb', 'SVM-rbf')","HC"),
#                                         ("('X', 'Binge', 'cb', 'SVM-rbf')","AAM"))],
#                             loc='inside', verbose=2, line_height=0.1)

# def test_plot(IN, data, col):
#     fig, axes = plt.subplots(nrows=3, ncols=2+1, figsize=(2*len(col), 16))
    
#     data_M = data[data['Sex']!='Female']
#     data_F = data[data['Sex']=='Female']
    
#     for i, data in enumerate([data, data_M, data_F]):
#         # Model count
#         ax = sns.countplot(data=data, x="Model", 
#                            hue='Predict', hue_order=[True,False],
#                            ax=axes[i,0], palette="Set2")
#         if i == 0:
#             axes[i,0].set_title(f'All: Session {data["Session"].values[0]}' +
#                                 f' (n = {len(data["Session"].tolist())//4}) by MODEL')
            
#         elif i==1:
#             axes[i,0].set_title(f'Male: Session {data["Session"].values[0]}' +
#                                 f' (n = {len(data["Session"].tolist())//4}) by MODEL')
#         else:
#             axes[i,0].set_title(f'Female: Session {data["Session"].values[0]}' +
#                                 f' (n = {len(data["Session"].tolist())//4}) by MODEL')
    
#         ax.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
#         for p in ax.patches:
#             ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()))
            
#         axes[i,0].legend(title='Prediction',loc='lower left')

#         # Model prediction = True
#         data2 = data[data['Predict']==True]
#         ax3 = sns.violinplot(data=data2, x="Model",
#                              y=col, hue="Class", hue_order=['HC','AAM'],
#                              inner="quartile", split=True,
#                              ax = axes[i,1], palette="Set3")
    
#         ax3.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
#         axes[i,1].set_title(f'{IN} - {col}, Test - True')
    
#         add_stat_annotation(ax3, data=data2, x='Model',
#                             y=col, test='t-test_ind',
#                             hue="Class", hue_order = ['HC', 'AAM'],
#                             box_pairs=[(("('X-Binge', 'cb', 'GB')","HC"),
#                                         ("('X-Binge', 'cb', 'GB')","AAM")),
#                                        (("('X-Binge', 'cb', 'LR')","HC"),
#                                         ("('X-Binge', 'cb', 'LR')","AAM")),
#                                        (("('X-Binge', 'cb', 'SVM-lin')","HC"),
#                                         ("('X-Binge', 'cb', 'SVM-lin')","AAM")),
#                                        (("('X-Binge', 'cb', 'SVM-rbf')","HC"),
#                                         ("('X-Binge', 'cb', 'SVM-rbf')","AAM"))],
#                             loc='inside', verbose=2, line_height=0.1)
    
#         # Model prediction = False    
#         data3 = data[data['Predict']==False]
#         ax2 = sns.violinplot(data=data3, x="Model",
#                              y=col, hue="Class", hue_order=['HC','AAM'],
#                              inner="quartile", split=True,
#                              ax = axes[i,2], palette="Set3")
    
#         ax2.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
#         axes[i,2].set_title(f'{IN} - {col}, Test - False')
    
#         add_stat_annotation(ax2, data=data3, x='Model',
#                             y=col, test='t-test_ind',
#                             hue="Class", hue_order = ['HC', 'AAM'],
#                             box_pairs=[(("('X-Binge', 'cb', 'GB')","HC"),
#                                         ("('X-Binge', 'cb', 'GB')","AAM")),
#                                        (("('X-Binge', 'cb', 'LR')","HC"),
#                                         ("('X-Binge', 'cb', 'LR')","AAM")),
#                                        (("('X-Binge', 'cb', 'SVM-lin')","HC"),
#                                         ("('X-Binge', 'cb', 'SVM-lin')","AAM")),
#                                        (("('X-Binge', 'cb', 'SVM-rbf')","HC"),
#                                         ("('X-Binge', 'cb', 'SVM-rbf')","AAM"))],
#                             loc='inside', verbose=2, line_height=0.1)


#     # Model count
#     ax = sns.countplot(data=data, x="Model", 
#                        hue='Predict',
#                        ax=axes[0,0], palette="Set2")
    
#     axes[0,0].set_title(f'Session {data["Session"].values[0]}' +
#                       f' (n = {len(data["Session"].tolist())//4}) by MODEL')
    
#     ax.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
#     for p in ax.patches:
#         ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()))
    
#     # Model prediction = False    
#     data2 = data[data['Predict']==False]
#     ax2 = sns.violinplot(data=data2, x="Model",
#                          y=col, hue="Class", hue_order=['HC','AAM'],
#                          inner="quartile", split=True,
#                          ax = axes[0,1], palette="Set3")
    
#     ax2.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
#     axes[0,1].set_title(f'{IN} - {col}, Model Predict - False')
    
#     add_stat_annotation(ax2, data=data2, x='Model',
#                         y=col, test='t-test_ind',
#                         hue="Class", hue_order = ['HC', 'AAM'],
#                         box_pairs=[(("('X', 'Binge', 'cb', 'GB')","HC"),
#                                     ("('X', 'Binge', 'cb', 'GB')","AAM")),
#                                    (("('X', 'Binge', 'cb', 'LR')","HC"),
#                                     ("('X', 'Binge', 'cb', 'LR')","AAM")),
#                                    (("('X', 'Binge', 'cb', 'SVM-lin')","HC"),
#                                     ("('X', 'Binge', 'cb', 'SVM-lin')","AAM")),
#                                    (("('X', 'Binge', 'cb', 'SVM-rbf')","HC"),
#                                     ("('X', 'Binge', 'cb', 'SVM-rbf')","AAM"))],
#                         loc='inside', verbose=2, line_height=0.1)
    
#     # Model prediction = True
#     data3 = data[data['Predict']==True]
#     ax3 = sns.violinplot(data=data3, x="Model",
#                          y=col, hue="Class", hue_order=['HC','AAM'],
#                          inner="quartile", split=True,
#                          ax = axes[0,2], palette="Set3")
    
#     ax3.set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
    
#     axes[0,2].set_title(f'{IN} - {col}, Model Predict - True')
    
#     add_stat_annotation(ax3, data=data3, x='Model',
#                         y=col, test='t-test_ind',
#                         hue="Class", hue_order = ['HC', 'AAM'],
#                         box_pairs=[(("('X', 'Binge', 'cb', 'GB')","HC"),
#                                     ("('X', 'Binge', 'cb', 'GB')","AAM")),
#                                    (("('X', 'Binge', 'cb', 'LR')","HC"),
#                                     ("('X', 'Binge', 'cb', 'LR')","AAM")),
#                                    (("('X', 'Binge', 'cb', 'SVM-lin')","HC"),
#                                     ("('X', 'Binge', 'cb', 'SVM-lin')","AAM")),
#                                    (("('X', 'Binge', 'cb', 'SVM-rbf')","HC"),
#                                     ("('X', 'Binge', 'cb', 'SVM-rbf')","AAM"))],
#                         loc='inside', verbose=2, line_height=0.1)


def tf_plot(data, roi):
    ax = sns.catplot(data=data, x="Model PN", y=roi, col="Model",
                     kind="violin", hue="Class", hue_order = ['HC', 'AAM'],
                     inner="quartile", split=True, palette="Set2",
                     height=6, aspect=1); 
    
    g = sns.catplot(data=data, x='Model', y=roi, col="Model PN",
                    kind="violin", hue="Class", hue_order = ['HC', 'AAM'],
                    inner="quartile", split=True, palette="Set2",
                    height=6, aspect=1);
    
    (g.set_axis_labels("", roi)
     .set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
     .set_titles("{col_name} {col_var}")
     .despine(left=True))

# def violin_plot(DATA, ROI):
#     for col in ROI:
#         sns.set(style="whitegrid", font_scale=1)
#         fig, axes = plt.subplots(nrows=1, ncols=len(DATA),
#                                  figsize = ((len(DATA)+1)**2, len(DATA)+1))
#         fig.suptitle(f'{col}', fontsize=15)
#         for i, (Key, DF) in enumerate(DATA):
#             axes[i].set_title(f'{Key} = {str(len(DF[col].dropna()))}')
#             sns.violinplot(x="Class", y=col, data = DF, order=['HC', 'AAM'],
#                            inner="quartile", ax = axes[i], palette="Set2")
#             add_stat_annotation(ax = axes[i], data=DF, x="Class", y=col,
#                                 box_pairs = [("HC","AAM")], order=["HC","AAM"],
#                                 test='t-test_ind', text_format='star', loc='inside')
    
# ax = sns.catplot(data=ML_CTQ, x="Predict", y='Denial sum', col="Model",
#                  kind="violin", hue="Class", hue_order = ['HC', 'AAM'],
#                  inner="quartile", split=True, palette="Set2",
#                  height=6, aspect=1); 

# g = sns.catplot(data=ML_CTQ, x='Model', y='Denial sum', col="Predict",
#                 kind="violin", hue="Class", hue_order = ['HC', 'AAM'],
#                 inner="quartile", split=True, palette="Set2",
#                 height=6, aspect=1);

# (g.set_axis_labels("", "Denial sum")
#   .set_xticklabels(["GB", "LR", "SVM-lin", "SVM-rbf"])
#   .set_titles("{col_name} {col_var}")
#   .despine(left=True))
    
# for (S, DF2) in [('All', binge_NEO),
#                 ('True GB', binge_NEO.set_index('ID').loc[GB_T, :]),
#                 ('False GB', binge_NEO.set_index('ID').loc[GB_F, :]),
#                 ('True LR', binge_NEO.set_index('ID').loc[LR_T, :]),
#                 ('False LR', binge_NEO.set_index('ID').loc[LR_F, :]),
#                 ('True SVM lin', binge_NEO.set_index('ID').loc[SVM_lin_T, :]),
#                 ('False SVM lin', binge_NEO.set_index('ID').loc[SVM_lin_F, :]),
#                 ('True SVM rbf', binge_NEO.set_index('ID').loc[SVM_rbf_T, :]),
#                 ('False SVM rbf', binge_NEO.set_index('ID').loc[SVM_rbf_F, :])]:
#     columns = DF2.columns[:5]
    
#     sns.set(style="whitegrid", font_scale=1.5)
#     fig, axes = plt.subplots(nrows=1, ncols=len(columns)+1,
#                              figsize=((len(columns)+1)**2, len(columns)+1))
#     sns.countplot(x="Class", hue='Sex', order=['HC', 'AAM'], data = DF2,
#                   ax = axes[0], palette="Set2").set(title=S)
    
#     for i, j in enumerate(columns):
#         axes[i+1].set_title(columns[i])
#         sns.violinplot(x="Class", y=j, data=DF2, order=['HC', 'AAM'],
#                        inner="quartile", ax = axes[i+1], palette="Set1")
#         add_stat_annotation(ax = axes[i+1], data=DF2, x="Class", y=j,
#                             box_pairs = [("HC","AAM")], order=["HC","AAM"],
#                             test='t-test_ind', text_format='star', loc='inside')         
        
# fig, axes = plt.subplots(nrows=1, ncols=len(col)+1, figsize=(6*len(col), 6))

# ax = sns.countplot(data=data, x="Sex", order=['Male', 'Female'],
#                    ax=axes[0], palette="Set2")

# axes[0].set_title(f'Session {data["Session"].values[0]}' +
#                   f' (n = {len(data["Session"].tolist())})')

# for p in ax.patches:
#     ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()+2))

# for i, roi in enumerate(col):
#     ax2 = sns.violinplot(data=data, x="Sex", order=['Male', 'Female'],
#                          y=roi,
#                          inner="quartile", split=True,
#                          ax = axes[i+1], palette="Set2")
#     axes[i+1].set_title(f'NEO - {roi[:-5]} (n = {str(len(data[roi].dropna()))})')

#     add_stat_annotation(ax2, data=data, x='Sex', 
#                     y=roi, test='t-test_ind',
#                     box_pairs=[(("Male"), ("Female"))],
#                     loc='inside', verbose=2, line_height=0.06)

# fig, axes = plt.subplots(nrows=1, ncols=len(col)+1, figsize=(6*len(col), 6))

# ax = sns.countplot(data=data, x="Sex", order=['Male', 'Female'],
#                    hue='Class',hue_order=['HC', 'AAM'],
#                    ax=axes[0], palette="Set2")

# axes[0].set_title(f'Session {data["Session"].values[0]}' +
#                   f' (n = {len(data["Session"].tolist())})')

# for p in ax.patches:
#     ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.16, p.get_height()+2))

# for i, roi in enumerate(col):
#     ax2 = sns.violinplot(data=data, x="Sex", order=['Male', 'Female'],
#                          y=roi, hue='Class', hue_order=['HC', 'AAM'],
#                          inner="quartile", split=True,
#                          ax = axes[i+1], palette="Set2")
#     axes[i+1].set_title(f'NEO - {roi[:-5]} (n = {str(len(data[roi].dropna()))})')
    
#     add_stat_annotation(ax2, data=data, x='Sex', order=['Male', 'Female'],
#                     y=roi, test='t-test_ind',
#                     hue='Class', hue_order = ['HC', 'AAM'],
#                     box_pairs=[(("Male","HC"), ("Male","AAM")),
#                                (("Female","HC"), ("Female","AAM")),
#                                (("Male","HC"),("Female","HC")),
#                                (("Male","AAM"),("Female","AAM"))],
#                     loc='inside', verbose=2, line_height=0.06)

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16*1, 9*1))

# ax = sns.countplot(data=b_NEO, x="Sex", order=['Male', 'Female'],
#                    hue='Class',hue_order=['HC', 'AAM'],
#                    ax=axes[0], palette="Set2")

# ax2 = sns.violinplot(data=b_NEO, x="Sex", order=['Male', 'Female'],
#                      y='Extroversion mean', hue='Class', hue_order=['HC', 'AAM'],
#                      inner="quartile", split=True, ax = axes[1], palette="Set2")

# for p in ax.patches:
#     ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.16, p.get_height()+2))
    
# add_stat_annotation(ax2, data=b_NEO, x='Sex', order=['Male', 'Female'],
#                     y='Extroversion mean', test='t-test_ind',
#                     hue='Class', hue_order = ['HC', 'AAM'],
#                     box_pairs=[(("Male","HC"), ("Male","AAM")),
#                                (("Female","HC"), ("Female","AAM")),
#                                (("Male","HC"),("Female","HC")),
#                                (("Male","AAM"),("Female","AAM"))],
#                     loc='outside', verbose=2, line_height=0.01)

# add_stat_annotation(ax2, data=b_NEO, x='Sex', 
#                     y='Extroversion mean', test='t-test_ind',
#                     box_pairs=[(("Male"), ("Female"))],
#                     loc='inside', verbose=2, line_height=0.06)

# plt.legend(loc='lower center')

# data = b_NEO
# col = c_NEO

# fig, axes = plt.subplots(nrows=2, ncols=len(col)+1, figsize=(6*len(col), 7*2))

# ax = sns.countplot(data=data, x="Sex", order=['Male', 'Female'],
#                    ax=axes[0,0], palette="Set2")

# axes[0,0].set_title(f'Session {data["Session"].values[0]}' +
#                   f' (n = {len(data["Session"].tolist())})')

# for p in ax.patches:
#     ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()+2))

# for i, roi in enumerate(col):
#     ax2 = sns.violinplot(data=data, x="Sex", order=['Male', 'Female'],
#                          y=roi,
#                          inner="quartile", split=True,
#                          ax = axes[0,i+1], palette="Set2")
#     axes[0,i+1].set_title(f'NEO - {roi[:-5]} (n = {str(len(data[roi].dropna()))})')

#     add_stat_annotation(ax2, data=data, x='Sex', 
#                     y=roi, test='t-test_ind',
#                     box_pairs=[(("Male"), ("Female"))],
#                     loc='inside', verbose=2, line_height=0.06)

# ax = sns.countplot(data=data, x="Sex", order=['Male', 'Female'],
#                    hue='Class',hue_order=['HC', 'AAM'],
#                    ax=axes[1,0], palette="Set2")

# axes[1,0].set_title(f'Session {data["Session"].values[0]}' +
#                   f' (n = {len(data["Session"].tolist())})')

# for p in ax.patches:
#     ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.16, p.get_height()+2))

# for i, roi in enumerate(col):
#     ax2 = sns.violinplot(data=data, x="Sex", order=['Male', 'Female'],
#                          y=roi, hue='Class', hue_order=['HC', 'AAM'],
#                          inner="quartile", split=True,
#                          ax = axes[1,i+1], palette="Set2")
#     axes[1,i+1].set_title(f'NEO - {roi[:-5]} (n = {str(len(data[roi].dropna()))})')
    
#     add_stat_annotation(ax2, data=data, x='Sex', order=['Male', 'Female'],
#                     y=roi, test='t-test_ind',
#                     hue='Class', hue_order = ['HC', 'AAM'],
#                     box_pairs=[(("Male","HC"), ("Male","AAM")),
#                                (("Female","HC"), ("Female","AAM")),
#                                (("Male","HC"),("Female","HC")),
#                                (("Male","AAM"),("Female","AAM"))],
#                     loc='inside', verbose=2, line_height=0.06)