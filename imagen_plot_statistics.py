#################################################################################
## Last-modified :  11th August 2021
## Author : JiHoon Kim
##
## USAGE EXAMPLE
## from imagen_plot_statistics import *
## plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO, binge_FU3_NEO.DATA)
## plot_binge_FU3_NEO.histogram()
## plot_binge_FU3_NEO.histogram(bins = False)
## plot_binge_FU3_NEO.pairplot()
## plot_binge_FU3_NEO.violinplot()
## plot_binge_FU3_NEO.catplot()
## 
## # Recommend Version
## from imagen_plot_statistics import *
## plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO, binge_FU3_NEO.DATA)
## _ = plot_binge_FU3_NEO.demographic_plot()
## print(plot_binge_FU3_NEO)
#################################################################################

import math
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

class plot_demographic:
    """
    Plot the demographic statistics of the data
    # Depends on other Session, it can add the Session and DATA from here.
    # def load_df
    """
    def __init__(self, DF, COL):
        self.DF = DF
        self.Columns = list(COL[:-3])
        self.Target = list(COL[-1:])

    def histogram(self, bins = 10):
        """
        Plot the histogram
        
        Argument:
                columns = variables
                target = class : AAM. HC
                title = average
                bins = {True : default, False : Sturge's Rule}
        
        Output:
                Histogram
        """
        # this colunms, target, data, title will move in load_DF later
        columns = self.Columns
        
        k = bins
        if not bins:
            b = 1 + (3.3*math.log(len(self.DF)))
            k = math.ceil(b)
        
        # PDF print version function needed
        return self.DF[columns].hist(bins = k, figsize=(20, 14))
    
    def pairplot(self):
        """
        Plot the pairplot
        
        Argument:
                columns = variables
                class = AAM, HC
        
        Output:
                Pairplot
        """
        columns = self.Columns
        
        # PDF print version function needed
        return sns.pairplot(data=self.DF, vars=columns, palette="Set1",
                            hue='Class', plot_kws={'alpha': 0.2}, 
                            height=3, diag_kind='kde')
    
    def violinplot(self):
        """
        Plot the violinplot
        
        Argument:
                columns = variables
                class = AAM, HC
        
        Output:
                Violinplot
        """
        columns  = self.Columns       
        title = self.Columns
        
        # violin plot
        sns.set(style="whitegrid", font_scale=1.5)
        fig, axes = plt.subplots(nrows=1, ncols=len(columns)+1,
                                     figsize=((len(columns)+1)**2, len(columns)+1))

        sns.countplot(x="Class", hue='Sex', order=['HC', 'AAM'],
                          data = self.DF, ax = axes[0], palette="Set2")

        for i, j in enumerate(columns):
            axes[i+1].set_title(title[i])
            sns.violinplot(x="Class", y=j, data=self.DF, order=['HC', 'AAM'],
                               inner="quartile", ax = axes[i+1], palette="Set1")
                
        # violin plot
        sns.set(style="whitegrid", font_scale=1.5)
        fig, axes = plt.subplots(nrows=1, ncols=len(columns)+1,
                                     figsize=((len(columns)+1)**2, len(columns)+1))

        sns.countplot(x="Class", hue='Sex', data = self.DF, order=['HC', 'AAM'],
                          ax = axes[0], palette="Set2")

        for i, j in enumerate(columns):
            axes[i+1].set_title(title[i])
            sns.violinplot(x="Class", y=j, hue='Sex', data=self.DF,
                               order=['HC', 'AAM'], inner="quartile",
                               ax = axes[i+1], split=True, palette="Set2")

    def catplot(self):
        """
        Plot the catplot
        
        Argument:
                columns = variables
                class = AAM, HC
        
        Output:
                Violinplot
        """
        columns = self.Columns
        
        # catplot
        sns.set(style="whitegrid", font_scale=1.5)

        for i, j in enumerate(columns):
            sns.catplot(x='Sex', y=j, hue = 'Class', col='Site', 
                            inner="quartile", data = self.DF, 
                            kind='violin', split=True,
                            height=4, aspect=.7, palette="Set2")
        
    def pdf_print(self):
        pass
    
    def demographic_plot(self, bins = 10):
        """
        Summary statistics
        
        """
        # later divided into one Categorical way, the other numerical
        self.histogram(bins)
        self.pairplot()
        self.violinplot()
        self.catplot()
    
    def __str__(self):
        return 'Plot 1. histogram: ' \
               + str(self.Title) \
               +"\n"+'Plot 2. pariplot: '\
               +'\n'+'Plot 3. violinplot: '\
               +'\n'+'Plot 4. catplot: '