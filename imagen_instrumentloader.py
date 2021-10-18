#################################################################################
""" IMAGEN Instrument Loader using H5DF file in all Session """
# Author: JiHoon Kim, <jihoon.kim@fu-berlin.de>, 18th October 2021
#
import os
import h5py
import pandas as pd
import numpy as np
import warnings
from glob import glob
from itertools import chain, repeat
warnings.filterwarnings('ignore')

class INSTRUMENT_loader:
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN"):
        """ Set up path
        
        Parameters
        ----------
        DATA_DIR : string, optional
            Directory IMAGEN absolute path
        
        """
        # Set the directory path: IMAGEN
        self.DATA_DIR = DATA_DIR
    
    def set_instrument(self, DATA, save=False):
        """ Save all session instrument in one file
        
        Parameters
        ----------
        DATA : string,
            instrument name
        save : boolean,
            save the pandas.dataframe to .csv file
            
        Returns
        -------
        DF3 : pandas.dataframe
            instrument in all session (BL, FU1, FU2, FU3)
            
        Examples
        --------
        >>> from imagen_instrumentloader import *
        >>> DATA = Instrument_loader()
        >>> DF3 = DATA.set_instrument(
        ...     'NEO', # instrument
        ...     save = True)  
        
        Notes
        -----
        If only one session has information,
        the same value is copied to all sessions based on ID.
        (e.g. CTQ)
        
        """
        # ----------------------------------------------------- #
        # ROI Columns: Psychological profile                    #
        # ----------------------------------------------------- #        
        if DATA == "NEO":
            # Set the files with session and roi columns
            NEO = [
                ('FU3','IMAGEN-IMGN_NEO_FFI_FU3.csv'),
                ('FU2','IMAGEN-IMGN_NEO_FFI_FU2-IMAGEN_SURVEY_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_NEO_FFI_CHILD_FU_RC5-IMAGEN_SURVEY_DIGEST.csv'),
                ('BL', 'IMAGEN-IMGN_NEO_FFI_CHILD_RC5-IMAGEN_SURVEY_DIGEST.csv')
            ]
            ROI = ['ID','Session','open_mean','cons_mean','extr_mean','agre_mean','neur_mean']
            # Generate the instrument files in one dataframe
            NEO_LIST = []
            for SES, CSV in NEO:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF2 = DF[ROI]
                NEO_LIST.append(DF2)
            NEO = pd.concat(NEO_LIST)
            # Rename the columns
            DF3 = NEO.rename(
                columns = {
                    "neur_mean" : "Neuroticism mean",
                    "extr_mean" : "Extroversion mean",
                    "open_mean" : "Openness mean",
                    "agre_mean" : "Agreeableness mean",
                    "cons_mean" : "Conscientiousness mean",
                }
            )

        if DATA == "SURPS":
            # Set the files with session and roi columns
            SURPS = [
                ('FU3','IMAGEN-IMGN_SURPS_FU3.csv'),
                ('FU2','IMAGEN-IMGN_SURPS_FU2-IMAGEN_SURVEY_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_SURPS_FU_RC5-IMAGEN_SURVEY_DIGEST.csv'),
                ('BL', 'IMAGEN-IMGN_SURPS_RC5-IMAGEN_SURVEY_DIGEST.csv')
            ]
            ROI = ['ID', 'Session', 'as_mean', 'h_mean', 'imp_mean', 'ss_mean']
            # Generate the instrument files in one dataframe
            SURPS_LIST = []
            for SES, CSV in SURPS:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF2 = DF[ROI]
                SURPS_LIST.append(DF2)
            SURPS = pd.concat(SURPS_LIST)
            # Rename the columns
            DF3 = SURPS.rename(
                columns = {
                    "as_mean" : "Anxiety Sensitivity mean",
                    "h_mean"  : "Hopelessness mean",
                    "imp_mean": "Impulsivity mean",
                    "ss_mean" : "Sensation seeking mean",
                }
            )

        # ----------------------------------------------------- #
        # ROI Columns: Socio-economic profile                   #
        # ----------------------------------------------------- #
        if DATA == "CTQ":
            # Set the files with session and roi columns
            CTQ = [
                ('FU3','IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv'),
                ('FU2','IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv'),
                ('BL', 'IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv')
            ]
            ROI = ['ID','Session','ea_sum','pa_sum','sa_sum','en_sum','pn_sum','dn_sum']
            # Set the columns for computation
            emot_abu = ['CTQ_3','CTQ_8','CTQ_14','CTQ_18','CTQ_25']
            phys_abu = ['CTQ_9','CTQ_11','CTQ_12','CTQ_15','CTQ_17']
            sexual_abu = ['CTQ_20','CTQ_21','CTQ_23','CTQ_24','CTQ_27']
            emot_neg = ['CTQ_5','CTQ_7','CTQ_13','CTQ_19','CTQ_28']
            phys_neg = ['CTQ_1','CTQ_2','CTQ_4','CTQ_6','CTQ_26']
            denial = ['CTQ_10','CTQ_16','CTQ_22']
            # Generate the instrument files in one dataframe
            CTQ_LIST = []
            for SES, CSV in CTQ:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/FU2/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF['ea_sum'] = DF[emot_abu].sum(axis=1,skipna=False)
                DF['pa_sum'] = DF[phys_abu].sum(axis=1,skipna=False)
                DF['sa_sum'] = DF[sexual_abu].sum(axis=1,skipna=False)
                DF['en_sum'] = DF[emot_neg].sum(axis=1,skipna=False)
                DF['pn_sum'] = DF[phys_neg].sum(axis=1,skipna=False)
                DF['dn_sum'] = DF[denial].sum(axis=1, skipna=False)
                DF2 = DF[ROI]
                CTQ_LIST.append(DF2)
            CTQ = pd.concat(CTQ_LIST)
            # Rename the columns
            DF3 = CTQ.rename(
                columns = {
                    "ea_sum" : "Emotional abuse sum",
                    "pa_sum" : "Physical abuse sum",
                    "sa_sum" : "Sexual abuse sum",
                    "en_sum" : "Emotional neglect sum",
                    "pn_sum" : "Physical neglect sum",
                    "dn_sum" : "Denial sum"
                }
            )

        if DATA == "CTS":
            # Set the files with session and roi columns
            CTS = [
                ('FU3','IMAGEN-IMGN_CTS_PARENT_RC5-BASIC_DIGEST.csv'),
                ('FU2','IMAGEN-IMGN_CTS_PARENT_RC5-BASIC_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_CTS_PARENT_RC5-BASIC_DIGEST.csv'),
                ('BL', 'IMAGEN-IMGN_CTS_PARENT_RC5-BASIC_DIGEST.csv')
            ]
            ROI = [
                'ID','Session','cts_assault','cts_injury','cts_negotiation',
                'cts_psychological_aggression','cts_sexual_coercion'
            ]
            # Generate the instrument files in one dataframe
            CTS_LIST = []
            for SES, CSV in CTS:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/BL/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF2 = DF[ROI]
                CTS_LIST.append(DF2)
            CTS = pd.concat(CTS_LIST)
            # Rename the columns
            DF3 = CTS.rename(
                columns = {
                    "cts_assault"                  : "Assault mean",
                    "cts_injury"                   : "Injury mean",
                    "cts_negotiation"              : "Negotiation mean",
                    "cts_psychological_aggression" : "Psychological Aggression mean",
                    "cts_sexual_coercion"          : "Sexual Coercion mean"
                }
            )

        if DATA == 'LEQ':
            # Set the files with session and roi columns
            LEQ = [
                ('FU3','IMAGEN-IMGN_LEQ_FU3.csv'),
                ('FU2','IMAGEN-IMGN_LEQ_FU2-IMAGEN_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_LEQ_FU_RC5-IMAGEN_DIGEST.csv'),
                ('BL' ,'IMAGEN-IMGN_LEQ_RC5-BASIC_DIGEST.csv')
            ]
            ROI = [
                'ID','Session','family_valence','accident_valence','sexuality_valence',
                'autonomy_valence','devience_valence','relocation_valence',
                'distress_valence','noscale_valence','overall_valence',
                'family_ever_meanfreq','accident_ever_meanfreq','sexuality_ever_meanfreq',
                'autonomy_ever_meanfreq','devience_ever_meanfreq','relocation_ever_meanfreq',
                'distress_ever_meanfreq','noscale_ever_meanfreq','overall_ever_meanfreq'
            ]
            # Generate the instrument files in one dataframe
            LEQ_LIST = []
            for SES, CSV in LEQ:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF2 = DF[ROI]
                LEQ_LIST.append(DF2)
            LEQ = pd.concat(LEQ_LIST)
            # Rename the columns
            DF3 = LEQ.rename(
                columns = {
                    # Mean valence of events
                    "family_valence"           : "Family valence",
                    "accident_valence"         : "Accident valence",
                    "sexuality_valence"        : "Sexuality valence",
                    "autonomy_valence"         : "Autonomy valence",
                    "devience_valence"         : "Devience valence",
                    "relocation_valence"       : "Relocation valence",
                    "distress_valence"         : "Distress valence",
                    "noscale_valence"          : "Noscale valence",
                    "overall_valence"          : "Overall valence",
                    # Mean frequency lifetime
                    "family_ever_meanfreq"     : "Family mean frequency",
                    "accident_ever_meanfreq"   : "Accident mean frequency",
                    "sexuality_ever_meanfreq"  : "Sexuality mean frequency",
                    "autonomy_ever_meanfreq"   : "Autonomy mean frequency",
                    "devience_ever_meanfreq"   : "Devience mean frequency",
                    "relocation_ever_meanfreq" : "Relocation mean frequency",
                    "distress_ever_meanfreq"   : "Distress mean frequency",
                    "noscale_ever_meanfreq"    : "Noscale mean frequency",
                    "overall_ever_meanfreq"    : "Overall mean frequency",
                }
            )
            
        if DATA == 'PBQ':
            # Set the files with session and roi columns
            PBQ = [
                ('FU1','IMAGEN-IMGN_PBQ_FU_RC1-BASIC_DIGEST.csv'),
                ('BL', 'IMAGEN-IMGN_PBQ_RC1-BASIC_DIGEST.csv')
            ]
            ROI = [
                'ID','Session','pbq_03','pbq_03a','pbq_03b','pbq_03c',
                'pbq_05','pbq_05a','pbq_05b','pbq_05c','pbq_06','pbq_06a',
                'pbq_12','pbq_13','pbq_13a','pbq_13b','pbq_13g',
            ]
            # Generate the columns
            def test(x):
                if x == 0: return 'No'
                elif x == 1: return 'Yes'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN
            def day(x):
                if x == 2: return 'yes, every day'
                elif x == 1: return 'yes, on occasion'
                elif x == 0: return 'no, not at all'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN
            def age(x):
                if x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return x
            def cigarettes(x):
                if x == 1: return 'Less than 1 cigarette per week'
                elif x == 2: return 'Less than 1 cigarette per day'
                elif x == 3: return '1-5 cigarettes per day'
                elif x == 8: return '6-10 cigarettes per day'
                elif x == 15: return '11-20 cigarettes per day'
                elif x == 25: return '21-30 cigarettes per day'
                elif x == 30: return 'More than 30 cigarettes per day'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN
            def alcohol(x):
                if x == 1: return 'Monthly or less'
                elif x == 2: return 'Two to four times a month'
                elif x == 3: return 'Two to three times a week'
                elif x == 4: return 'Four or more times a week'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN
            def drinks(x):
                if x == 0: return '1 or 2'
                elif x == 1: return '3 or 4'
                elif x == 2: return '5 or 6'
                elif x == 3: return '7 to 9'
                elif x == 4: return '10 or more'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN
            def stage(x):
                if x == 1: return 'first trimester'
                elif x == 2: return 'second trimester'
                elif x == 3: return 'third trimester'
                elif x == 12: return 'first and second'
                elif x == 23: return 'second and third'
                elif x == 13: return 'first and third'
                elif x == 4: return 'whole pregnancy'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN            
            # Generate the instrument files in one dataframe
            PBQ_LIST = []
            for SES, CSV in PBQ:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                # Rename the values
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF["pbq_03"] = DF['pbq_03'].apply(test)
                DF['pbq_03a'] = DF['pbq_03a'].apply(day)
                DF['pbq_03b'] = DF['pbq_03b'].apply(age)
                DF["pbq_03c"] = DF['pbq_03c'].apply(cigarettes)
                DF["pbq_05"] = DF['pbq_05'].apply(test)
                DF["pbq_05a"] = DF['pbq_05a'].apply(cigarettes)
                DF["pbq_05b"] = DF['pbq_05b'].apply(cigarettes)
                DF["pbq_05c"] = DF['pbq_05c'].apply(cigarettes)
                DF["pbq_06"] = DF['pbq_06'].apply(test)
                DF["pbq_06a"] = DF['pbq_06a'].apply(cigarettes)
                DF["pbq_12"] = DF['pbq_12'].apply(test)
                DF["pbq_13"] = DF['pbq_13'].apply(test)
                DF["pbq_13a"] = DF['pbq_13a'].apply(alcohol)
                DF["pbq_13b"] = DF['pbq_13b'].apply(drinks)
                DF["pbq_13g"] = DF['pbq_13g'].apply(stage)
                DF2 = DF[ROI]
                PBQ_LIST.append(DF2)
            PBQ = pd.concat(PBQ_LIST)
            # Exclude the rows:          
            # Duplicate ID: 71766352, 58060181, 15765805, 15765805 in FU1
            for i in [71766352, 58060181, 15765805, 12809392]:
                is_out = (PBQ['ID']==i) & (PBQ['Session']=='FU1')
                PBQ = PBQ[~is_out]
            # Different ID: 12809392 in both BL and FU1
            for i in [12809392]:
                is_out = (PBQ['ID']==i) & (PBQ['Session']=='BL')
                PBQ = PBQ[~is_out]
            DF3 = PBQ
        
        if DATA == 'GEN':
            # Set the files with session and roi columns
            GEN = [
                ('FU3','IMAGEN-IMGN_GEN_RC5-BASIC_DIGEST.csv'),
                ('FU2','IMAGEN-IMGN_GEN_RC5-BASIC_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_GEN_RC5-BASIC_DIGEST.csv'),
                ('BL','IMAGEN-IMGN_GEN_RC5-BASIC_DIGEST.csv')
            ]
            ROI = ['ID','Session','Paternal_disorder','Maternal_disorder']
            # Generate the columns
            def disorder(x):
                if x == 'ALC': return 'Alcohol problems'
                elif x == 'DRUG': return 'Drug problems'
                elif x == 'SCZ': return 'Schizophrenia'
                elif x == 'SCZAD': return 'Schizoaffective Disorder'
                elif x == 'DPR_R': return 'Major Depression recurrent'
                elif x == 'DPR_SE': return 'Major Depression single episode'
                elif x == 'BIP_I': return 'Bipolar I Disorder'
                elif x == 'BIP_II': return 'Bipolar II Disorder'
                elif x == 'OCD': return 'Obessive-compulsive Disroder'
                elif x == 'ANX': return 'Anxiety Disorder'
                elif x == 'EAT': return 'Eating Disorder'
                elif x == 'SUIC': return 'Suicide / Suicidal Attempt'
                elif x == 'OTHER': return 'Other'
                else: return np.NaN
            # Generate the instrument files in one dataframe
            GEN_LIST = []
            for SES, CSV in GEN:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/BL/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF['Disorder_PF_1'] = DF['Disorder_PF_1'].apply(disorder)
                DF['Disorder_PF_2'] = DF['Disorder_PF_2'].apply(disorder)
                DF['Disorder_PF_3'] = DF['Disorder_PF_3'].apply(disorder)
                DF['Disorder_PF_4'] = DF['Disorder_PF_4'].apply(disorder)
                DF['Disorder_PM_1'] = DF['Disorder_PM_1'].apply(disorder)
                DF['Disorder_PM_2'] = DF['Disorder_PM_2'].apply(disorder)
                DF['Disorder_PM_3'] = DF['Disorder_PM_3'].apply(disorder)
                DF['Disorder_PM_4'] = DF['Disorder_PM_4'].apply(disorder)
                DF['Disorder_PM_5'] = DF['Disorder_PM_5'].apply(disorder)
                DF['Disorder_PM_6'] = DF['Disorder_PM_6'].apply(disorder)                
                Variables = [
                    'ID','Session','Disorder_PF_1','Disorder_PF_2','Disorder_PF_3',
                    'Disorder_PF_4','Disorder_PM_1','Disorder_PM_2','Disorder_PM_3',
                    'Disorder_PM_4','Disorder_PM_5','Disorder_PM_6'
                ]
                Check_DF = DF[Variables]
                Check_DF['Paternal_disorder'] = Check_DF.loc[:, Check_DF.columns[2:6]].apply(
                    lambda x: ','.join(x.dropna().astype(str)), axis=1)
                Check_DF['Maternal_disorder'] = Check_DF.loc[:, Check_DF.columns[6:12]].apply(
                    lambda x: ','.join(x.dropna().astype(str)), axis=1)
                DF2 = Check_DF[ROI]
                GEN_LIST.append(DF2)
            GEN = pd.concat(GEN_LIST)
            DF3 = GEN

        # ----------------------------------------------------- #
        # ROI Columns: Other co-morbidities                     #
        # ----------------------------------------------------- #
        if DATA == 'FTND':
            # Set the files with session and roi columns
            FTND = [
                ('FU3','IMAGEN-IMGN_ESPAD_FU3.csv'),
                ('FU2','IMAGEN-IMGN_ESPAD_CHILD_FU2-IMAGEN_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_ESPAD_CHILD_FU_RC5-IMAGEN_DIGEST.csv'),
                ('BL','IMAGEN-IMGN_ESPAD_CHILD_RC5-IMAGEN_DIGEST.csv')
            ]
            ROI = ['ID','Session','Likelihood of nicotine dependence child']
            # Generate the columns
            def test(x):
                if (7<=x and x <=10): return 'highly dependent'
                elif (4<=x and x <=6): return 'moderately dependent'
                elif (x<4): return 'less dependent'
                else: return np.NaN
            # Generate the instrument files in one dataframe
            FTND_LIST = []
            for SES, CSV in FTND:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                # Rename the values
                DF['Likelihood of nicotine dependence child'] = DF['ftnd_sum'].apply(test)
                DF2 = DF[ROI]
                FTND_LIST.append(DF2)
            FTND = pd.concat(FTND_LIST)
            DF3 = FTND

        if DATA == 'DAST':
            # Set the files with session and roi columns
            
            # Generate the columns
            
            # Generate the instrument files in one dataframe
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'DAST' == self.DATA: # 'DAST'
#                 self.VARIABLES, self.NEW_DF2 = DAST_SESSION(self.SESSION)
            pass

        if DATA == 'SCID':
            # Set the files with session and roi columns
            
            # Generate the columns
            
            # Generate the instrument files in one dataframe
#         def SCID_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'SCID' == self.DATA: # 'SCID'
#                 self.VARIABLES, self.NEW_DF2 = SCID_SESSION(self.SESSION)
            pass

        if DATA == 'DMQ':
            # Set the files with session and roi columns
            
            # Generate the columns
            
            # Generate the instrument files in one dataframe
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'DMQ' == self.DATA: # 'DMQ'
#                 self.VARIABLES, self.NEW_DF2 = DMQ_SESSION(self.SESSION)
            pass

        if DATA == 'BSI':
            # Set the files with session and roi columns
            
            # Generate the columns
            
            # Generate the instrument files in one dataframe
#             if SESSION == 'FU3':
#                 ## Somatization
#                 Somatization_labels = ['BSI_02', 'BSI_07', 'BSI_23',
#                                        'BSI_29', 'BSI_30', 'BSI_33', 'BSI_37']
#                 ## Obsession-Compulsion
#                 Obsession_compulsion_labels = ['BSI_05', 'BSI_15', 'BSI_26',
#                                                'BSI_27', 'BSI_32', 'BSI_36']
#                 ## Interpersonal Sensitivity
#                 Interpersonal_sensitivity_labels = ['BSI_20', 'BSI_21',
#                                                     'BSI_22', 'BSI_42']
#                 ## Depression
#                 Depression_labels = ['BSI_09', 'BSI_16', 'BSI_17',
#                                      'BSI_18', 'BSI_35', 'BSI_50']
#                 ## Anxiety
#                 Anxiety_labels = ['BSI_01', 'BSI_12', 'BSI_19',
#                                   'BSI_38', 'BSI_45', 'BSI_49']
#                 ## Hostility
#                 Hostility_labels = ['BSI_06', 'BSI_13', 'BSI_40',
#                                     'BSI_41', 'BSI_46']
#                 ## Phobic Anxiety
#                 Phobic_anxiety_labels = ['BSI_08', 'BSI_28', 'BSI_31',
#                                          'BSI_43', 'BSI_47']
#                 ## Paranoid Ideation
#                 Paranoid_ideation_labels = ['BSI_04', 'BSI_10', 'BSI_24',
#                                             'BSI_48', 'BSI_51']
#                 ## Psychoticism
#                 Psychoticism_labels = ['BSI_03', 'BSI_14', 'BSI_34',
#                                        'BSI_44', 'BSI_53']
#                 DF_BSI['somatization_mean'] = DF_CTQ[Somatization_labels].mean(axis=1,
#                                                                                skipna=False)
#                 DF_BSI['obsession_compulsion_mean'] = DF_CTQ[Somatization_labels].mean(axis=1,
#                                                                                skipna=False)
            pass

        if DATA == 'AUDIT':
            # Set the files with session and roi columns
            
            # Generate the columns
            
            # Generate the instrument files in one dataframe
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#         elif 'AUDIT' == self.DATA: # 'AUDIT'
#             self.VARIABLES, self.NEW_DF2 = AUDIT_SESSION(self.SESSION)
            pass

        if DATA == 'MAST':
            # Set the files with session and roi columns
            
            # Generate the columns
            
            # Generate the instrument files in one dataframe
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#         elif 'MAST' == self.DATA: # 'MAST'
#             self.VARIABLES, self.NEW_DF2 = MAST_SESSION(self.SESSION)  
            pass

        if save == True:
            save_path = f"{self.DATA_DIR}/Instrument/{DATA}.csv"
            # set the save option
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            DF3.to_csv(save_path, index=None)
        return DF3

    def get_instrument(self, LIST, NAME):
        """ Select the ROI instruments as file
        
        Parameters
        ----------
        LIST : list
            instrument roi name list
        NAME : string
            set the name of the csv
        save : boolean
            save it in the folder
            
        Examples
        --------
        >>> from imagen_instrumentloader import *
        >>> DATA = Instrument_loader()
        >>> DF3 = DATA.get_instrument(
        ...     LIST,                            # instrument list
        ...     NAME,                            # instrument name
        ...     save = True)

        """
#         self.load_HDF5(h5py_file)
#         self.load_INSTRUMENT(SESSION, DATA, instrument_file)
#         self.generate_new_instrument(save, viz)

        # All Instrument
#         NEO_SURPS = pd.merge(NEO, SURPS, on=['ID','Session'], how='outer')
#         NEO_SURPS_CTQ = pd.merge(NEO_SURPS, CTQ, on=['ID','Session'], how='outer')
#         NSC_CTS = pd.merge(NEO_SURPS_CTQ, CTS, on=['ID','Session'], how='outer')
#         NSCCL = pd.merge(NSC_CTS, LEQ, on=['ID','Session'], how='outer')
        
#         save = True
#         save_absolute_path = f"{DATA_DIR}/Instrument/instrument.csv"
#         if save == True:
#             # set the save option
#             if not os.path.isdir(os.path.dirname(save_absolute_path)):
#                 os.makedirs(os.path.dirname(save_absolute_path))
#             NSCCL.to_csv(save_absolute_path, index=None)
        pass
    
#     def __str__(self):
#         """ Print the instrument loader steps """
#         return "Step 1. load the instrument: " \
#                + "\n        File = " + str(self.instrument_path) \
#                + "\n        The dataset contains " + str(self.DF.shape[0]) \
#                + " samples and " + str(self.DF.shape[1]) + " columns" \
#                + "\n        Variables = " + str(self.VARIABLES)
#             print(f"{'-'*83} \n{self.__str__()} \n{'-'*83}")
#             print(f"{self.NEW_DF.info(), self.NEW_DF.describe()}")

class HD5F_loader:
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN"):
        """ Set up path
        
        Parameters
        ----------
        DATA_DIR : string, optional
            Directory IMAGEN absolute path
        
        """
        # Set the directory path: IMAGEN
        self.DATA_DIR = DATA_DIR    

    def set_HD5F(self, DATA, save=False):
        """ Save all session y in one file
        
        Parameters
        ----------
        DATA : string,
            y name
        save : boolean,
            save the pandas.dataframe to .csv file
            
        Returns
        -------
        DF3 : pandas.dataframe
            instrument in all session (BL, FU1, FU2, FU3)

        Examples
        --------
        >>> from imagen_instrumentloader import *
        >>> DATA = HD5F_loader()
        >>> DF3 = DATA.set_HD5F(
        ...     DATA,                           # instrument
        ...     save = True)                    # summary  

        Notes
        -----
        There are no session in FU2
        
        """
        if DATA == "Binge":
            # Set the files with session and roi columns
            BINGE = [
                ('FU3','Training','newlbls-fu3-espad-fu3-19a-binge-n650.h5'),
                ('FU3','Holdout', 'newholdout-fu3-espad-fu3-19a-binge-n102.h5'),
                ('FU2','Training','newlbls-fu2-espad-fu3-19a-binge-n634.h5'),
                ('FU2','Holdout', 'newholdout-fu2-espad-fu3-19a-binge-n102.h5'),
                ('BL', 'Training','newlbls-bl-espad-fu3-19a-binge-n620.h5'),
                ('BL', 'Holdout', 'newholdout-bl-espad-fu3-19a-binge-n102.h5')
            ]
            ROI = ['ID','Session','y','Dataset','Sex','Site','Class']
            # Generate the instrument files in one dataframe
            BINGE_LIST = []
            for SES, DATASET, HD5F in BINGE:
                path = f"{self.DATA_DIR}/h5files/{HD5F}"
                # Convert HD5F to List
                d = h5py.File(path,'r')
                # Set All, HC, and AAM
                b_list = list(np.array(d[list(d.keys())[0]]))
                ALL = list(np.array(d['i']))
                HC = [ALL[i] for i, j in enumerate(b_list) if j%2==0]
                AAM = [ALL[i] for i, j in enumerate(b_list) if j%2==1]
                # Set Sex
                sex = list(np.array(d['sex']))
                SEX = ['Male' if i==0 else 'Female' for i in sex]
                # Set Site
                sites = list(np.array(d['site']))
                center = {0: 'Paris', 1: 'Nottingham', 2:'Mannheim', 3:'London',
                          4: 'Hamburg', 5: 'Dublin', 6:'Dresden', 7:'Berlin'}
                SITE = [center[i] for i in sites]
                # Set Class
                target = list(np.array(d[list(d.keys())[0]]))
                CLASS = ['HC' if i==0 else 'AAM' for i in target]
                # Generate the DF
                DF2 = pd.DataFrame(
                    {"ID" : ALL,
                    "Session" : SES,
                    "y" : list(d.keys())[0],
                    "Dataset" : DATASET,
                    "Sex" : SEX,
                    "Site" : SITE,
                    "Class" : CLASS}
                )
                BINGE_LIST.append(DF2)
            DF3 = pd.concat(BINGE_LIST)

        if save == True:
            save_path = f"{self.DATA_DIR}/Instrument/{DATA}.csv"
            # set the save option
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            DF3.to_csv(save_path, index=None)
        return DF3
    
class IMAGEN_instrument(INSTRUMENT_loader, HD5F_loader):
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN"):
        """ Set up path
        
        Parameters
        ----------
        DATA_DIR : string, optional
            Directory IMAGEN absolute path
        
        """
        # Set the directory path: IMAGEN
        self.DATA_DIR = DATA_DIR
        
    def read_HD5F(self, h5py_file):
        """ Generate the list,
        all subject (ALL), healthy control (HC),
        adolscent alcohol misuse (AAM), Sex, Site, and Class
        
        Parameters
        ----------
        h5py_file : string,
            The IMAGEN's h5df file (*.csv)
        
        Returns
        -------
        self.HD5F : pandas.dataframe
            The HD5F dataframe
            
        Examples
        --------
        >>> from imagen_instrumentloader import *
        >>> DATA = IMAGEN_instrument()
        >>> HD5F = DATA.read_HD5F(
        ...     h5py_file)                      # instrument

        Notes
        -----
        Dataset:
        Training and Holdout

        """
        # Set HD5F file
        self.hd5f_file = h5py_file
        # Load the hd5f file
        hd5f_path = f"{self.DATA_DIR}/Instrument/{self.hd5f_file}"
        DF = pd.read_csv(hd5f_path, low_memory=False)
        self.HD5F = DF
        return self.HD5F
    
#         Hd5f = INST.read_HD5F('Binge.csv')
#         Hd5f_FU3 = Hd5f.groupby('Session').get_group('FU3')

    def read_INSTRUMENT(self, instrument_file):
        """ Load the INSTRUMENT file
        
        Parameters
        ----------            
        instrument_file : string
            The IMAGEN's instrument file (*.csv)

        Returns
        -------
        self.INST : pandas.dataframe
            The Instrument dataframe
        
        Notes
        -----
        This function rename the columns and select the ROI columns:
        Psychological profile:
            NEO, SURPS
        Socio-economic profile:
            CTQ, CTS, LEQ, PBQ, GEN
        Other co-morbidities:
            FTND, DAST, SCID, DMQ, BSI, AUDIT, MAST

        Examples
        --------
        >>> from imagen_instrumentloader import *
        >>> DATA = IMAGEN_instrument()
        >>> DF = DATA.read_INSTRUMENT(
        ...     instrument_file)               # instrument
        ... DF_FU3 = DF.groupby('Session').get_group('FU3')

        """
        # Set Instrument file
        self.instrument_file = instrument_file
        # Load the instrument file       
        instrument_path = f"{self.DATA_DIR}/Instrument/{self.instrument_file}"
        DF = pd.read_csv(instrument_path, low_memory=False)
        self.INST = DF
        return self.INST
            

    
    def read_RUN(self, run_file, save=False):
        """ Load the ML RUN result in both Test & Holdout in all session
        
        Parameters
        ----------
        run_file : string
            ML models result run.csv path
        save : boolean
            if save == True, then save it as .csv
        
        Returns
        -------
        self.RUN : pandas.dataframe
            The RUN dataframe
        
        Examples
        --------
        >>> from imagen_instrumentloader import *
        >>> DATA = IMAGEN_instrument()
        >>> DF = DATA.read_RUN(
        ...     run_file)                             # run
        ... DF_FU3 = DF.groupby('Session').get_group('FU3')
        
        """
        df = pd.read_csv(run_file, low_memory = False)
        
        DF_RUN = []
        for i in range(83):
            # Test
            test_ids = eval(df['test_ids'].values[i])
            test_lbls = eval(df['test_lbls'].values[i])
            test_probs = [probs[1] for probs in eval(df['test_probs'].values[i])]
            # Holdout
            holdout_ids = eval(df['holdout_ids'].values[i])
            holdout_lbls = eval(df['holdout_lbls'].values[i])
            holdout_preds = [probs[1] for probs in eval(df['holdout_preds'].values[i])]
#             rename the columns may be needed
            DF_TEST = pd.DataFrame({
                # Model configuration
                "i" : df.iloc[i][7],
                "o" : df.iloc[i][8],
                "io" : df.iloc[i][1],
                "technique" : df.iloc[i][2],
                "Session" : df.iloc[i][25],
                "Trial" : df.iloc[i][4],
                "path" : df.iloc[i][24],
                "n_samples" : df.iloc[i][5],
                "n_samples_cc" : df.iloc[i][6],
                "i_is_conf" : df.iloc[i][9],
                "o_is_conf" : df.iloc[i][10],
                "Model" : df.iloc[i][3],
                "model_SVM-rbf__C" : df.iloc[i][18],
                "model_SVM-rbf__gamma" : df.iloc[i][19],
                "runtime" : df.iloc[i][20],
                "model_SVM-lin__C" : df.iloc[i][21],
                "model_GB__learning_rate" : df.iloc[i][22],
                "model_LR__C" : df.iloc[i][23],
                # Result
                "train_score" : df.iloc[i][11],
                "valid_score" : df.iloc[i][12],
                "test_score" : df.iloc[i][13],
                "roc_auc" : df.iloc[i][14],
                "holdout_score" : df.iloc[i][26],
                "holdout_roc_auc" : df.iloc[i][27],
                # Test
                "dataset" : "Test set",
                "ID" : test_ids,
                "true_label" : test_lbls,
                "prediction" : test_probs,
            })
            DF_HOLDOUT = pd.DataFrame({
                # Model configuration
                "i" : df.iloc[i][7],
                "o" : df.iloc[i][8],
                "io" : df.iloc[i][1],
                "technique" : df.iloc[i][2],
                "Session" : df.iloc[i][25],
                "Trial" : df.iloc[i][4],
                "path" : df.iloc[i][24],
                "n_samples" : df.iloc[i][5],
                "n_samples_cc" : df.iloc[i][6],
                "i_is_conf" : df.iloc[i][9],
                "o_is_conf" : df.iloc[i][10],
                "Model" : df.iloc[i][3],
                "model_SVM-rbf__C" : df.iloc[i][18],
                "model_SVM-rbf__gamma" : df.iloc[i][19],
                "runtime" : df.iloc[i][20],
                "model_SVM-lin__C" : df.iloc[i][21],
                "model_GB__learning_rate" : df.iloc[i][22],
                "model_LR__C" : df.iloc[i][23],
                # Result
                "train_score" : df.iloc[i][11],
                "valid_score" : df.iloc[i][12],
                "test_score" : df.iloc[i][13],
                "roc_auc" : df.iloc[i][14],
                "holdout_score" : df.iloc[i][26],
                "holdout_roc_auc" : df.iloc[i][27],
                # Holdout
                "dataset" : "Holdout set",
                "ID" : holdout_ids,
                "true_label" : holdout_lbls,
                "prediction" : holdout_preds
            })
            DF_RUN.append(DF_TEST)
            DF_RUN.append(DF_HOLDOUT)
        RUN = pd.concat(DF_RUN)
#         rename the values may be needed
        self.RUN = RUN
        if save == True:
            save_absolute_path = 'IMAGEN_run.csv'
            RUN.to_csv(save_absolute_path, index=False)
        return self.RUN
    
    def to_posthoc(self, h5py, instruemnt, run):
        """ Set the Posthoc file

        Examples
        --------
        >>> from imagen_instrumentloader import *
        >>> NEO = IMAGEN_instrument()
        >>> df_binge_FU3_NEO = NEO.read_instrument(
        ...     "IMAGEN-IMGN_NEO_FFI_FU3.csv",   # instrument
        ...     viz = True)                      # summary  

        Notes
        -----
        # ----------------------------------------------------- #
        # Data, Session, ID, Sex, Site, Class columns           #
        # ----------------------------------------------------- #
        DF_2 = self.DF.set_index('ID').reindex(self.ALL)
        DF_2['Data'] = self.h5py_file[:-3]
        DF_2['Session'] = self.SESSION
        DF_2['ID'] = DF_2.index
        DF_2['Sex'] = self.SEX
        DF_2['Site'] = self.SITE
        DF_2['Class'] = self.CLASS
        
        """
        read_HDF5()
        read_INSTRUMENT()
        read_RUN()

#     def __str__(self):
#         """ Print the instrument loader steps """
#         return "Step 1. load the phenotype: " \
#                + str(self.h5py_file.replace(".h5", "")) \
#                + "\n        Class = " + str(list(self.d.keys())[0]) \
#                + ", n = " + str(len(self.ALL)) + " (HC = " \
#                + str(len(self.HC)) + ", AAM = " + str(len(self.AAM)) +")" \
#                + "\n" + "Step 2. load the instrument dataset: " \
#                + str(self.instrument_file.replace(".csv",'')) \
#                + "\n" + "Step 3. generate the " + str(self.SESSION) +"_" \
#                + str(self.h5py_file.replace(".h5", "")) \
#                + str(self.instrument_file.replace("IMAGEN-IMGN", "")) \
#                + "\n        The dataset contains " + str(self.NEW_DF.shape[0]) \
#                + " samples and " + str(self.NEW_DF.shape[1]) + " columns" \
#                + "\n" + "Step 4. select " + str(self.SESSION) +"_" \
#                + str(self.h5py_file.replace(".h5", "")) \
#                + str(self.instrument_file.replace("IMAGEN-IMGN", "")) \
#                + "\n        The dataset contains " + str(self.NEW_DF.shape[0]) \
#                + " samples and " + str(self.NEW_DF.shape[1]) + " columns" \
#                + "\n        Variables = " + str(self.Variables)
#################################################################################