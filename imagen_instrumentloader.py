#################################################################################
## Last-modified :  15th August 2021
## Author : JiHoon Kim
##
## USAGE EXAMPLE
## from imagen_instrumentloader import * 
## DATA_DIR = '/ritter/share/data/IMAGEN'
## binge_FU3_NEO = IMAGEN_instrument()
## _ = binge_FU3_NEO.load_HDF5("newlbls-fu3-espad-fu3-19a-binge-n650.h5")
## _ = binge_FU3_NEO.load_INSTRUMENT("FU3", "NEO", "IMAGEN-IMGN_NEO_FFI_FU3.csv")
## df_binge_FU3_NEO = binge_FU3_NEO.generate_new_instrument(save = True)
##
## # Equivalent to
## from imagen_instrumentloader import *  
## DATA_DIR = '/ritter/share/data/IMAGEN'
## binge_FU3_NEO = IMAGEN_variables()
## df_binge_FU3_NEO = binge_FU3_NEO.to_instrument(
##     "newlbls-fu3-espad-fu3-19a-binge-n650.h5", # h5files
##     "FU3",                                     # session
##     "IMAGEN-IMGN_NEO_FFI_FU3.csv",             # instrument
##     "NEO",                                     # name
##     save = False)                              # save
##
## # Pipeline wise
## Hdf5 = "newlbls-fu3-espad-fu3-19a-binge-n650.h5"
## Session = "FU3"
## Instrument = ["IMAGEN-IMGN_NEO_FFI_FU3.csv", # NEO-Fünf-Faktoren-Inventar
##               "IMAGEN-IMGN_SURPS_FU3.csv",
##               "IMAGEN-IMGN_ESPAD_FU3.csv", # FTND contained in ESPAD-Syntax
##               "IMAGEN-IMGN_LEQ_FU3.csv"]
## for i in Instrument:
##     _ = c.get_instrument(Hdf5, Session, i, save = True)
#################################################################################

import os
import h5py
import pandas as pd
import numpy as np

class IMAGEN_instrument:
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN"):
        # Set the directory path: IMAGEN
        self.DATA_DIR = DATA_DIR

    def load_HDF5(self, h5py_file):
        """
        Load HDF5 file (.h5) and convert it to list:
                all subject (ALL), healthy control (HC),
                adolscent alcohol misuse (AAM), Sex, Site, and Class

        Argument:
                h5py_file: IMAGEN's h5df file name
        
        Output:
                ALL: all ID list
                SEX: Male and Female
                SITE: Paris, Nottingham, Mannheim, London,
                      Hamburg, Dublin, Dresden, and Berlin
                CLASS: all target HC and AAM
        """     
        # If OSError, export HDF5_USE_FILE_LOCKING='FALSE'
        # set the h5 file absolute path
        self.h5py_file = h5py_file
        h5py_absolute_path = f'{self.DATA_DIR}/h5files/{self.h5py_file}'
        
        # Convert h5py to list
        self.d = h5py.File(h5py_absolute_path, 'r')
        # Set ALL, HC, and AAM
        b_list = list(np.array(self.d[list(self.d.keys())[0]]))
        self.ALL = list(np.array(self.d['i']))
        self.HC = [self.ALL[i] for i, j in enumerate(b_list) if j%2==0]
        self.AAM = [self.ALL[i] for i, j in enumerate(b_list) if j%2==1]
        # Set Sex
        sex = list(np.array(self.d['sex']))
        self.SEX = ['Male' if i==0 else 'Female' for i in sex]
        # Set Site
        sites = list(np.array(self.d['site']))
        center = {0: 'Paris', 1: 'Nottingham', 2:'Mannheim', 3:'London',
                  4: 'Hamburg', 5: 'Dublin', 6:'Dresden', 7:'Berlin'}
        self.SITE = [center[i] for i in sites]
        # Set Class
        target = list(np.array(self.d[list(self.d.keys())[0]]))
        self.CLASS = ['HC' if i==0 else 'AAM' for i in target]
        return self.ALL, self.SEX, self.SITE, self.CLASS

    def load_INSTRUMENT(self, SESSION, DATA, instrument_file):
        """
        Load the INSTRUMENT file (.csv) using both SESSION (BL, FU1, FU2, FU3)
        and instrument_file name, add the ID columns

        Argument:
                SESSION: only one either BL, or FU1, or FU2, or FU3
                instrument_file: *.csv
        
        Output:
                DF: dataframe of *.csv
        """
        # Set Session, ROI data, and Instrument file
        self.SESSION = SESSION
        self.DATA = DATA
        self.instrument_file = instrument_file
        # Load the instrument file       
        instrument_path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{self.SESSION}"+\
                            f"/psytools/{self.instrument_file}"
        DF = pd.read_csv(instrument_path, low_memory=False)
        # Add the column : ID
        def usercode(x):
            return int(x[:12])
        DF['ID'] = DF['User code'] if self.SESSION=='FU3' \
        else DF['User code'].apply(usercode)
        self.DF = DF
        return self.DF

    def generate_new_instrument(self, save = True):
        """
        Generate the new intrument:
                Rows: Subjects by ID HDF5 & INSTRUMENT
                Cols: ID, ROI Columns, Sex, Site, Class (HC, AAM)
        
        Argument:
                save : if it is true, save the new dataframe
        
        Output:
                NEW_DF: new dataframe of *.csv
        """
        # -------------------------------------- #
        # ID, Sex, Site, Class columns           #
        # -------------------------------------- #
        DF_2 = self.DF.set_index('ID').reindex(self.ALL)
        DF_2['Sex'] = self.SEX
        DF_2['Site'] = self.SITE
        DF_2['Class'] = self.CLASS
        
        # -------------------------------------- #
        # ROI Columns: Psychological profile     #
        # -------------------------------------- #
        self.VARIABLES = []
        
        if self.DATA == 'NEO':
            # Rename the columns
            NEW_DF = DF_2.rename(
                columns={
                    "neur_mean":"Neuroticism mean",
                    "extr_mean":"Extroversion mean",
                    "open_mean":"Openness mean",
                    "agre_mean":"Agreeableness mean",
                    "cons_mean":"Conscientiousness mean",
                }
            )
            # Set the roi variables
            self.Variables = [
                'Neuroticism mean','Extroversion mean','Openness mean',
                'Agreeableness mean','Conscientiousness mean',
                'Sex','Site','Class']
        
        if self.DATA == 'SURPS':
            # Rename the columns
            NEW_DF = DF_2.rename(
                columns={
                    "h_mean":"Hopelessness mean",
                    "as_mean":"Anxiety sensitivity mean",
                    "imp_mean":"Impulsivity mean",
                    "ss_mean":"Sensation seeking mean",
                }
            )
            # Set the roi variables
            self.Variables = [
                'Hopelessness mean','Anxiety sensitivity mean',
                'Impulsivity mean', 'Sensation seeking mean',
                'Sex', 'Site', 'Class']    

        # -------------------------------------- #
        # ROI Columns: Socio-economic profile    #
        # -------------------------------------- #
  
        if self.DATA == 'LEQ':
            # Rename the columns
            LEQ = DF_2.rename(
                columns={
                    # Mean valence of events
                    "family_valence":"Family valence",
                    "accident_valence":"Accident valence",
                    "sexuality_valence":"Sexuality valence",
                    "autonomy_valence":"Autonomy valence",
                    "devience_valence":"Devience valence",
                    "relocation_valence":"Relocation valence",
                    "distress_valence":"Distress valence",
                    "noscale_valence":"Noscale valence",
                    "overall_valence":"Overall valence",
                    # Mean frequency since last IMAGEN assessment
                    "family_ever_meanfreq":"Family ever meanfrequency",
                    "accident_ever_meanfreq":"Accident ever meanfrequency",
                    "sexuality_ever_meanfreq":"Sexuality ever meanfrequency",
                    "autonomy_ever_meanfreq":"Autonomy ever meanfrequency",
                    "devience_ever_meanfreq":"Devience ever meanfrequency",
                    "relocation_ever_meanfreq":"Relocation ever meanfrequency",
                    "distress_ever_meanfreq":"Distress ever meanfrequency",
                    "noscale_ever_meanfreq":"Noscale ever meanfrequency",
                    "overall_ever_meanfreq":"Overall ever meanfrequency",
                    # Frequency since last IMAGEN
                    "family_ever_freq":"Family ever frequency",
                    "accident_ever_freq":"Accident ever frequency",
                    "sexuality_ever_freq":"Sexuality ever frequency",
                    "autonomy_ever_freq":"Autonomy ever frequency",
                    "devience_ever_freq":"Devience ever frequency",
                    "relocation_ever_freq":"Relocation ever frequency",
                    "distress_ever_freq":"Distress ever frequency",
                    "noscale_ever_freq":"Noscale ever frequency",
                    "overall_ever_freq":"Overall ever frequency",
                }
            )

            if self.SESSION == 'BL':
# ---------------------- BL needed new columns Year --------------------------- #
                NEW_DF = LEQ
#                 NEW_DF = LEQ.rename(
#                     columns={
                        
#                     }
#                 )
                self.Variables = [
                    'Family valence','Accident valence','Sexuality valence',
                    'Autonomy valence','Devience valence','Relocation valence',
                    'Distress valence','Noscale valence','Overall valence',
                    'Family ever meanfrequency','Accident ever meanfrequency',
                    'Sexuality ever meanfrequency','Autonomy ever meanfrequency',
                    'Devience ever meanfrequency','Relocation ever meanfrequency',
                    'Distress ever meanfrequency','Noscale ever meanfrequency',
                    'Overall ever meanfrequency','Family ever frequency',
                    'Accident ever frequency','Sexuality ever frequency',
                    'Devience ever frequency','Relocation ever frequency',
                    'Distress ever frequency','Noscale ever frequency',
                    'Overall ever frequency','Sex', 'Site', 'Class']  
            else:
                NEW_DF = LEQ.rename(
                    columns={
                        # Mean age at occurance
                        "family_age_mean":"Family age mean",
                        "accident_age_mean":"Accident age mean",
                        "sexuality_age_mean":"Sexuality age mean",
                        "autonomy_age_mean":"Autonomy age mean",
                        "devience_age_mean":"Devience age mean",
                        "relocation_age_mean":"Relocation age mean",
                        "distress_age_mean":"Distress age mean",
                        "noscale_age_mean":"Noscale age mean",
                        "overall_age_mean":"Overall age mean",
                    }
                )
                # Set the roi variables
                self.Variables = [
                    'Family valence','Accident valence','Sexuality valence',
                    'Autonomy valence','Devience valence','Relocation valence',
                    'Distress valence','Noscale valence','Overall valence',
                    'Family age mean','Accident age mean','Sexuality age mean',
                    'Autonomy age mean','Devience age mean','Relocation age mean',
                    'Distress age mean','Noscale age mean','Overall age mean',
                    'Family ever meanfrequency','Accident ever meanfrequency',
                    'Sexuality ever meanfrequency','Autonomy ever meanfrequency',
                    'Devience ever meanfrequency','Relocation ever meanfrequency',
                    'Distress ever meanfrequency','Noscale ever meanfrequency',
                    'Overall ever meanfrequency','Family ever frequency',
                    'Accident ever frequency','Sexuality ever frequency',
                    'Devience ever frequency','Relocation ever frequency',
                    'Distress ever frequency','Noscale ever frequency',
                    'Overall ever frequency','Sex', 'Site', 'Class']    
 
        if self.DATA == 'CTQ':
            emot_abuse = ['CTQ_3', 'CTQ_8', 'CTQ_14', 'CTQ_18', 'CTQ_25']
            phys_abuse = ['CTQ_9', 'CTQ_11', 'CTQ_12', 'CTQ_15', 'CTQ_17']
            sexual_abuse = ['CTQ_20', 'CTQ_21', 'CTQ_23', 'CTQ_24', 'CTQ_27']
            emot_neglect = ['CTQ_5', 'CTQ_7', 'CTQ_13', 'CTQ_19', 'CTQ_28']
            phys_neglect = ['CTQ_1', 'CTQ_2', 'CTQ_4', 'CTQ_6', 'CTQ_26']
            denial = ['CTQ_10', 'CTQ_16', 'CTQ_22']
            
            # Generate the columns
            DF_2['Emotional abuse sum'] = DF_2[emot_abuse].sum(axis=1,
                                                               skipna=False)
            DF_2['Physical abuse sum'] = DF_2[phys_abuse].sum(axis=1,
                                                              skipna=False)
            DF_2['Sexsual abuse sum'] = DF_2[sexual_abuse].sum(axis=1,
                                                               skipna=False)
            DF_2['Emotional neglect sum'] = DF_2[emot_neglect].sum(axis=1,
                                                                   skipna=False)
            DF_2['Physical neglect sum'] = DF_2[phys_neglect].sum(axis=1,
                                                                  skipna=False)
            DF_2['Denial sum'] = DF_2[denial].sum(axis=1, skipna=False)
            NEW_DF = DF_2
            # Set the roi variables
            self.Variables = [
                'Emotional abuse sum','Physical abuse sum','Sexsual abuse sum',
                'Emotional neglect sum','Physical neglect sum','Denial sum',
                'Sex','Site','Class'
            ]

        if self.DATA == 'PBQ':
# ------------------- Categorical data selction needed ------------------------ #
            # Rename the columns
            NEW_DF = DF_2
            # Set the roi variables
            self.Variables = [
                'Sex', 'Site', 'Class'
            ]
        
        if self.DATA == 'BMI':
# ------------------------ BMI calculation needed ----------------------------- #
            # Rename the columns
            NEW_DF = DF_2
            # Set the roi variables
            self.Variables = [
                'Sex', 'Site', 'Class'
            ]
        
        if self.DATA == 'GEN':
            # Rename the columns
# ------------------ Categorical data selection needed ------------------------ #
            NEW_DF = DF_2
            # Set the roi variables
            self.Variables = [
                'Sex', 'Site', 'Class'
            ]
        
        if self.DATA == 'CTS':
            # Rename the columns
            NEW_DF = DF_2.rename(
                columns={
                    "cts_assault":"Assault mean",
                    "cts_injury":"Injury mean",
                    "cts_negotiation":"Negotiation mean",
                    "cts_psychological_aggression":"Psychological aggression mean",
                    "cts_sexual_coercion":"Sexual coercion mean",
                }
            )
            # Set the roi variables
            self.Variables = [
                'Assault mean','Injury mean','Negotiation mean',
                'Psychological aggression mean','Sexual coercion mean',
                'Sex','Site','Class']

        # -------------------------------------- #
        # ROI Columns: Other co-morbidities      #
        # -------------------------------------- #

        if self.DATA == 'FTND':
            # Generate the columns
            def test(x):
                if (7<=x and x <=10): return 'highly dependent'
                elif (4<=x and x <=6): return 'moderately dependent'
                elif (x<4): return 'less dependent'
                else: return np.NaN
            DF_2["Nicotine dependence"] = DF_2['ftnd_sum'].apply(test)
            NEW_DF = DF_2
            # Set the roi variables
            self.Variables = [
                'Nicotine dependence','Sex','Site','Class'
            ]       
        
#         def DAST_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['ftnd_sum', 'sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'DAST' == self.DATA: # 'DAST'
#                 self.VARIABLES, self.NEW_DF2 = DAST_SESSION(self.SESSION)


#         def SCID_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'SCID' == self.DATA: # 'SCID'
#                 self.VARIABLES, self.NEW_DF2 = SCID_SESSION(self.SESSION)


#         def DMQ_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['ftnd_sum', 'sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'DMQ' == self.DATA: # 'DMQ'
#                 self.VARIABLES, self.NEW_DF2 = DMQ_SESSION(self.SESSION)


#         def BSI_SESSION(SESSION):
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
#                 DF_BSI = self.NEW_DF
#                 DF_BSI['somatization_mean'] = DF_CTQ[Somatization_labels].mean(axis=1,
#                                                                                skipna=False)
#                 DF_BSI['obsession_compulsion_mean'] = DF_CTQ[Somatization_labels].mean(axis=1,
#                                                                                skipna=False)             
#         elif 'BSI' == self.DATA: # 'BSI'
#             self.VARIABLES, self.NEW_DF2 = BSI_SESSION(self.SESSION)             
#                 Variables = Somatization_labels + Obsession_compulsion_labels \
#                         + Interpersonal_sensitivity_labels + Depression_labels \
#                         + Anxiety_labels + Hostility_labels \
#                         + Phobic_anxiety_labels + Paranoid_ideation_labels \
#                         + Psychoticism_labels + ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF


#         def AUDIT_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#         elif 'AUDIT' == self.DATA: # 'AUDIT'
#             self.VARIABLES, self.NEW_DF2 = AUDIT_SESSION(self.SESSION)


#         def MAST_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#         elif 'MAST' == self.DATA: # 'MAST'
#             self.VARIABLES, self.NEW_DF2 = MAST_SESSION(self.SESSION)        
        
#################################################################################
        self.NEW_DF = NEW_DF[self.Variables]
        if save == True:
            phenotype = self.h5py_file.replace(".h5", "")
            save_absolute_path = f"{self.DATA_DIR}/Instrument/"+\
                                 f"{phenotype}_{self.SESSION}_{self.DATA}.csv"
            # set the save option
            if not os.path.isdir(os.path.dirname(save_absolute_path)):
                os.makedirs(os.path.dirname(save_absolute_path))
            self.NEW_DF.to_csv(save_absolute_path)
            print(f"{'-'*83} \n{self.__str__()} \n{'-'*83}")
            print(f"{self.NEW_DF.info(), self.NEW_DF.describe()}")
        return self.NEW_DF

    def __str__(self):
        return "Step 1. load the phenotype: " \
               + str(self.h5py_file.replace(".h5", "")) \
               + "\n        Class = " + str(list(self.d.keys())[0]) \
               + ", n = " + str(len(self.ALL)) + " (HC = " \
               + str(len(self.HC)) + ", AAM = " + str(len(self.AAM)) +")" \
               + "\n" + "Step 2. load the instrument dataset: " \
               + str(self.instrument_file.replace(".csv",'')) \
               + "\n" + "Step 3. generate the " + str(self.SESSION) +"_" \
               + str(self.h5py_file.replace(".h5", "")) \
               + str(self.instrument_file.replace("IMAGEN-IMGN", "")) \
               + "\n        The dataset contains " + str(self.NEW_DF.shape[0]) \
               + " samples and " + str(self.NEW_DF.shape[1]) + " columns" \
               + "\n" + "Step 4. select " + str(self.SESSION) +"_" \
               + str(self.h5py_file.replace(".h5", "")) \
               + str(self.instrument_file.replace("IMAGEN-IMGN", "")) \
               + "\n        The dataset contains " + str(self.NEW_DF.shape[0]) \
               + " samples and " + str(self.NEW_DF.shape[1]) + " columns" \
               + "\n        Variables = " + str(self.VARIABLES)
    
    def to_instrument(self, h5py_file, SESSION,
                      instrument_file, DATA, save = True):
        self.load_HDF5(h5py_file)
        self.load_INSTRUMENT(SESSION, DATA, instrument_file)
        self.generate_new_instrument(save)
        return self.NEW_DF