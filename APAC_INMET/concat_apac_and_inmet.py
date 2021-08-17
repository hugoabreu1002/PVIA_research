import numpy as np
import pandas as pd
import os
import sys
import argparse
import re
import csv

class CsvManipulator():

    def append_csv_files(self, to_be_appended, to_append, delimiter, encoding=None):
        #print(to_be_appended, to_append)
        df_tobe = pd.read_csv(to_be_appended, sep=delimiter, encoding = encoding)
        df_toap = pd.read_csv(to_append, sep=delimiter, encoding = encoding, names=df_tobe.columns)
        if "ano" in df_toap.columns:
            df_toap = df_toap[df_toap.ano !="ano"]
        else:
            df_toap = df_toap[df_toap.hora !="hora"]
        #print(df_tobe.columns == df_toap.columns)
        df_tobe = df_tobe.append(df_toap, sort=False)
        df_tobe.to_csv(to_be_appended, sep = delimiter, encoding = encoding, index=False)

    def addkeycolumns(self, filepath, newfilepath, newcolumns, values, delimiter, encoding=None):
        df = pd.read_csv(filepath, sep=delimiter, encoding = encoding)
        for e, newcolumn in enumerate(newcolumns):
            df.insert(loc=0, column = newcolumn, value=[values[e]]*len(df))
        df.to_csv(newfilepath, sep=delimiter, encoding = encoding, index=False)

    def create_and_add_columns(self, csv_file, columns):
        with open(csv_file, 'w') as fcv:
            fcv.write(columns)
            fcv.write('\r') 

        fcv.close()

class DataManipulator(CsvManipulator):

    def __init__(self, Dict_map_cidades_apac_inmet):
        super().__init__()
        self.Dict_map_cidades_apac_inmet = Dict_map_cidades_apac_inmet

    def parser_apac_filename(self, File):
        if "Sol_" in File:
            CutfileSol = File[File.find("Sol_")+len("Sol_"):]
            return CutfileSol[:CutfileSol.rfind("_")]
        elif "t00z" in File and "Sol_" not in File:
            CutfileSol = File[File.find("t00z")+len("t00z"):]
            return CutfileSol[:CutfileSol.rfind("2020")-1]
        else:
            CutfileSol = File[File.find("Val_")+len("Val_"):]
            return CutfileSol[:CutfileSol.rfind("2020")-1]


    def run_to_append_apac_plants(self, Apac_month_folder, columns):
        for plants in self.Dict_map_cidades_apac_inmet:
            self.create_and_add_columns(Apac_month_folder+os.path.sep+plants+'1'+'.csv',columns) 
            self.create_and_add_columns(Apac_month_folder+os.path.sep+plants+'2'+'.csv',columns)
            self.create_and_add_columns(Apac_month_folder+os.path.sep+plants+'3'+'.csv',columns)

        for Valid_Dayh in list(filter(lambda x: not '.csv' in x, list(os.listdir(Apac_month_folder)))):
            Valid_Day_folder = Apac_month_folder+os.path.sep+Valid_Dayh
            #print(Valid_Day_folder)
            Files_days_month = list(set(list(map(lambda x: (int(x[-6:-4]), x[-9:-6]), list(filter(lambda x: ".txt" in x or ".csv" in x, os.listdir(Valid_Day_folder)))))))
            Files_days = list(map(lambda t: t[0] if t[1] == self.english_moth else t[0]+31, Files_days_month))
            #print(Files_days_month, sorted(Files_days))
            for File in list(filter(lambda x: ".txt" in x or ".csv" in x,sorted(os.listdir(Valid_Day_folder)))):
                File_day = int(File[-6:-4])
                File_month = File[-9:-6]
                if File_month != self.english_moth:
                    File_day += 31
                parsed_apac_filename = self.parser_apac_filename(File)
                if parsed_apac_filename in self.Dict_map_cidades_apac_inmet:
                    count = sorted(Files_days).index(File_day) + 1
                    csv_1 = Apac_month_folder+os.path.sep+parsed_apac_filename+str(count)+'.csv'
                    csv_2 = Valid_Day_folder+os.path.sep+File
                    self.append_csv_files(csv_1, csv_2,';', "ISO-8859-1")               
                    
    def append_APAC(self, Apac_month_folder, columns):
        pt_month = Apac_month_folder[Apac_month_folder.find("_Data\\")+len("_Data\\"):Apac_month_folder.find("_WRF")]
        self.english_moth = {"Maio":"MAY", "Abril":"APR", "Junho":"JUN"}[pt_month]
        self.run_to_append_apac_plants(Apac_month_folder, columns)
        columns = "planta;diasprev;"+columns
        month = Apac_month_folder.split(os.path.sep)[-1]
        final_filepath = Apac_month_folder+os.path.sep+month+'.csv'
        self.create_and_add_columns(final_filepath, columns)

        for plant_fday in list(filter(lambda x: '.csv' in x and not month in x, os.listdir(Apac_month_folder))):
            print(plant_fday)
            planta = plant_fday[:-5]
            diasprev = plant_fday[-5]
            to_append_filepath = Apac_month_folder+os.path.sep+plant_fday
            tempfilepath = to_append_filepath.replace('.csv','_temp.csv')
            self.addkeycolumns(to_append_filepath, tempfilepath, ["diasprev","planta"], [diasprev, planta], ';',  "ISO-8859-1")
            os.replace(tempfilepath, to_append_filepath)
            self.append_csv_files(final_filepath, to_append_filepath, ';', "ISO-8859-1")
            os.remove(to_append_filepath)


    def append_INMET(self, INMET_month_folder, columns):

        columns = "planta,"+columns

        month = INMET_month_folder.split(os.path.sep)[-1]
        final_filepath = INMET_month_folder+os.path.sep+month+'.csv'
        self.create_and_add_columns(final_filepath, columns)

        for plant in list(filter(lambda x: '.csv' in x and not month in x, os.listdir(INMET_month_folder))):
            print(plant)
            to_append_filepath = INMET_month_folder+os.path.sep+plant
            tempfilepath = to_append_filepath.replace('.csv','_temp.csv')
            self.addkeycolumns(to_append_filepath, tempfilepath, ["planta"], [plant[:-4]], ',')
            self.append_csv_files(final_filepath, tempfilepath,',')
            os.remove(tempfilepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ad", "--apac_directory", type=str, help="Diretório para o mês de dados da APAC que deseja-se concatenar")
    parser.add_argument("-id", "--inmet_directory", type=str, help="Diretório para o mês de dados do INMET que deseja-se concatenar")

    args = parser.parse_args()

    Dict_map_cidades_apac_inmet = ["Coremas","Aquiraz","Assu","Barreiras","BJLapaI","BJLapaII","BJLapaIII","BJLapaIV","Juazeiro",
        "Mossoro","Quixere","SGGurgeia","SJPiaui","TabocasI","TabocasII","Tacaratu", "Aguas_Belas", "Arco_Verde", "Cabrobo", 
        "Floresta", "Garanhuns", "Ibimirim", "Ouricuri", "Palmares", "Petrolina", "Recife", "Salgueiro", "Surubim", "Serra_Talhada"]

    dataManipulator = DataManipulator(Dict_map_cidades_apac_inmet)

    if args.apac_directory:
        Apac_month_folder = args.apac_directory
        if Apac_month_folder[-1] == '\\':
            Apac_month_folder = Apac_month_folder[:-1]

        columns = "ano;mes;dia;hora(utc);temperatura;umidade;pontodeorvalho;pressao;VVel;Vdir;Irradiacao;chuva"
        dataManipulator.append_APAC(Apac_month_folder, columns)

    elif args.inmet_directory:
        columns = "codigo_estacao,data,hora,temp_inst,temp_max,temp_min,umid_inst,umid_max,umid_min,pto_orvalho_inst,pto_orvalho_max,pto_orvalho_min,pressao,pressao_max,pressao_min,vento_vel,vento_direcao,vento_rajada,radiacao,precipitacao"

        Inmet_month_folder = args.inmet_directory
        if Inmet_month_folder[-1] == '\\':
            Inmet_month_folder = Inmet_month_folder[:-1]

        dataManipulator.append_INMET(Inmet_month_folder, columns)

    else:
        raise ("Por favor, selecione uma opção")