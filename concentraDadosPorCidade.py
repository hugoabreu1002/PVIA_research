import os
import glob
import pandas as pd
import argparse

def main(args):

    os.chdir(args.inputFolder)
    all_filenames = [i for i in glob.glob('INMET*{0}*.csv'.format(args.city.upper()))]

    #combine all files in the list
    colunas = ["data", "hora", "preciptacao_total_mm", "pressao_atm_mB", "pressao_atm_max_mB",\
            "pressao_atm_min_mB","radiacao_global_wpm2", "temp_ar_bulbo_seco_c", "temp_ponto_orvalho_c","temp_max_c",\
            "temp_min_c","temp_orvalho_max_c", "temp_orvalor_min_c", "umidade_relativa_max_prcnt",\
            "umidade_relativa_min_prcnt","umidade_relativa_prcnt", "vento_dir_gr", "vento_rajada_max_mps",\
            "vento_velocidade_mps"]

    combined_csv = pd.read_csv(all_filenames[0], sep=';', encoding = "ISO-8859-1")
    try:
        combined_csv.drop(["Unnamed: 19"], inplace=True, axis=1)
    except Exception:
        pass
    combined_csv.columns = colunas
    combined_csv["ano"] = combined_csv["data"].apply(lambda x: str(x).replace("/","-")[0:4])
    combined_csv["mes"] = combined_csv["data"].apply(lambda x: str(x).replace("/","-")[5:7])
    combined_csv["dia"] = combined_csv["data"].apply(lambda x: str(x).replace("/","-")[-2:])
    combined_csv["hora"] = combined_csv["hora"].apply(lambda x: int(str(x)[0:2]))

    for f in all_filenames[1:]:
        print(f)
        
        df = pd.read_csv(f, sep=';', encoding = "ISO-8859-1")
        try:
            df.drop(["Unnamed: 19"], inplace=True, axis=1)
        except Exception:
            pass
        df.columns = colunas
        df["data"] = df["data"].apply(lambda x: str(x).replace("/","-"))
        df["ano"] = df["data"].apply(lambda x: str(x).replace("/","-")[0:4])
        df["mes"] = df["data"].apply(lambda x: str(x).replace("/","-")[5:7])
        df["dia"] = df["data"].apply(lambda x: str(x).replace("/","-")[-2:])
        df["hora"] = df["hora"].apply(lambda x: int(str(x)[0:2]))
        
        print(df.columns)
        combined_csv = combined_csv.append([df])
        
    #export to csv
    if os.path.isfile("historical_data_{0}.csv".format(args.city)):
        number = len(list(filter(lambda x: "historical" in x, os.listdir("./")))) + 1
        combined_csv.to_csv("historical_data_{0}_{1}.csv".format(args.city, number), index=False, encoding="ISO-8859-1")
    else:
        combined_csv.to_csv("historical_data_{0}.csv".format(args.city), index=False, encoding="ISO-8859-1")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--inputFolder", type=str, help="Folder with .CSV data")
    parser.add_argument("-C", "--city", type=str, help="City to concatenate data.")
    args = parser.parse_args()
    main(args)