from urllib.request import urlopen
import re
import os
from datetime import date, timedelta
import shutil

def match_Val_txt(s):
    return re.search(r"Val_(.*?).txt", s)

def download(url, colunas, out_folder="FTP_Data", is_csv=True):
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    lines = urlopen(url).readlines()  
    txts_list = list(map(lambda x: match_Val_txt(x).group(0), list(filter(lambda l: match_Val_txt(l) is not None , list(map(str,lines))))))

    for tl in txts_list:
        tlw = out_folder+'/'+tl
        urltl = url + tl
        
        if is_csv:
            tlw = tlw.replace('.txt', '.csv')

        if os.path.isfile(tlw):
            f = open(tlw, 'at')
        else:
            f = open(tlw, 'wt')

        web_page = str(urlopen(urltl).read())
        correct_web_pag = web_page.replace('\\n','\n').replace('b','').replace("\'","")
        with_columns_web_pag = colunas + '\n' + correct_web_pag
        f.write(with_columns_web_pag)

if __name__ == "__main__":
    url = "http://200.238.105.69/previsao_municipios/Diaria/"
    run_day = date.today()
    out_folder = "APAC_Data"+os.path.sep+"Valid_"+str(run_day).replace('-','')
    colunas = "ano;mes;dia;hora(utc);temperatura;umidade;pontodeorvalho;pressão;VVel;Vdir;Irradiação;chuva"
    download(url, colunas, out_folder, is_csv=True)
    shutil.make_archive(out_folder, 'zip', out_folder)
    shutil.rmtree(out_folder)
