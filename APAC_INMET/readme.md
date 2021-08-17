# Hourly forecast Solar Powerplants

Utilizando dados da APAC, INMET e ONS

# APAC

## Download de Previsões 

O Download é feito por colect_from_apac_ftp.py. Esse script deve ser executado diariamente. Os dados são salvos na pasta FTP_Data

## Concatenção dos dados para 1 mês

A concatenação é feita pelo scrip concat_apac.py.

Examplo:
> python concat_apac.py -ad FTP_Data/Maio_WRF_APAC

### Caracterização 

#### colunas
> "ano;mes;dia;hora(utc);temperatura;umidade;pontodeorvalho;pressão;VVel;Vdir;Irradiação;chuva"

#### Os dados Iniciados com Sol_ foram definidos para os pontos das Usinas.

|Nome_Usina |Cidade |Estado |Latitude   |Longitude  |PotInst (MW)   |
|-----------|:-----:|:-----:|:---------:|:---------:|:-------------:|
|fontes_solar_1 |Tacaratu   |PE |-9,07  |-38,14 |4  |
|fontes_solar_2 |Tacaratu   |PE |-9,07  |-38,14 |4  |
|rio_alto   |Coremas    |PB |-6,96  |-37,99 |56 |
|calcario   |Quixere    |CE |-5,03  |-37,79 |135    |   
|sol_do_futuro  |Aquiraz    |CE |-3,92  |-38,4  |81,48  |
|bjl_solar  |Bom Jesus da Lapa  |BA |-13,31 |-43,34 |42 |
|bom_jesus  |Bom Jesus da Lapa  |BA |-13,29 |-43,33 |68 |
|horizonte_mp   |Tabocas do Brejo Velho  |BA |-12,6  |-44,08 |76 |
|ituvera    |Tabocas    |BA |-12,6  |-44,11 |190,47 |
|juazeiro_solar |Juzaeiro   |BA |-9,48  |-40,47 |120    |
|lapa|  Bom Jesus da Lapa   |BA |-13,3  |-43,33 |64,28  |
|sao_pedro  |Bom Jesus da Lapa  |BA |-13,31 |-43,35 |54,28  |
|nova_olinda    |Sao Joao do Piaui  |PI |-8,21  |-42,54 |229,17 |
|assu_5 |Assu   |RN |-5,58  |-37,01 |30 |
|floresta   |Mossoro    |RN |-5,15   |-37,35  |100 |
|sao_goncalo    |São Gonçalo do Gurguéia    |PI |-10,09 |-45,26 |475    |
|sertao_solar_barreiras |Barreiras  |BA |-12,14 |-45,01 |93,1   |

#### Todos os outros pontos são Pernambuco

 * Arco Verde
 * Cabrobo
 * Caruaru
 * Floresta
 * Garanhuns
 * Ibimirim
 * Ouricuri
 * Palmares
 * Petrolina (Tacaratu)
 * Recife
 * Salgueiro
 * Surubim
 * Serra_Talhada

#### Val_t00z... Quando for a rodada das 00UTC com ate 72h de previsao

#### Val_t12z... quando for a previsao da tarde. Com 48h de previsao

# Windows Scheduler

https://www.youtube.com/watch?v=n2Cr_YRQk7o

# INMET

Downloaded from

> http://www.inmet.gov.br/portal/index.php?r=estacoes/estacoesAutomaticas

# 00 e 12

00 UTC = 21 hora local 

Então quando chega as 00 UTC lança uma análise daquela hora. Análise é o conjunto de dados meteorológicos globais. Dessa análise é feito uma previsão global para X dias. O modelo regional, que é o WRF, esse que tas pegando os dados, fz uma previsão desses dados dos modelos globais de 00 UTC. A APAC faz previsão de 3 dias entre as 21h D+0 até 21h D+3 (72h).

De 09hora local (12UTC), esse ciclo se repete. 
A APAC pega os dados de análise, porém só roda 48h 09h D+0 a 09 D+2.

Resumindo, a previsão das 12 UTC faz uma "atualização" das previsões das 09 D+0 as 09 D+2 do modelo das 00UTC.

Daí entraria a análise que pode ser feita, essa atualização melhora ou piora a previsão anterior, e em quanto %?
