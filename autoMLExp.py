import pickle
from tensorflow.python.training.tracking.util import add_variable
from mlopt.TimeSeriesUtils import train_test_split_with_Exog
from mlopt.TimeSeriesUtils import train_test_split as train_test_split_noExog
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import argparse
import tpot
from sklearn.metrics import mean_absolute_error, mean_squared_error
from hpsklearn import HyperoptEstimator
from hpsklearn import any_regressor
from hpsklearn import any_preprocessing
import pickle
import autokeras as ak
import os
from matplotlib import pyplot as plt
import warnings
from numpy.random import seed
import tensorflow as tf
import numpy as np
from mlopt.ACOLSTM import ACOLSTM

# OMP_NUM_THREADS=1
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     pass

def loadData(csvFile, serie_column='radiacao_global_wpm2',
             exogenous_columns = ['preciptacao_total_mm', 'temp_ar_bulbo_seco_c',
                                  'umidade_relativa_prcnt','vento_velocidade_mps', 'vento_rajada_max_mps']):

    df_inmet = pd.read_csv(csvFile, sep=',', encoding = "ISO-8859-1")
    
    for c in exogenous_columns:
        if df_inmet[c].isnull().sum(axis = 0) > len(df_inmet)*0.5:
            df_inmet.drop(c,inplace=True, axis=1)
            if c in exogenous_columns:
                exogenous_columns.remove(c)
        elif (c not in ['data', 'hora']) and (df_inmet[c].dtype != "float64"):
            df_inmet[c] = df_inmet[c].apply(lambda x: float(str(x).replace(",","."))).fillna(method='ffill')
        else:
            df_inmet[c] = df_inmet[c].fillna(method='ffill').fillna(0)

    if df_inmet[serie_column].dtypes == "object":
        df_inmet[serie_column] = df_inmet[serie_column].apply(lambda x: float(str(x).replace(",","."))).fillna(0)

    print(df_inmet.describe())
    posicao_final=15*24
    posicao_inicial=0
    exog = df_inmet[exogenous_columns].iloc[posicao_inicial:posicao_final,:].values
    gen = df_inmet[serie_column].iloc[posicao_inicial:posicao_final].values.reshape(-1,1)
    print("data hora inicial: ", df_inmet.iloc[posicao_inicial,:].data, df_inmet.iloc[posicao_inicial,:].hora,
        "data hora final: ", df_inmet.iloc[posicao_final,:].data,df_inmet.iloc[posicao_final,:].hora)

    if np.isnan(np.sum(gen)):
        raise("Gen still has nan")

    for i in range(exog.shape[1]):
        ex_var = exog[i]
        if np.isnan(np.sum(ex_var)):
            raise("exog still has nan in column {0}".format(i))
    
    return gen, exog, df_inmet

def scaleData(data):
    maxAbsScaler_data = MaxAbsScaler().fit(data)
    scaled_data = maxAbsScaler_data.transform(data)

    return scaled_data, maxAbsScaler_data


def applyTPOT(X_train, y_train, X_test, y_test, SavePath, popSize=20, number_Generations=5, kFolders=5, TPOTSingleMinutes=1, TPOTFullMinutes = 10):
    pipeline_optimizer = tpot.TPOTRegressor(generations=number_Generations, #number of iterations to run the training
                                            population_size=popSize, #number of individuals to train
                                            cv=kFolders, #number of folds in StratifiedKFold
                                            max_eval_time_mins=TPOTSingleMinutes, #time in minutes for each trial
                                            max_time_mins=TPOTFullMinutes,
                                            scoring="neg_mean_absolute_error") #time in minutes for whole optimization
    
    pipeline_optimizer.fit(X_train, y_train) #fit the pipeline optimizer - can take a long time
    print("TPOT - Score: ")
    print(-pipeline_optimizer.score(X_test, y_test)) #print scoring for the pipeline
    y_hat = pipeline_optimizer.predict(X_test)
    print("MAE: %.4f" % mean_absolute_error(y_hat, y_test))
    pipeline_optimizer.export(SavePath)

    return y_hat

def applyHyperOpt(X_train, y_train, X_test, y_test, SavePath, max_evals=100, trial_timeout=100):
    HyperOptModel = HyperoptEstimator(regressor=any_regressor('reg'),
                              preprocessing=any_preprocessing('pre'),
                              loss_fn=mean_squared_error,
                              max_evals=max_evals,
                              trial_timeout=trial_timeout)
    # perform the search
    HyperOptModel.fit(X_train, y_train)
    # summarize performance
    mae = HyperOptModel.score(X_test, y_test)
    y_hat = HyperOptModel.predict(X_test)
    print("HYPEROPT - Score: ")
    print("MAE: %.4f" % mae)
    # summarize the best model
    print(HyperOptModel.best_model())
    pickle.dump(HyperOptModel, open(SavePath+".pckl", 'wb'))

    return y_hat
    
def applyAutoKeras(X_train, y_train, X_test, y_test, SavePath, max_trials=100, epochs=300):
    
    input_node = ak.StructuredDataInput()
    output_node = ak.DenseBlock()(input_node)
    #output_node = ak.ConvBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)
    AKRegressor = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        max_trials=max_trials,
        overwrite=True,
        tuner="bayesian",
        project_name=SavePath+"/keras_auto_model"
    )
    print(" X_train shape: {0}\n y_train shape: {1}\n X_test shape: {2}\n y_test shape: {3}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    AKRegressor.fit(x=X_train, y=y_train[:,0],epochs=epochs,verbose=1, batch_size=int(X_train.shape[0]/10), shuffle=False, use_multiprocessing=True)
    AKRegressor.export_model()
    y_hat = AKRegressor.predict(X_test)
    print("AUTOKERAS - Score: ")
    print("MAE: %.4f" % mean_absolute_error(y_hat, y_test[:,0]))
        
    return y_hat

def applyACOLSTM(X_train, y_train, X_test, y_test, SavePath, antNumber=10, antTours=10):
    options_ACO={'antNumber':antNumber, 'antTours':antTours, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1}
    lstmOptimizer = ACOLSTM(X_train, y_train, X_test, y_test, 1 ,options_ACO=options_ACO, verbose=True)
    final_model, y_hat = lstmOptimizer.optimize(Layers_Qtd=[[40, 50, 60, 70], [20, 25, 30], [5, 10, 15]], epochs=[100, 200, 300])
    final_model.save(SavePath)

    print("ACOLSTM - Score: ")
    print("MAE: %.4f" % mean_absolute_error(y_hat, y_test[:,0]))

    return y_hat

def saveResultFigure(df_inmet, genscaler, y_test, y_hats, labels, city_save_path):
    logResults = ""
    logResults += "Scores" + "\n"
    print("Scores")
    
    _, ax = plt.subplots(1,1, figsize=(14,7), dpi=300)
    ticks_X = df_inmet.data.astype('str') + '-' + df_inmet.hora.astype('str')
    len_dt = len(y_test)
    ticks_X = ticks_X[-len_dt:].values
    ax.plot(ticks_X, genscaler.inverse_transform(y_test.reshape(-1, 1)), 'k', label='Original')

    for y_hat, plotlabel in zip(y_hats, labels):
        logResults += "{0} ".format(plotlabel) + "- MAE: %.4f" % mean_absolute_error(y_hat, y_test[:,0]) + "\n"
        trueScale_yhat = genscaler.inverse_transform(y_hat[-len_dt:].reshape(-1, 1))
        ax.plot(ticks_X, trueScale_yhat, label=plotlabel)

    plt.xticks(ticks_X[::3], rotation=45, ha='right', fontsize=12)
    ax.grid(axis='x')
    ax.legend(fontsize=12)
    ax.set_ylabel('W/m2', fontsize=12)
    plt.tight_layout()
    plt.savefig(city_save_path+"/AutoMLS_result.png", dpi=300)
    #plt.show()

    return logResults

def executeForCity(city, citiesRootFolder, city_save_path, plot=True):
    cityDir = "{0}/{1}".format(citiesRootFolder, city)
    csv_name = list(filter(lambda x: "historical" in x, os.listdir(cityDir)))[0]
    outputLog = ""
    outputLog += "City: {0}, CSV: {1}".format(city, csv_name) + "\n"
    print("City {0}, CSV {1}".format(city, csv_name))
    
    gen, exog, df_inmet = loadData(citiesRootFolder+city+os.sep+csv_name)
    gen, genscaler = scaleData(gen)
    print("scaled gen shape", gen.shape)
    exog, exogscaler = scaleData(exog)
    X_train, y_train, X_test, y_test = train_test_split_with_Exog(gen[:,0], exog, 24, [80,20])

    y_hats = []
    labels = []

    try:
        print("TPOT Evaluation...")
        y_hat_tpot = applyTPOT(X_train, y_train, X_test, y_test, city_save_path+"/tpotModel_{0}".format(city), popSize=10, number_Generations=5)
        y_hats.append(y_hat_tpot)
        labels.append("TPOT")
    except Exception:
        pass

    try:
        print("HYPEROPT Evaluation...")
        y_hat_hyperopt = applyHyperOpt(X_train, y_train, X_test, y_test, city_save_path+"/hyperoptModel_{0}".format(city), max_evals=50)
        y_hats.append(y_hat_hyperopt)
        labels.append("HYPEROPT")
    except Exception:
        pass
    
    try:
        print("AUTOKERAS Evaluation...")
        y_hat_autokeras = applyAutoKeras(X_train, y_train, X_test, y_test, city_save_path+"/autokerastModel_{0}".format(city), max_trials=2, epochs=50)
        y_hats.append(y_hat_autokeras)
        labels.append("AUTOKERAS")
    except Exception:
        pass


    X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm = train_test_split_noExog(gen[:,0], 23,
                                                                        tr_vd_ts_percents = [80, 20],
                                                                        print_shapes = True)
    try:
        print("ACOLSTM Evaluation...")
        y_hat_acolstm = applyACOLSTM(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, city_save_path+"/acolstmModel_{0}".format(city), antNumber=10, antTours=4)
        y_hats.append(y_hat_acolstm)
        labels.append("ACOLSTM")
    except Exception:
        pass
    
    if plot:
        outputLog += saveResultFigure(df_inmet, genscaler, y_test, y_hats, labels, city_save_path)

    with open(city_save_path+"/results.txt", "w") as text_file:
        text_file.write(outputLog)


def main(args):
    citiesRootFolder = args.cidadesRootFolder
    citiesFolders = args.listaCidades

    for city in citiesFolders:
        cityResultsPath = "{0}/{1}/AutoMLResults".format(citiesRootFolder, city)
        if not os.path.isdir(cityResultsPath):
            os.mkdir(cityResultsPath)

        executeForCity(city, citiesRootFolder, cityResultsPath, True)
    
    return None

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    CLI=argparse.ArgumentParser()
    CLI.add_argument("-crf", "--cidadesRootFolder", type=str, default="./cidades")
    CLI.add_argument( "-l", "--listaCidades",  # name on the CLI - drop the `--` for positional/required parameters
                     nargs="*",  # 0 or more values expected => creates a list
                     type=str,
                     default=["recife", "natal"],  # default if nothing is provided
                     )
    
    args = CLI.parse_args()

    main(args)