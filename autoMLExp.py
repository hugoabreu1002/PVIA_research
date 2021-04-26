import pickle
from mlopt.TimeSeriesUtils import train_test_split_with_Exog
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import argparse
import tpot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from hpsklearn import HyperoptEstimator
from hpsklearn import any_regressor
from hpsklearn import any_preprocessing
from hyperopt import tpe
import pickle
import autokeras as ak
import os
from matplotlib import pyplot as plt
from tensorflow.random import set_seed
import warnings
from numpy.random import seed
import tensorflow as tf
import numpy as np

OMP_NUM_THREADS=1
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed(1)
set_seed(2)
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

def loadData(csvFile, serie_column='radiacao_global_wpm2',
             exogenous_columns = ['preciptacao_total_mm', 'temp_ar_bulbo_seco_c',
                                  'umidade_relativa_prcnt','vento_velocidade_mps', 'vento_rajada_max_mps']):

    df_inmet = pd.read_csv(csvFile, sep=',', encoding = "ISO-8859-1")

    for c in df_inmet.columns:
        if (c not in ['data', 'hora']) and (df_inmet[c].dtype != "float64"):
            df_inmet[c] = df_inmet[c].apply(lambda x: float(str(x).replace(",","."))).fillna(method='ffill')
    

    ultimoas_horas = 15*24
    posicao_final=len(df_inmet)-1
    posicao_inicial=posicao_final - ultimoas_horas
    exog = df_inmet[exogenous_columns].iloc[posicao_inicial:,:]
    gen = df_inmet[serie_column].iloc[posicao_inicial:].values.reshape(-1,1)
    print("data hora inicial: ", df_inmet.iloc[posicao_inicial,:].data, df_inmet.iloc[posicao_inicial,:].hora,
        "data hora final: ", df_inmet.iloc[posicao_final,:].data,df_inmet.iloc[posicao_final,:].hora)

    return gen, exog, df_inmet

def scaleData(data):
    MaxAbsScaler_data = MaxAbsScaler().fit(data)
    scaled_data = MaxAbsScaler_data.transform(data)

    return scaled_data, MaxAbsScaler_data


def applyTPOT(X_train, y_train, X_test, y_test, SavePath, popSize=20, number_Generations=5, kFolders=5, TPOTSingleMinutes=1, TPOTFullMinutes = 10):
    pipeline_optimizer = tpot.TPOTRegressor(generations=number_Generations, #number of iterations to run the training
                                            population_size=popSize, #number of individuals to train
                                            cv=kFolders, #number of folds in StratifiedKFold
                                            max_eval_time_mins=TPOTSingleMinutes, #time in minutes for each trial
                                            max_time_mins=TPOTFullMinutes,
                                            scoring="neg_mean_absolute_error") #time in minutes for whole optimization
    
    pipeline_optimizer.fit(X_train, y_train) #fit the pipeline optimizer - can take a long time
    print("TPOT - Score: ")
    print(pipeline_optimizer.score(X_test, y_test)) #print scoring for the pipeline
    y_hat = pipeline_optimizer.predict(X_test)
    print("MAE: %.4f" % mean_absolute_error(y_hat, y_test))
    pipeline_optimizer.export(SavePath)
    pickle.dump(pipeline_optimizer, open(SavePath+'.pckl', 'wb'))

    return y_hat

def applyHyperOpt(X_train, y_train, X_test, y_test, SavePath, max_evals=100, trial_timeout=100):
    HyperOptModel = HyperoptEstimator(regressor=any_regressor('reg'),
                              preprocessing=any_preprocessing('pre'),
                              loss_fn=mean_absolute_error,
                              algo=tpe.suggest,
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
        project_name="./AutoMLResults/keras_auto_model"
    )
    print(" X_train shape: {0}\n y_train shape: {1}\n X_test shape: {2}\n y_test shape: {3}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    AKRegressor.fit(x=X_train, y=y_train[:,0],epochs=epochs,verbose=1, batch_size=1, shuffle=False, use_multiprocessing=True)
    AKRegressor.export_model(SavePath)
    y_hat = AKRegressor.predict(X_test)
    print("AUTOKERAS - Score: ")
    print("MAE: %.4f" % mean_absolute_error(y_hat, y_test[:,0]))
        
    return y_hat

def executeForCity(city, plot=True):
    csv_name = list(filter(lambda x: "historical" in x, os.listdir(city)))[0]
    print("City {0}, CSV {1}".format(city, csv_name))
    gen, exog, df_inmet = loadData(city+os.sep+csv_name)
    gen, genscaler = scaleData(gen)
    exog, _ = scaleData(exog)
    X_train, y_train, X_test, y_test = train_test_split_with_Exog(gen[:,0], exog, 24, [80,20])
    #print("TPOT Evaluation...")
    #y_hat_tpot = applyTPOT(X_train, y_train, X_test, y_test, "./AutoMLResults/tpotModel_{0}".format(city), popSize=5, number_Generations=1)
    # print("HYPEROPT Evaluation...")
    # y_hat_hyperopt = applyHyperOpt(X_train, y_train, X_test, y_test, "./AutoMLResults/hyperoptModel_{0}".format(city), max_evals=10)
    print("AUTOKERAS Evaluation...")
    y_hat_autokeras = applyAutoKeras(X_train, y_train, X_test, y_test, "./AutoMLResults/autokerastModel_{0}".format(city), max_trials=20)

    if plot:
        _, ax = plt.subplots(1,1, figsize=(14,7), dpi=300)
        ticks_X = df_inmet.data.astype('str') + '-' + df_inmet.hora.astype('str')
        len_dt = len(y_test)
        ticks_X = ticks_X[-len_dt:]
        # ax.plot(ticks_X, genscaler.inverse_transform(y_hat_tpot[-len_dt:].reshape(-1, 1)), 'y--', label='TPOT')
        # ax.plot(ticks_X, genscaler.inverse_transform(y_hat_hyperopt[-len_dt:].reshape(-1, 1)), 'r--', label='HYPEROPT')
        ax.plot(ticks_X, genscaler.inverse_transform(y_hat_autokeras[-len_dt:].reshape(-1, 1)), 'b--', label='AUTOKERAS')
        ax.plot(ticks_X, genscaler.inverse_transform(y_test.reshape(-1, 1)), 'k', label='Original')
        plt.xticks(ticks_X[::3], rotation=45, ha='right', fontsize=12)
        ax.grid(axis='x')
        ax.legend(fontsize=12)
        ax.set_ylabel('W/m2', fontsize=12)
        plt.tight_layout()
        plt.savefig('./AutoMLResults/AutoMLS_result.png', dpi=300)

def main(args):
    citiesFolders = args.listaCidades
    if not os.path.isdir("./AutoMLResults"):
        os.mkdir("./AutoMLResults")

    for city in citiesFolders:
        executeForCity(city)
    
    return None

if __name__ == '__main__':
    CLI=argparse.ArgumentParser()
    CLI.add_argument( "-l", "--listaCidades",  # name on the CLI - drop the `--` for positional/required parameters
                     nargs="*",  # 0 or more values expected => creates a list
                     type=str,
                     default=["maceio", "floripa", "bomJesusDaLapaBA"],  # default if nothing is provided
                     )
    
    args = CLI.parse_args()

    main(args)