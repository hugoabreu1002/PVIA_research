import pickle
from mlopt.TimeSeriesUtils import train_test_split_with_Exog
from mlopt.TimeSeriesUtils import train_test_split as train_test_split_noExog
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import argparse
import tpot
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from hpsklearn import HyperoptEstimator, any_regressor, any_preprocessing
import pickle
import autokeras as ak
import os
from matplotlib import pyplot as plt
import warnings
from numpy.random import seed
import tensorflow as tf
import numpy as np
from mlopt.ACOLSTM import ACOLSTM, ACOCLSTM
from mlopt.MMFFBleding_Regressor import AGMMFFBleding
import traceback

def loadData(csvFile, serie_column='radiacao_global_wpm2',
             exogenous_columns = ['preciptacao_total_mm', 'temp_ar_bulbo_seco_c',
                                  'umidade_relativa_prcnt','vento_velocidade_mps', 'vento_rajada_max_mps']):

    df_inmet = pd.read_csv(csvFile, sep=',', encoding = "ISO-8859-1")
    print(exogenous_columns)
    for c in exogenous_columns.copy():
        if df_inmet[c].isnull().sum(axis = 0) >= len(df_inmet)*0.5:
            print("Will drop column " + c)
            df_inmet.drop(c,inplace=True, axis=1)
            exogenous_columns.remove(c)
        elif df_inmet[c].dtype != "float64":
            print("Will parser and fill column " + c)
            df_inmet[c] = df_inmet[c].apply(lambda x: float(str(x).replace(",","."))).fillna(method='ffill')
        else:
            print("Will fill column " + c)
            df_inmet[c] = df_inmet[c].fillna(method='ffill').fillna(0)

    if df_inmet[serie_column].dtypes == "object":
        df_inmet[serie_column] = df_inmet[serie_column].apply(lambda x: float(str(x).replace(",","."))).fillna(0)

    print(df_inmet[exogenous_columns+[serie_column]].describe())
    posicao_final=15*24
    posicao_inicial=0
    exog = df_inmet[exogenous_columns].iloc[posicao_inicial:posicao_final,:].values
    gen = df_inmet[serie_column].iloc[posicao_inicial:posicao_final].values.reshape(-1,1)
    print("data hora inicial: ", df_inmet.iloc[posicao_inicial,:].data, df_inmet.iloc[posicao_inicial,:].hora,
        "data hora final: ", df_inmet.iloc[posicao_final,:].data,df_inmet.iloc[posicao_final,:].hora)

    if np.isnan(np.sum(gen)):
        raise("Gen still has nan")

    print(exog)
    for i in range(exog.shape[1]):
        ex_var = exog[:,i]
        if np.isnan(np.sum(ex_var)):
            raise("exog still has nan in column {0}".format(i))
    
    return gen, exog, df_inmet

def scaleData(data):
    maxAbsScaler_data = MaxAbsScaler().fit(data)
    scaled_data = maxAbsScaler_data.transform(data)

    return scaled_data, maxAbsScaler_data


def applyTPOT(X_train, y_train, X_test, y_test, SavePath, popSize=20,
              number_Generations=5, kFolders=5, TPOTSingleMinutes=1,
              TPOTFullMinutes = 10, useSavedModels = True):
    if not useSavedModels:
        pipeline_optimizer = tpot.TPOTRegressor(generations=number_Generations, #number of iterations to run the training
                                                population_size=popSize, #number of individuals to train
                                                cv=kFolders, #number of folds in StratifiedKFold
                                                max_eval_time_mins=TPOTSingleMinutes, #time in minutes for each trial
                                                max_time_mins=TPOTFullMinutes,
                                                scoring="neg_mean_absolute_error") #time in minutes for whole optimization
        
        pipeline_optimizer.fit(X_train, y_train) #fit the pipeline optimizer - can take a long time
        pipeline_optimizer.export(SavePath)
        
    else:
        print("######### PLACE THE EXPORTED PIPELINE CODE HERE ########")
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures

        # Average CV score on the training set was: -0.04784394817861738
        pipeline_optimizer = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            ExtraTreesRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=12, min_samples_split=13, n_estimators=100)
        )

        pipeline_optimizer.fit(X_train, y_train)
        
    print("TPOT - Score: {0}".format(-pipeline_optimizer.score(X_test, y_test)))
    y_hat = pipeline_optimizer.predict(X_test)
    print("MAE: %.4f" % mean_absolute_error(y_test, y_hat))
        
    return y_hat

def applyHyperOpt(X_train, y_train, X_test, y_test, SavePath,
                  max_evals=100, trial_timeout=100, useSavedModels = True):

    if not useSavedModels:
        HyperOptModel = HyperoptEstimator(regressor=any_regressor('reg'),
                                preprocessing=any_preprocessing('pre'),
                                loss_fn=mean_squared_error,
                                max_evals=max_evals,
                                trial_timeout=trial_timeout)
        # perform the search
        HyperOptModel.fit(X_train, y_train)
        pickle.dump(HyperOptModel, open(SavePath+".pckl", 'wb'))
    else:
        HyperOptModel = pickle.load(open(SavePath+".pckl", 'wb'))

    # summarize performance
    score = HyperOptModel.score(X_test, y_test)
    y_hat = HyperOptModel.predict(X_test)
    print("HYPEROPT - Score: ")
    print("MAE: %.4f" % score)
    # summarize the best model
    print(HyperOptModel.best_model())
    
    return y_hat
    
def applyAutoKeras(X_train, y_train, X_test, y_test, SavePath,
                   max_trials=100, epochs=300, useSavedModels = True):

    if not useSavedModels:
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
    else:
        AKRegressor = tf.keras.models.load_model(SavePath+"/keras_auto_model/best_model/saved_model.pb")
        
    y_hat = AKRegressor.predict(X_test)
    print("AUTOKERAS - Score: ")
    print("MAE: %.4f" % mean_absolute_error(y_test[:,0], y_hat))
        
    return y_hat

def applyACOLSTM(X_train, y_train, X_test, y_test, SavePath,
                 Layers_Qtd=[[40, 50, 60, 70], [20, 25, 30], [5, 10, 15]],
                 epochs=[100,200,300],
                 options_ACO={'antNumber':5, 'antTours':5, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1},
                 useSavedModels = True):

    if not useSavedModels:
        lstmOptimizer = ACOLSTM(X_train, y_train, X_test, y_test, 1 ,options_ACO=options_ACO, verbose=True)
        final_model, y_hat = lstmOptimizer.optimize(Layers_Qtd = Layers_Qtd, epochs=epochs)
        final_model.save(SavePath)
    else:
        final_model = tf.keras.models.load_model(SavePath+"/saved_model.pb")
        y_hat = final_model.predict(X_test)
        
    print("ACOLSTM - Score: ")
    print("MAE: %.4f" % mean_absolute_error(y_test[:,0], y_hat[:,0]))

    return y_hat

def applyACOCLSTM(X_train, y_train, X_test, y_test, SavePath,
                  Layers_Qtd=[[50, 30, 20, 10], [20, 15, 10], [10, 20], [5, 10], [2, 4]],
                  ConvKernels=[[8, 12], [4, 6]],
                  epochs=[10],
                  options_ACO={'antNumber':5, 'antTours':5, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1},
                  useSavedModels = True):

    if not useSavedModels:
        clstmOptimizer = ACOCLSTM(X_train, y_train, X_test, y_test, 1 ,options_ACO=options_ACO, verbose=True)
        final_model, y_hat = clstmOptimizer.optimize(Layers_Qtd = Layers_Qtd, ConvKernels = ConvKernels, epochs=epochs)
        final_model.save(SavePath)
    else:
        final_model = tf.keras.models.load_model(SavePath+"/saved_model.pb")
        y_hat = final_model.predict(X_test)
        

    print("ACOLSTM - Score: ")
    print("MAE: %.4f" % mean_absolute_error(y_test[:,0], y_hat[:,0]))

    return y_hat

def applyGAMMFF(X_train, y_train, X_test, y_test, SavePath,
                  epochs=5, size_pop=40, useSavedModels = True):

    agMMGGBlending = AGMMFFBleding(X_train, y_train, X_test, y_test, epochs=epochs, size_pop=size_pop)
    if not useSavedModels:
        final_blender = agMMGGBlending.train()
        y_hat = agMMGGBlending.predict(X_test=X_test, blender=final_blender)
        pickle.dump(final_blender, open(SavePath+".pckl", 'wb'))
    else:
        final_blender = pickle.dump(open(SavePath+".pckl", 'wb'))
        y_hat = agMMGGBlending.predict(X_test=X_test, blender=final_blender)
        

    print("AGMMFF - Score: ")
    print("MAE: %.4f" % mean_absolute_error(y_test, y_hat))

    return y_hat

def saveResultFigure(df_inmet, genscaler, y_test, y_hats, labels, city_save_path):
    logResults = ""
    logResults += "Scores" + "\n"
    print("Scores")
    
    _, ax = plt.subplots(1,1, figsize=(14,7), dpi=300)
    ticks_X = df_inmet.data.astype('str') + '-' + df_inmet.hora.astype('str')
    len_dt = len(y_test)
    ticks_X = ticks_X[-len_dt:].values
    ax.plot(ticks_X, genscaler.inverse_transform(y_test.reshape(-1, 1)), 'k-o', label='Original', linewidth=2.0)

    for y_hat, plotlabel in zip(y_hats, labels):
        logResults += "{0} ".format(plotlabel) + "- MAE: %.4f" % mean_absolute_error(y_test, y_hat) + "\n"
        logResults += "{0} ".format(plotlabel) + "- MAPE: %.4f" % mean_absolute_percentage_error(y_test, y_hat) + "\n"
        logResults += "{0} ".format(plotlabel) + "- MSE: %.4f" % mean_squared_error(y_test, y_hat) + "\n"
        trueScale_yhat = genscaler.inverse_transform(y_hat[-len_dt:].reshape(-1, 1))
        ax.plot(ticks_X, trueScale_yhat, '--o', label=plotlabel)

    plt.xticks(ticks_X[::3], rotation=45, ha='right', fontsize=12)
    ax.grid(axis='x')
    ax.legend(fontsize=12)
    ax.set_ylabel('W/m2', fontsize=12)
    plt.tight_layout()
    plt.savefig(city_save_path+"/AutoMLS_result.png", dpi=300)
    #plt.show()

    return logResults

def executeForCity(city, citiesRootFolder, city_save_path, plot=True, useSavedModels=True):
    #TODO use already trained models
    
    cityDir = "{0}/{1}".format(citiesRootFolder, city)
    csv_name = list(filter(lambda x: "historical" in x, os.listdir(cityDir)))[0]
    if csv_name == "historical_data_maceio.csv":
        csv_name = "historical_data_maceio_2.csv"
    outputLog = ""
    outputLog += "City: {0}, CSV: {1}".format(city, csv_name) + "\n"
    print("City {0}, CSV {1}".format(city, csv_name))
    
    gen, exog, df_inmet = loadData(citiesRootFolder+city+os.sep+csv_name)
    gen, genscaler = scaleData(gen)
    print("scaled gen shape", gen.shape)
    exog, exogscaler = scaleData(exog)
    X_train, y_train, X_test, y_test  = train_test_split_with_Exog(gen[:,0], exog, 24, [80,20])

    y_hats = []
    labels = []

    np.savetxt(city_save_path+"./y_test", y_test.reshape(-1, 1), delimiter=';')
    
    try:
        print("TPOT Evaluation...")
        y_hat_tpot = applyTPOT(X_train, y_train, X_test, y_test, city_save_path+"/tpotModel_{0}".format(city),
                               popSize=10, number_Generations=10,
                               useSavedModels = True)
        y_hats.append(y_hat_tpot)
        labels.append("TPOT")
        np.savetxt(city_save_path+"./y_hat_TPOT", y_hat_tpot, delimiter=';')
    except Exception:
        traceback.print_exc()
        pass

    try:
        print("HYPEROPT Evaluation...")
        y_hat_hyperopt = applyHyperOpt(X_train, y_train, X_test, y_test, city_save_path+"/hyperoptModel_{0}".format(city),
                                       max_evals=100, useSavedModels = True)
        y_hats.append(y_hat_hyperopt)
        labels.append("HYPEROPT")
        np.savetxt(city_save_path+"./y_hat_HYPEROPT", y_hat_hyperopt, delimiter=';')
    except Exception:
        traceback.print_exc()
        pass
    
    try:
        print("AUTOKERAS Evaluation...")
        y_hat_autokeras = applyAutoKeras(X_train, y_train, X_test, y_test, city_save_path+"/autokerastModel_{0}".format(city),
                                         max_trials=10, epochs=100, useSavedModels = True)
        y_hats.append(y_hat_autokeras)
        labels.append("AUTOKERAS")
        np.savetxt(city_save_path+"./y_hat_AUTOKERAS", y_hat_autokeras, delimiter=';')
    except Exception:
        traceback.print_exc()
        pass

    try:
        print("AGMMFF Evaluation...")
        y_hat_agmmff= applyGAMMFF( X_train, y_train, X_test, y_test,
                                      city_save_path+"/mmffModel_{0}".format(city),
                                      epochs=3, size_pop=20, useSavedModels = True)
        
        print("SHAPE HAT {0}".format(y_hat_agmmff.shape))
        y_hats.append(y_hat_agmmff)
        labels.append("AGMMFF")
        np.savetxt(city_save_path+"./y_hat_AGMMFF", y_hat_agmmff, delimiter=';')
    except Exception:
        traceback.print_exc()
        pass

    X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm = train_test_split_noExog(gen[:,0], 23,
                                                                        tr_vd_ts_percents = [80, 20],
                                                                        print_shapes = True)
    options_ACO={'antNumber':6, 'antTours':5, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1}
    
    try:
        print("ACOLSTM Evaluation...")
        Layers_Qtd=[[40], [20], [10]]
        epochs=[200]        
        y_hat_acolstm = applyACOLSTM(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm,
                                     city_save_path+"/acolstmModel_{0}".format(city),
                                     Layers_Qtd, epochs, options_ACO, useSavedModels = True)
        y_hats.append(y_hat_acolstm)
        labels.append("ACOLSTM")
        np.savetxt(city_save_path+"./y_hat_ACOLSTM", y_hat_acolstm, delimiter=';')
    except Exception:
        traceback.print_exc()
        pass

    try:
        print("ACOCLSTM Evaluation...")
        Layers_Qtd=[[5, 10], [10, 12], [40, 30], [20, 10], [10, 5]]
        ConvKernels=[[2, 8], [4, 6]]
        epochs=[200]
        y_hat_acoclstm = applyACOCLSTM(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm,
                                      city_save_path+"/acoclstmModel_{0}".format(city),
                                      Layers_Qtd, ConvKernels, epochs, options_ACO, useSavedModels = True)
        
        print("SHAPE HAT {0}".format(y_hat_acoclstm.shape))
        y_hats.append(y_hat_acoclstm)
        labels.append("ACOCLSTM")
        np.savetxt(city_save_path+"./y_hat_ACOCLSTM", y_hat_acoclstm, delimiter=';')
    except Exception:
        traceback.print_exc()
        pass

    if plot:
        outputLog += saveResultFigure(df_inmet, genscaler, y_test, y_hats, labels, city_save_path)

    with open(city_save_path+"/results.txt", "w") as text_file:
        text_file.write(outputLog)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    citiesRootFolder = args.cidadesRootFolder
    citiesFolders = args.listaCidades
    useSavedModels = args.useSavedModels

    for city in citiesFolders:
        cityResultsPath = "{0}/{1}/AutoMLResults".format(citiesRootFolder, city)
        if not os.path.isdir(cityResultsPath):
            os.mkdir(cityResultsPath)

        executeForCity(city, citiesRootFolder, cityResultsPath, plot = True, useSavedModels = useSavedModels)
    
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
    CLI.add_argument("-us", "--useSavedModels", type=str2bool, nargs='?',const=True, default=False,
                        help="Activate nice mode.")
    
    args = CLI.parse_args()

    main(args)