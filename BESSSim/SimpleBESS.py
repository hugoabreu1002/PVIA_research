import pandas as pd
import numpy as np

class SimpleBEES(object):
    def __init__(self, CostAndDemandFile="./BESSSim/Cost_and_hourly_demand.xlsx") -> None:
        super().__init__()
        self._CostAndDemandFile = CostAndDemandFile

    def simulate(self, R_true, R_pred, solarRadeFi=0.15, DemandData=None, DemandFactor=0.3, State_of_Chage_I=0.5, State_of_Chage_F = 1.0, type='f'):
        #TODO update description
        """ 
            Charge_data: The data of the charge on the BESS, if None, uses the data loaded by the CostAndDemandFile.
            Default = None.

            returns power_flux_true, power_flux_pred
        """
        if DemandData == None:
            self.loadCostAndDemandCurves(DemandFactor)
        else:
            self._DemandData = DemandData*DemandFactor

        if type=='f':
            power_generated_pred = solarRadeFi*R_pred
            power_generated_true = solarRadeFi*R_true

            batery_charge_pred = np.zeros(len(self._DemandData))
            batery_charge_true = np.zeros(len(self._DemandData))
            batery_charge_pred[0] = State_of_Chage_I
            batery_charge_true[0] = State_of_Chage_I
            
            for i in range(1, len(self._DemandData)):

                batery_charge_pred[i] = batery_charge_pred[i-1] + power_generated_pred[i] - self._DemandData[i]
                batery_charge_true[i] = batery_charge_true[i-1] + power_generated_true[i] - self._DemandData[i]

                if batery_charge_pred[i] < 0:
                    batery_charge_pred[i] = 0
                    
                if batery_charge_true[i] < 0:
                    batery_charge_true[i] = 0

                if batery_charge_pred[i] > 1:
                    batery_charge_pred[i] = 1
                    
                if batery_charge_true[i] > 1:
                    batery_charge_true[i] = 1
        #TODO type control
        if type == 'm':
            power_generated_pred = solarRadeFi*R_pred
            power_generated_true = solarRadeFi*R_true

            peak_radiation = np.argmax(power_generated_true)

            batery_charge_pred = np.zeros(len(self._DemandData))
            batery_charge_true = np.zeros(len(self._DemandData))
            batery_charge_pred[0] = State_of_Chage_I
            batery_charge_true[0] = State_of_Chage_I
            
            for i in range(1, len(self._DemandData)):
                
                batery_charge_pred[i] = batery_charge_pred[i-1] + power_generated_pred[i] - self._DemandData[i]
                batery_charge_true[i] = batery_charge_true[i-1] + power_generated_true[i] - self._DemandData[i]

                if (batery_charge_true[i-1] >= State_of_Chage_F) and (self._DemandData[i] >= power_generated_true[i]) and i>peak_radiation:
                    batery_charge_true[i] = State_of_Chage_F
                    batery_charge_pred[i] = State_of_Chage_F

                if batery_charge_pred[i] < 0:
                    batery_charge_pred[i] = 0
                    
                if batery_charge_true[i] < 0:
                    batery_charge_true[i] = 0

                if batery_charge_pred[i] > 1:
                    batery_charge_pred[i] = 1
                    
                if batery_charge_true[i] > 1:
                    batery_charge_true[i] = 1
                    
        return batery_charge_true, batery_charge_pred

    def syncDemand(self):
        return None    

    def loadCostAndDemandCurves(self, DemandFactor):
        df = pd.read_excel(self._CostAndDemandFile, engine="openpyxl")
        df_p = self.resampleDF(df)
        self._DemandData = df_p["Demand Variation [%]"].values
        self._DemandData = self._DemandData/100*DemandFactor
        self._CoEpu = df_p["CoE [pu]"].values

    def resampleDF(self, df, resample_rate=2, offset=0):
        df_new = {}
        for column in df.columns:
            df_new[column] = df[column].values[0::resample_rate]

        return pd.DataFrame(df_new)