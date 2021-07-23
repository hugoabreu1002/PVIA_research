class SimpleBEES(object):
    def __init__(self, eFi, backUpPercent) -> None:
        super().__init__()
        self._eFi = eFi
        self._backUpPercent = backUpPercent

    def simulate(self, R_true, R_pred, Charge_data, Initial_Cond=1):
        
        Charge_data = Charge_data/max(Charge_data)
        R_true = R_true/max(R_true)
        R_pred = R_pred/max(R_pred)

        power_flux_pred = Initial_Cond + self._eFi*R_pred - Charge_data
        power_flux_true = Initial_Cond + self._eFi*R_true - Charge_data