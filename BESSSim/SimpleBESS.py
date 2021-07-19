class SimpleBEES(object):
    def __init__(self, eFi, backUpPercent) -> None:
        super().__init__()
        self._eFi = eFi
        self._backUpPercent = backUpPercent

    def simulate(self, R_true, R_pred, Charge_data, Initial_Cond=1):
        