class LinearDecay:
    def __init__(self, alpha=0, beta=0):
        self.alpha = alpha
        self.beta = beta
        self.setMeta()

    def __str__(self):
        return str(self.meta)

    def __repr__(self):
        return str(self.meta)

    def loadFromDict(self, dictionary):
        self.alpha = dictionary['alpha']
        self.beta = dictionary['beta']
        self.setMeta()

    def setMeta(self):
        self.meta = dict()
        self.meta["DecayFunction"] = "LinearDecay"
        self.meta["alpha"] = self.alpha
        self.meta["beta"] = self.beta

    def decay(self, t):
        val = self.beta - (t * self.alpha)
        if val > 0:
            return val
        else:
            return 0.0

    def toJSON(self):
        return self.meta


REGISTER = {
    'LinearDecay' : LinearDecay
}