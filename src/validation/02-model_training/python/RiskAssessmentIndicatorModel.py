from sklearn.model_selection import train_test_split

class RiskAssessmentIndicatorModel():
    def __init__(self, model, name = 'observational', seed = 0):
        self.model = model
        self.name = name
        self.seed = seed

    def fit(self, df, parameters, sample_weight = None):
        train, test = self._preprocess_data(df, parameters)
        features = parameters['features']['training']
        target = parameters['target']
        sample_weight = parameters['sample_weight']
        if sample_weight:
            self.model.fit(train[features], train[target], 
                           sample_weight = train[sample_weight])
        else:
            self.model.fit(train[features], train[target])
        return self.model

    def _preprocess_data(self, df, parameters):
        if parameters['is_train']:
            (_, test), (_, train) = tuple(df.groupby(parameters['is_train']))
        else:
            train, test = train_test_split(df, test_size=0.3, random_state=self.seed)
        if self.name == 'counterfactual':
            treat_dict = parameters['treat']
            train = train[train[treat_dict['name']] == 0]
            test = test[test[treat_dict['name']] == 0]
        return train, test