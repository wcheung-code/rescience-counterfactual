import numpy as np
import cvxpy as cp

class EqualizedOddsPostProcesser():
    def __init__(self, parameters, y_pred_name = 'counterfactual', 
                 y_true_name = '_Y', sensitive_feature_name = '_A', seed = 0):
        self.seed = seed
        self.parameters = parameters
        self.pred_name = y_pred_name
        self.true_name = self.parameters['target']['observational']
        self.sensitive_feature_name = self.parameters['features']['sensitive']

    def post_process(self, data, metric_frame, mix_rates = None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.mix_rates(data, metric_frame)

        data, results = dict(tuple(data.groupby(self.sensitive_feature_name))), {}
        for value in [0, 1]:

            p2p, n2p = mix_rates[value, :]

            fair_pred = data[value][self.pred_name].values
            __ = 1 - fair_pred.copy()

            pp_indices, = self.__transform_preds(fair_pred)
            pn_indices, = self.__transform_preds(__)

            np.random.seed(self.seed)
            np.random.shuffle(pp_indices)
            np.random.shuffle(pn_indices)

            n2p_indices = pn_indices[:int(len(pn_indices) * n2p)]
            fair_pred[n2p_indices] = 1 - fair_pred[n2p_indices]
            p2n_indices = pp_indices[:int(len(pp_indices) * (1 - p2p))]
            fair_pred[p2n_indices] = 1 - fair_pred[p2n_indices]

            results[value] = fair_pred

        if not has_mix_rates:
            return results, mix_rates
        else:
            return results

    def mix_rates(self, data, metric_frame):
        base_rates = metric_frame.pop(item = 'base_rate')
        S, N = tuple(np.reshape(metric_frame.to_numpy(), 
                                newshape = (2,2,2)).astype(float))

        sensitive = cp.Variable((2,2), nonneg=True)
        nonsensitive = cp.Variable((2,2), nonneg=True)

        sfpr_sfnr = cp.trace(S @ sensitive)
        ofpr_ofnr = cp.trace(N @ nonsensitive)

        error = sfpr_sfnr + ofpr_ofnr

        sm = self._get_booleans(
            data[data[self.sensitive_feature_name] == 1])
        om = self._get_booleans(
            data[data[self.sensitive_feature_name] == 0])

        spn_given_p = cp.sum(cp.multiply(
            self._given_p(data, sm), sensitive)) / base_rates[1]
        spp_given_n = cp.sum(cp.multiply(
            self._given_n(data, sm), sensitive)) / (1 - base_rates[1])
        opn_given_p = cp.sum(cp.multiply(
            self._given_p(data, om), nonsensitive)) / base_rates[0]
        opp_given_n = cp.sum(cp.multiply(
            self._given_n(data, om), nonsensitive)) / (1 - base_rates[0])

        constraints = [
            cp.sum(sensitive, axis = 1) == 1,
            cp.sum(nonsensitive, axis = 1) == 1,
            sensitive[:, 0] <= 1-1e-4,
            sensitive[:, 0] >= 0,
            nonsensitive[:, 0] <= 1-1e-4,
            nonsensitive[:, 0] >= 0,
            spp_given_n == opp_given_n,
            spn_given_p == opn_given_p
        ]

        prob = cp.Problem(cp.Minimize(error), constraints)
        prob.solve()

        results = [nonsensitive.value[::-1, 0], sensitive.value[::-1, 0]]
        return np.vstack(results)

    def _given_p(self, data, booleans):
        const, flip = self._get_complements(data)
        return np.array([
            [(flip * booleans['fn']).mean(), (const * booleans['fn']).mean()],
            [(const * booleans['tp']).mean(), (flip * booleans['tp']).mean()], 
        ])

    def _given_n(self, data, booleans):
        const, flip = self._get_complements(data)
        return np.array([
            [(flip * booleans['tn']).mean(), (const * booleans['tn']).mean()],
            [(const * booleans['fp']).mean(), (flip * booleans['fp']).mean()], 
        ])


    def _get_complements(self, data):
        return data[self.pred_name], 1 - data[self.pred_name]

    def _get_booleans(self, data):
        return {
            'tn': self._booleans(data, (0, 0)),
            'fp': self._booleans(data, (1, 0)),
            'fn': self._booleans(data, (0, 1)),
            'tp': self._booleans(data, (1, 1)),
        }

    def _booleans(self, data, indicator_tuple):
        i, j = indicator_tuple
        return np.logical_and(data[self.pred_name].round() == i,
                              data[self.true_name] == j)

    def __transform_preds(self, p):
        return np.nonzero(np.round(p))