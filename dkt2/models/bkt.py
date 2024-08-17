from torch.nn import Module
import torch
class BKT(Module):
    def __init__(self, slip_logit=0.1, guess_logit=0.3, train_p=0.1, learn_p=0.5):
        super().__init__()
        self.slip_logit = slip_logit
        self.guess_logit = guess_logit
        self.train_p = train_p
        self.learn_p = learn_p
        

    def forward(self, feed_dict):
        # q = feed_dict['skills']
        r = feed_dict['responses']
        r_shft = r[:, 1:]
        masked_r = r * (r > -1).long()
        r_input = masked_r[:, :-1]
        pred = []
        for s_r in r_input:
            model_values = []
            k = 0.0
            Ktm1 = self.learn_p

            for r_ in s_r:
                if r_ == 1:
                    k = Ktm1 * (1 - self.slip_logit) / (Ktm1 * (1 - self.slip_logit) + self.guess_logit * (1 - Ktm1))
                else:
                    k = (Ktm1 * self.slip_logit) / (Ktm1 * self.slip_logit + (1 - Ktm1) * (1 - self.guess_logit))
                Kt = k + (1 - k) * self.train_p
                Ktm1 = Kt
                model_values.append(Kt)
            pred.append(model_values)
        
        out_dict = {
            "pred": torch.tensor(pred),
            "true":  r_shft.float(),
        }
        return out_dict
