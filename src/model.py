import torch
import torch.nn as nn

# exclude global orientation from theta + betas 69 + 10 = 79
GRU_INPUT_SIZE = 79
RECURRENT_STATE_SIZE = 1500

class GarmentFitRegressor(nn.Module):
    def __init__(self, nv):
        super(GarmentFitRegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(10, 20),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(20, nv * 3)
        )

    def forward(self, beta):
        disp = self.regressor(beta)
        disp = disp.view(beta.size(0), -1, 3)
        return disp

class SimpleGarmentWrinkleRegressor(nn.Module):
    def __init__(self, nv):
        super(SimpleGarmentWrinkleRegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(79, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, nv * 3)
        )

    def forward(self, inputs):
        return self.regressor(inputs), None


class GarmentWrinkleRegressor(nn.Module):
    def __init__(self, nv):
        super(GarmentWrinkleRegressor, self).__init__()
        self.gru_model = nn.GRU(
            input_size=GRU_INPUT_SIZE,
            hidden_size=RECURRENT_STATE_SIZE,
            num_layers=1,
            dropout=0,
            batch_first=True
        )
        self.output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(RECURRENT_STATE_SIZE, nv * 3),
        )
        self.gru_state = None

    def repackage_rnn_state(self):
        self.gru_state = self._detach_rnn_state(self.gru_state)
        return

    def _detach_rnn_state(self, h):
        if isinstance(h, torch.Tensor):     return h.detach()
        else:   return tuple(self._detach_rnn_state(v) for v in h)

    def forward(self, inputs, input_state):
        # outputs: N x T x F propagate upward
        outputs, _ = self.gru_model(inputs, input_state)
        if not isinstance(outputs, torch.Tensor):
            outputs, sizes = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        disp = self.output_layer(outputs)
        return disp, outputs

if __name__ == '__main__':
    betas = torch.randn(256, 10)
    model = GarmentFitRegressor(4000)
    disp = model(betas)
    print(disp.shape)
    torch.save(model, "fit_model.pth.tar")

    model = GarmentWrinkleRegressor(4000)
    # mimic a propagation logic
    bs = 16; seq_len = 90
    inputs = torch.randn(bs, seq_len, GRU_INPUT_SIZE)
    seqlens = [8] * 8 + [22, 4, 5, 88, 77, 90, 11, 55]
    inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, seqlens, enforce_sorted=False, batch_first=True)
    initial_state = torch.randn(1, bs, RECURRENT_STATE_SIZE)
    disp, outputs = model(inputs, initial_state)
    print(model.gru_state.shape, initial_state.shape)
