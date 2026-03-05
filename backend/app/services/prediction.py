import os
import numpy as np
import torch
import joblib

from app.config import settings


class CognitiveLSTM(torch.nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, num_classes=3):
        super(CognitiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


LABELS = ["Low", "Medium", "High"]

_model = None
_scaler = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    _model = CognitiveLSTM()
    if os.path.exists(settings.MODEL_PATH):
        _model.load_state_dict(torch.load(settings.MODEL_PATH, weights_only=True))
        _model.eval()
    return _model


def _load_scaler():
    global _scaler
    if _scaler is not None:
        return _scaler

    if os.path.exists(settings.SCALER_PATH):
        _scaler = joblib.load(settings.SCALER_PATH)
    else:
        from sklearn.preprocessing import StandardScaler
        _scaler = StandardScaler()
        _scaler.mean_ = np.zeros(6)
        _scaler.scale_ = np.ones(6)
        _scaler.var_ = np.ones(6)
        _scaler.n_features_in_ = 6
    return _scaler


def predict_cognitive_load(features: list[float]) -> dict:
    """
    Predict cognitive load from 6 features.
    Returns dict with predicted_load, confidence, and load_percentage.
    """
    model = _load_model()
    scaler = _load_scaler()

    features_array = np.array(features).reshape(1, -1)
    scaled = scaler.transform(features_array)
    # LSTM expects (batch, seq_len, features) — use seq_len=1 for single-step
    tensor_input = torch.FloatTensor(scaled).unsqueeze(1)

    with torch.no_grad():
        output = model(tensor_input)
        probabilities = torch.softmax(output, dim=1).numpy()[0]

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    # Load percentage: weighted sum where Low=0-33, Medium=34-66, High=67-100
    load_percentage = float(
        probabilities[0] * 16.5
        + probabilities[1] * 50.0
        + probabilities[2] * 83.5
    )

    return {
        "predicted_load": LABELS[predicted_class],
        "confidence": round(confidence, 4),
        "load_percentage": round(load_percentage, 2),
    }
