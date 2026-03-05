import json

from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from jose import JWTError, jwt

from app.config import settings
from app.services.prediction import predict_cognitive_load

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: int):
        self.active_connections.pop(user_id, None)

    async def send_json(self, user_id: int, data: dict):
        ws = self.active_connections.get(user_id)
        if ws:
            await ws.send_json(data)


manager = ConnectionManager()


def _authenticate_token(token: str) -> int | None:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        user_id = payload.get("sub")
        return int(user_id) if user_id is not None else None
    except (JWTError, ValueError):
        return None


@router.websocket("/ws/cognitive")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001)
        return

    user_id = _authenticate_token(token)
    if user_id is None:
        await websocket.close(code=4001)
        return

    await manager.connect(websocket, user_id)
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            features = [
                data.get("typing_speed", 0),
                data.get("speed_variance", 0),
                data.get("backspace_rate", 0),
                data.get("mouse_distance", 0),
                data.get("mouse_jitter", 0),
                data.get("tab_switch_count", 0),
            ]
            prediction = predict_cognitive_load(features)
            await manager.send_json(user_id, {
                "prediction": prediction,
                "features": {
                    "typing_speed": features[0],
                    "speed_variance": features[1],
                    "backspace_rate": features[2],
                    "mouse_distance": features[3],
                    "mouse_jitter": features[4],
                    "tab_switch_count": features[5],
                },
            })
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception:
        manager.disconnect(user_id)
