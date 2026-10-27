"""Browser test transport: one WebSocket owns one real EISeg TCP connection."""

import asyncio
import json
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .config import MAX_IMAGE_BYTES
from .eiseg_tcp import DELIMITER, MAX_FRAME_BYTES


router = APIRouter()


@router.websocket("/ui/tcp-test")
async def tcp_test(websocket: WebSocket, port: int = 6063):
    # WebSockets bypass CORS; only the service's own browser page may open a bridge.
    origin = websocket.headers.get("origin")
    authority = websocket.headers.get("host", "").lower()
    if origin and origin.lower() not in {f"http://{authority}", f"https://{authority}"}:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    writer = None
    tasks = []
    try:
        # Only the listening port is user controlled; the bridge cannot reach arbitrary hosts.
        host = "127.0.0.1"
        if not 1 <= port <= 65535:
            raise ValueError("TCP 端口必须在 1–65535 之间")
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, limit=MAX_FRAME_BYTES), timeout=10
        )
        await websocket.send_json({
            "type": "connected", "host": host, "port": port, "max_image_bytes": MAX_IMAGE_BYTES,
        })

        async def browser_to_tcp():
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                payload = message.get("text")
                if payload is None or len(payload) > MAX_FRAME_BYTES:
                    raise ValueError("仅支持大小限制内的 JSON 文本消息")
                request = json.loads(payload)
                if (not isinstance(request, dict) or type(request.get("cmd")) is not int
                        or request["cmd"] not in {8000, 8001, 8002, 8004}
                        or not isinstance(request.get("data", {}), dict)):
                    raise ValueError("仅支持 EISeg 8000/8001/8002/8004 命令")
                frame = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
                # Prevent a JSON string containing the delimiter from injecting TCP frames.
                if len(frame) > MAX_FRAME_BYTES or DELIMITER in frame:
                    raise ValueError("TCP 消息过大或包含协议分隔符 ##@##")
                writer.write(frame + DELIMITER)
                await writer.drain()

        async def tcp_to_browser():
            while True:
                # readuntil handles both split packets and several replies in one packet.
                frame = await asyncio.wait_for(reader.readuntil(DELIMITER), timeout=600)
                response = json.loads(frame[:-len(DELIMITER)])
                if not isinstance(response, dict):
                    raise ValueError("TCP 返回的消息不是 JSON 对象")
                await websocket.send_json({"type": "response", "response": response})

        # Read both directions so closing the page also interrupts a stalled TCP reply.
        tasks = [asyncio.create_task(browser_to_tcp()), asyncio.create_task(tcp_to_browser())]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    except (OSError, ValueError, asyncio.IncompleteReadError, asyncio.LimitOverrunError) as exc:
        if isinstance(exc, TimeoutError):
            message = "TCP 连接或空闲等待超时"
        elif isinstance(exc, asyncio.IncompleteReadError):
            message = "TCP 服务已断开连接（或响应不完整）"
        elif isinstance(exc, asyncio.LimitOverrunError):
            message = "TCP 响应超过大小限制"
        else:
            message = str(exc)
        with suppress(RuntimeError, OSError, WebSocketDisconnect):
            await websocket.send_json({"type": "error", "message": message})
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if writer is not None:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()
        with suppress(RuntimeError, OSError, WebSocketDisconnect):
            await websocket.close()
