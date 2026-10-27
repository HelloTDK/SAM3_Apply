from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


def find_service_root():
    for directory in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (directory / "sam3_service").is_dir():
            return directory
        if (directory / "SAM3_Apply" / "sam3_service").is_dir():
            return directory / "SAM3_Apply"
    raise RuntimeError("Could not locate the SAM3_Apply/sam3_service package")


SERVICE_ROOT = find_service_root()
ENGINE_DIR = Path(
    os.getenv(
        "SAM3_NATIVE_ENGINE_DIR",
        str(SERVICE_ROOT / "engines" / "native_interactive"),
    )
)
sys.path.insert(0, str(SERVICE_ROOT))
from sam3_service.native_click_trt import timed_interaction, timed_step  # noqa: E402


BACKEND = None
BACKEND_LOCK = threading.Lock()


def get_backend():
    global BACKEND
    if BACKEND is None:
        with BACKEND_LOCK:
            if BACKEND is None:
                from sam3_service.native_click_trt import NativeInteractiveBackend

                with timed_step("后端初始化"):
                    BACKEND = NativeInteractiveBackend(ENGINE_DIR)
    return BACKEND


@timed_interaction("加载图片")
def load_image(image: Image.Image | None):
    if image is None:
        return None, "请先上传图片"

    try:
        image = image.convert("RGB")
        session = {
            "image": image,
            "features": get_backend().encode_image(image),
            "clicks": [],
            "history": [],
        }
        return session, f"图片已加载：{image.width} × {image.height}"
    except Exception as exc:
        return None, f"图片加载失败：{exc}"


@timed_step("预览绘制")
def render(session: dict | None):
    if not session:
        return None, None, "请先上传图片"
    image = session["image"]
    if not session["history"]:
        return image, None, "尚无分割结果"

    record = session["history"][-1]
    mask = record["mask"].astype(np.uint8) * 255
    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    rgba[mask > 0] = (0, 190, 140, 115)
    preview = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(rgba))
    draw = ImageDraw.Draw(preview)
    radius = max(5, min(12, min(image.size) // 80))
    for x, y, label in session["clicks"]:
        color = (0, 220, 130, 255) if label == 1 else (255, 75, 75, 255)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline="white",
            width=2,
        )

    label = "正例" if record["point"][2] == 1 else "负例"
    status = f"{label}，点击数：{len(session['clicks'])}，置信度：{record['score']:.3f}"
    return preview.convert("RGB"), Image.fromarray(mask), status


@timed_interaction("点击分割")
def predict_click(image: Image.Image | None, session: dict | None, event: gr.EventData):
    if image is None:
        return session, None, None, "请先上传图片"
    if session is None:
        session, status = load_image(image)
        if session is None:
            return None, image, None, status

    x, y, label = int(event.x), int(event.y), event.label
    if type(label) is not int or label not in (0, 1):
        return session, None, None, "无效的点击类型"
    if not (0 <= x < image.width and 0 <= y < image.height):
        return session, None, None, "点击位置超出图片范围"

    clicks = session["clicks"] + [(x, y, label)]
    previous_logits = session["history"][-1]["logits"] if session["history"] else None
    try:
        mask, score, logits = get_backend().segment(
            session["features"], image.size, clicks, previous_logits
        )
    except Exception as exc:
        preview, mask, _ = render(session)
        return session, preview, mask, f"分割失败：{exc}"

    session["clicks"] = clicks
    session["history"].append(
        {"point": (x, y, label), "mask": mask, "score": score, "logits": logits}
    )
    preview, mask, status = render(session)
    return session, preview, mask, status


@timed_interaction("撤销点击")
def undo_click(session: dict | None):
    if session and session["history"]:
        session["clicks"].pop()
        session["history"].pop()
    preview, mask, status = render(session)
    return session, preview, mask, status


@timed_interaction("清空点击")
def clear_clicks(session: dict | None):
    if session:
        session["clicks"].clear()
        session["history"].clear()
    preview, mask, status = render(session)
    return session, preview, mask, status


@timed_interaction("清除图片")
def clear_image():
    return None, None, None, "请先上传图片"


with gr.Blocks(title="SAM3 TensorRT 点击分割 Demo") as demo:
    gr.Markdown("# SAM3 TensorRT 点击分割")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil", label="输入图片", height=600, elem_id="sam3-trt-input-image"
            )
        with gr.Column():
            result_image = gr.Image(label="分割预览", height=500)
            mask_image = gr.Image(label="二值 Mask", height=500)
    with gr.Row():
        undo = gr.Button("撤销上一个点")
        clear = gr.Button("清空点击")
    status = gr.Textbox(label="状态", interactive=False)
    session_state = gr.State(None, time_to_live=1800)
    click_source = gr.HTML(
        visible="hidden",
        elem_id="sam3-trt-click-source",
        js_on_load="""
            const imageRoot = "#sam3-trt-input-image";
            // The native image event does not distinguish positive and negative clicks.
            document.addEventListener("mousedown", (event) => {
                const image = event.target instanceof HTMLImageElement ? event.target : null;
                if (!image || !image.closest(imageRoot) || ![0, 2].includes(event.button)) return;

                if (event.button === 2) event.preventDefault();
                const rect = image.getBoundingClientRect();
                const scale = Math.min(
                    rect.width / image.naturalWidth,
                    rect.height / image.naturalHeight
                );
                const width = image.naturalWidth * scale;
                const height = image.naturalHeight * scale;
                const left = rect.left + (rect.width - width) / 2;
                const top = rect.top + (rect.height - height) / 2;
                const x = Math.floor((event.clientX - left) / width * image.naturalWidth);
                const y = Math.floor((event.clientY - top) / height * image.naturalHeight);
                if (x < 0 || y < 0 || x >= image.naturalWidth || y >= image.naturalHeight) return;

                trigger("click", {x, y, label: event.button === 0 ? 1 : 0});
            }, true);
            document.addEventListener("contextmenu", (event) => {
                if (event.target instanceof HTMLImageElement && event.target.closest(imageRoot)) {
                    event.preventDefault();
                }
            }, true);
        """,
    )

    image_input.change(
        load_image,
        image_input,
        [session_state, status],
        concurrency_id="sam3_tensorrt",
    )
    image_input.clear(
        clear_image,
        outputs=[session_state, result_image, mask_image, status],
    )
    click_source.click(
        predict_click,
        [image_input, session_state],
        [session_state, result_image, mask_image, status],
        concurrency_id="sam3_tensorrt",
    )
    undo.click(
        undo_click,
        session_state,
        [session_state, result_image, mask_image, status],
        concurrency_id="sam3_tensorrt",
    )
    clear.click(
        clear_clicks,
        session_state,
        [session_state, result_image, mask_image, status],
        concurrency_id="sam3_tensorrt",
    )


def main():
    from sam3_service.eiseg_tcp import InteractiveSegmentationServer

    address = (
        os.getenv("SAM3_TCP_HOST", "0.0.0.0"),
        int(os.getenv("SAM3_TCP_PORT", "6063")),
    )
    # TCP and Gradio share engines; each TCP connection retains its own features and logits.
    with InteractiveSegmentationServer(address, backend=get_backend()) as server:
        tcp_thread = threading.Thread(target=server.start_server, daemon=True)
        tcp_thread.start()
        print(f"SAM3 TCP 服务已启动：{server.server_address}", flush=True)
        try:
            demo.queue(default_concurrency_limit=1).launch(
                server_name="0.0.0.0", server_port=7861
            )
        finally:
            server.shutdown()
            tcp_thread.join()


if __name__ == "__main__":
    main()
