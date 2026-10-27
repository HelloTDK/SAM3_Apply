// Keep TCP test state independent of the HTTP detection and sample annotation forms.
(() => {
    'use strict';
    const ui = (name) => document.getElementById(`tcp${name}`);
    let socket = null;
    let connected = false;
    let busy = false;
    let pending = null;
    let imageReady = false;
    let imageData = '';
    let imageId = '';
    let maxImageBytes = 20 * 1024 * 1024;
    let clicks = [];
    let polygons = [];
    let boxes = [];

    function showError(message = '') {
        ui('Error').textContent = message;
        ui('Error').classList.toggle('show', Boolean(message));
    }

    function updateControls() {
        ui('Port').disabled = socket !== null;
        ui('Connect').disabled = socket !== null;
        ui('Disconnect').disabled = socket === null;
        for (const name of ['InitModel', 'Models', 'Model', 'UseGpu']) {
            ui(name).disabled = !connected || busy;
        }
        ui('ImageFile').disabled = busy;
        ui('LoadImage').disabled = !connected || busy || !imageData;
        for (const name of ['SendClick', 'Undo', 'Finish', 'Reset', 'X', 'Y', 'ClickType']) {
            ui(name).disabled = !connected || busy || !imageReady;
        }
        ui('Canvas').style.cursor = imageReady && !busy ? 'crosshair' : 'not-allowed';
    }

    function draw() {
        const canvas = ui('Canvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Keep markers legible when a large image is scaled down by the page layout.
        const scale = canvas.width / (canvas.getBoundingClientRect().width || canvas.width);
        ctx.lineWidth = 2 * scale;
        ctx.strokeStyle = '#667eea';
        ctx.fillStyle = 'rgba(102, 126, 234, 0.3)';
        for (const polygon of polygons) {
            if (!Array.isArray(polygon) || polygon.length < 3) continue;
            ctx.beginPath();
            polygon.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
            ctx.closePath();
            ctx.fill('evenodd'); // EISeg bridges holes into a single flat polygon.
            ctx.stroke();
        }
        ctx.strokeStyle = '#d97706';
        ctx.setLineDash([6 * scale, 4 * scale]);
        for (const box of boxes) {
            if (Array.isArray(box) && box.length >= 4) ctx.strokeRect(...box.slice(0, 4));
        }
        ctx.setLineDash([]);
        for (const {type, coordinate: [x, y]} of clicks) {
            ctx.beginPath();
            ctx.arc(x, y, 5 * scale, 0, Math.PI * 2);
            ctx.fillStyle = type === 1 ? '#15803d' : '#dc2626';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.stroke();
            ctx.fillStyle = '#fff';
            ctx.font = `bold ${10 * scale}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(type === 1 ? '+' : '−', x, y);
        }
    }

    function clearSession() {
        imageReady = false;
        imageId = '';
        clicks = [];
        polygons = [];
        boxes = [];
        ui('ResultInfo').textContent = '图片尚未载入 TCP 会话';
        draw();
    }

    function settlePending(error, response) {
        if (!pending) return;
        const current = pending;
        pending = null;
        clearTimeout(current.timer);
        if (error) current.reject(error);
        else current.resolve(response);
    }

    function request(cmd, data = {}) {
        if (!connected || pending) return Promise.reject(new Error('TCP 未连接或上一条请求尚未完成'));
        const message = {cmd, data};
        const visible = {...message, data: {...data}};
        if (visible.data.image_base64) {
            visible.data.image_base64 = `[Base64 已省略，${data.image_base64.length} 字符]`;
        }
        ui('Request').textContent = JSON.stringify(visible, null, 2);
        ui('Response').textContent = '等待 TCP 响应…';
        const started = performance.now();
        return new Promise((resolve, reject) => {
            pending = {cmd, resolve, reject, started, timer: setTimeout(() => {
                // A late reply cannot safely be paired with the next command: reconnect instead.
                settlePending(new Error('TCP 请求超过 120 秒，请重新连接后重试'));
                socket?.close();
            }, 120000)};
            try {
                socket.send(JSON.stringify(message));
            } catch (error) {
                settlePending(error);
                socket?.close();
            }
        });
    }

    async function run(action) {
        if (busy) return;
        busy = true;
        showError();
        updateControls();
        try {
            await action();
        } catch (error) {
            if (error.message.includes('响应命令与请求不匹配')) socket?.close();
            showError(error.message);
        } finally {
            busy = false;
            updateControls();
        }
    }

    ui('ConnectForm').addEventListener('submit', (event) => {
        event.preventDefault();
        if (socket) return;
        showError();
        clearSession();
        const url = new URL('/ui/tcp-test', window.location.href);
        url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        url.searchParams.set('port', ui('Port').value);
        const connection = new WebSocket(url);
        socket = connection;
        ui('Status').textContent = '正在连接…';
        updateControls();
        connection.addEventListener('message', (event) => {
            if (socket !== connection) return;
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'connected') {
                    connected = true;
                    maxImageBytes = message.max_image_bytes;
                    ui('Status').textContent = `已连接 ${message.host}:${message.port}`;
                    updateControls();
                } else if (message.type === 'error') {
                    showError(message.message);
                    settlePending(new Error(message.message));
                } else if (message.type === 'response') {
                    const response = message.response;
                    ui('Response').textContent = JSON.stringify(response, null, 2);
                    if (!pending || response.cmd !== pending.cmd + 1000) {
                        throw new Error('TCP 响应命令与请求不匹配，请重新连接');
                    }
                    ui('Status').textContent = `已连接 127.0.0.1:${ui('Port').value} · ${Math.round(performance.now() - pending.started)} ms`;
                    const failure = response.cmd !== 9004 && Number(response.data?.status) !== 1;
                    settlePending(failure ? new Error(response.data?.message || 'TCP 操作失败') : null, response.data);
                }
            } catch (error) {
                showError(error.message);
                settlePending(error);
                connection.close();
            }
        });
        connection.addEventListener('error', () => {
            if (socket === connection) {
                showError('WebSocket 连接失败，请检查 HTTP 服务和代理是否支持 WebSocket');
                connection.close();
            }
        });
        connection.addEventListener('close', () => {
            if (socket !== connection) return;
            socket = null;
            connected = false;
            settlePending(new Error('TCP 连接已断开，请重新连接并载入图片'));
            clearSession();
            ui('Status').textContent = '已断开';
            updateControls();
        });
    });

    ui('Disconnect').addEventListener('click', () => socket?.close());
    window.addEventListener('pagehide', () => socket?.close());
    ui('Models').addEventListener('click', () => run(() => request(8004)));
    ui('InitModel').addEventListener('click', () => run(async () => {
        clearSession();
        await request(8000, {model: Number(ui('Model').value), use_gpu: ui('UseGpu').value === 'true'});
    }));

    ui('ImageFile').addEventListener('change', () => run(async () => {
        clearSession();
        imageData = '';
        ui('Preview').hidden = true;
        ui('ImageInfo').textContent = '请选择图片';
        const file = ui('ImageFile').files[0];
        if (!file) return;
        if (file.size > maxImageBytes) throw new Error(`图片不能超过 ${maxImageBytes / 1024 / 1024} MiB`);
        const data = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(new Error('读取图片失败'));
            reader.readAsDataURL(file);
        });
        ui('Image').src = data;
        await ui('Image').decode();
        imageData = data;
        ui('Canvas').width = ui('Image').naturalWidth;
        ui('Canvas').height = ui('Image').naturalHeight;
        ui('X').max = ui('Image').naturalWidth - 1;
        ui('Y').max = ui('Image').naturalHeight - 1;
        ui('X').value = ui('Y').value = '0';
        ui('Preview').hidden = false;
        ui('ImageInfo').textContent = `${file.name} · ${ui('Image').naturalWidth} × ${ui('Image').naturalHeight} · 尚未载入 TCP`;
        draw();
    }));

    ui('LoadImage').addEventListener('click', () => run(async () => {
        clearSession();
        if (ui('ImageFile').files[0].size > maxImageBytes) throw new Error('图片超过目标服务的大小限制');
        // A fresh filename-safe ID avoids the original EISeg service reusing its disk cache.
        imageId = `tcp-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
        await request(8001, {image_id: imageId, image_url: '', image_base64: imageData});
        imageReady = true;
        ui('ImageInfo').textContent = `${ui('Image').naturalWidth} × ${ui('Image').naturalHeight} · image_id: ${imageId}`;
        ui('ResultInfo').textContent = '图片已载入，可添加正负点';
    }));

    function sendClick(type, coordinate) {
        return run(async () => {
            if (!imageReady) throw new Error('请先连接 TCP 并载入图片');
            const data = {image_id: imageId, click_type: type};
            if (coordinate) data.coordinate = coordinate;
            const result = await request(8002, data);
            // Only successful replies change the displayed history, matching TCP rollback.
            if (type === 1 || type === 2) clicks.push({type, coordinate});
            else if (type === 3) clicks.pop();
            else clicks = [];
            polygons = Array.isArray(result.ploygon) ? result.ploygon : [];
            boxes = Array.isArray(result.bnd_boxs) ? result.bnd_boxs : [];
            ui('ResultInfo').textContent = `当前目标 ${clicks.length} 个点 · ${polygons.length} 个多边形${type === 4 ? ' · 已完成，可继续标注下一目标' : ''}`;
            draw();
        });
    }

    ui('ClickForm').addEventListener('submit', (event) => {
        event.preventDefault();
        sendClick(Number(ui('ClickType').value), [Number(ui('X').value), Number(ui('Y').value)]);
    });
    for (const [name, type] of [['Undo', 3], ['Finish', 4], ['Reset', 5]]) {
        ui(name).addEventListener('click', () => sendClick(type));
    }
    function canvasClick(event, type) {
        if (!imageReady || busy) return;
        const rect = ui('Canvas').getBoundingClientRect();
        // Screen pixels must be mapped back to original-image pixels for both services.
        const x = Math.max(0, Math.min(ui('Canvas').width - 1, Math.floor((event.clientX - rect.left) * ui('Canvas').width / rect.width)));
        const y = Math.max(0, Math.min(ui('Canvas').height - 1, Math.floor((event.clientY - rect.top) * ui('Canvas').height / rect.height)));
        ui('X').value = x;
        ui('Y').value = y;
        sendClick(type, [x, y]);
    }
    ui('Canvas').addEventListener('click', (event) => canvasClick(event, Number(ui('ClickType').value)));
    ui('Canvas').addEventListener('contextmenu', (event) => {
        event.preventDefault();
        canvasClick(event, 2);
    });
    window.addEventListener('resize', draw);
    updateControls();
})();
