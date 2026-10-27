// Run with node --test tests/test_tcp_frontend.cjs; no browser packages are required.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

function setup(t, protocol = 'http:') {
    const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');
    const drawing = [];
    const context = new Proxy({}, {get: (_, name) => (...args) => drawing.push([name, ...args]), set: () => true});
    class Element extends EventTarget {
        constructor() {
            super();
            this.value = '';
            this.textContent = '';
            this.style = {};
            this.classList = {toggle() {}};
            this.files = [];
            this.width = 300;
            this.height = 150;
            this.naturalWidth = 400;
            this.naturalHeight = 300;
        }
        getContext() { return context; }
        getBoundingClientRect() { return {left: 10, top: 20, width: 200, height: 150}; }
        decode() { return Promise.resolve(); }
    }
    const elements = Object.fromEntries([...html.matchAll(/id="(tcp[^"]+)"/g)].map((match) => [match[1], new Element()]));
    const ui = (name) => elements[`tcp${name}`];
    for (const [name, value] of [['Port', '6063'], ['Model', '0'], ['UseGpu', 'true'], ['ClickType', '1']]) ui(name).value = value;
    class FakeWebSocket extends EventTarget {
        static connections = [];
        constructor(url) {
            super();
            this.url = url;
            this.requests = [];
            FakeWebSocket.connections.push(this);
        }
        send(text) { this.requests.push(JSON.parse(text)); }
        message(value) {
            this.dispatchEvent(Object.assign(new Event('message'), {data: JSON.stringify(value)}));
        }
        close() { this.dispatchEvent(new Event('close')); }
    }
    class FakeFileReader {
        readAsDataURL() {
            this.result = 'data:image/png;base64,QUJD';
            this.onload();
        }
    }
    const window = Object.assign(new EventTarget(), {location: {protocol, href: `${protocol}//sam.example:8006/`}});
    vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/tcp_test.js'), 'utf8'), {
        document: {getElementById: (id) => elements[id]}, window,
        WebSocket: FakeWebSocket, FileReader: FakeFileReader, URL,
        setTimeout, clearTimeout, performance,
    });
    t.after(() => FakeWebSocket.connections.forEach((connection) => connection.close()));
    const flush = () => new Promise(setImmediate);
    async function event(name, type = 'click', extra = {}) {
        ui(name).dispatchEvent(Object.assign(new Event(type, {cancelable: true}), extra));
        await flush();
    }
    async function connect(port = '6063') {
        ui('Port').value = port;
        await event('ConnectForm', 'submit');
        const connection = FakeWebSocket.connections.at(-1);
        connection.message({type: 'connected', host: '127.0.0.1', port: Number(port), max_image_bytes: 20 * 1024 * 1024});
        return connection;
    }
    async function reply(connection, data = {status: 1}) {
        connection.message({type: 'response', response: {cmd: connection.requests.at(-1).cmd + 1000, data}});
        await flush();
    }
    async function load(connection) {
        ui('ImageFile').files = [{name: 'image.png', size: 3}];
        await event('ImageFile', 'change');
        await event('LoadImage');
        await reply(connection);
    }
    return {ui, event, connect, reply, load, drawing, flush};
}

test('TCP controls preserve protocol state, scaled clicks, rollback, undo, finish and reset', async (t) => {
    const {ui, event, connect, reply, load, drawing} = setup(t);
    assert.equal(ui('SendClick').disabled, true);
    const connection = await connect();
    assert.equal(connection.url.protocol, 'ws:');
    assert.equal(connection.url.searchParams.get('port'), '6063');
    await event('Models');
    assert.deepEqual(connection.requests.at(-1), {cmd: 8004, data: {}});
    await reply(connection, [{model: 0, model_name: 'Normal model'}]);
    await event('InitModel');
    assert.deepEqual(connection.requests.at(-1), {cmd: 8000, data: {model: 0, use_gpu: true}});
    await reply(connection);
    await load(connection);
    const image = connection.requests.at(-1);
    assert.equal(image.cmd, 8001);
    assert.equal(image.data.image_url, '');
    assert.equal(image.data.image_base64, 'data:image/png;base64,QUJD');
    assert.match(image.data.image_id, /^tcp-[a-z0-9-]+$/);
    assert.equal(ui('Request').textContent.includes('QUJD'), false);
    assert.equal(ui('SendClick').disabled, false);
    await event('Canvas', 'click', {clientX: 60, clientY: 70});
    assert.deepEqual(connection.requests.at(-1).data.coordinate, [100, 100]);
    assert.equal(connection.requests.at(-1).data.image_id, image.data.image_id);
    assert.equal(connection.requests.at(-1).data.click_type, 1);
    assert.equal(ui('Undo').disabled, true);
    const count = connection.requests.length;
    await event('Canvas', 'click', {clientX: 60, clientY: 70});
    assert.equal(connection.requests.length, count, 'A pending request blocks extra clicks');
    const geometry = {status: 1, ploygon: [[[0, 0], [20, 0], [20, 20]]], bnd_boxs: [[0, 0, 21, 21]]};
    await reply(connection, geometry);
    assert.match(ui('ResultInfo').textContent, /1 个点 · 1 个多边形/);
    assert.ok(drawing.some((call) => call[0] === 'fill' && call[1] === 'evenodd'));
    await event('Canvas', 'contextmenu', {clientX: 70, clientY: 80});
    assert.equal(connection.requests.at(-1).data.click_type, 2);
    await reply(connection, {status: 0, message: 'inference failed'});
    assert.equal(ui('Error').textContent, 'inference failed');
    assert.match(ui('ResultInfo').textContent, /1 个点/);
    ui('ClickType').value = '2';
    ui('X').value = '180';
    ui('Y').value = '120';
    await event('ClickForm', 'submit');
    assert.deepEqual(connection.requests.at(-1).data.coordinate, [180, 120]);
    await reply(connection, geometry);
    assert.match(ui('ResultInfo').textContent, /2 个点/);
    for (const [name, type, remaining] of [['Undo', 3, 1], ['Finish', 4, 0], ['Reset', 5, 0]]) {
        await event(name);
        assert.equal(connection.requests.at(-1).data.click_type, type);
        assert.equal('coordinate' in connection.requests.at(-1).data, false);
        await reply(connection, type === 5 ? {status: 1, ploygon: [], bnd_boxs: []} : geometry);
        assert.match(ui('ResultInfo').textContent, new RegExp(`${remaining} 个点`));
    }
    await event('InitModel');
    await reply(connection);
    assert.equal(ui('SendClick').disabled, true, 'Model initialization invalidates the image');
});

test('changing from 6063 to 6062 reconnects and requires a fresh image session', async (t) => {
    const {ui, event, connect, reply, load} = setup(t, 'https:');
    const first = await connect();
    assert.equal(first.url.protocol, 'wss:');
    await load(first);
    const firstId = first.requests.at(-1).data.image_id;
    await event('Models');
    await event('Disconnect');
    assert.equal(ui('SendClick').disabled, true);
    assert.match(ui('Error').textContent, /已断开/);
    const second = await connect('6062');
    assert.equal(second.url.searchParams.get('port'), '6062');
    assert.equal(ui('SendClick').disabled, true);
    await event('LoadImage');
    assert.notEqual(second.requests.at(-1).data.image_id, firstId);
    await reply(second);
    assert.equal(ui('SendClick').disabled, false);
    // Events from an obsolete WebSocket must not replace the new session.
    first.message({type: 'error', message: 'stale error'});
    assert.notEqual(ui('Error').textContent, 'stale error');
});

test('connection errors, mismatched replies and oversized files leave controls usable', async (t) => {
    const {ui, event, connect, reply, flush} = setup(t);
    let connection = await connect();
    connection.message({type: 'error', message: 'Connection refused'});
    connection.close();
    assert.equal(ui('Error').textContent, 'Connection refused');
    assert.equal(ui('Connect').disabled, false);
    connection = await connect();
    ui('ImageFile').files = [{name: 'large.png', size: 21 * 1024 * 1024}];
    await event('ImageFile', 'change');
    assert.match(ui('Error').textContent, /20 MiB/);
    assert.equal(ui('LoadImage').disabled, true);
    await event('Models');
    connection.message({type: 'response', response: {cmd: 9002, data: {status: 1}}});
    await flush();
    assert.match(ui('Error').textContent, /不匹配/);
    assert.equal(ui('Connect').disabled, false);
    connection = await connect('6062');
    await event('Models');
    await reply(connection, []);
    assert.equal(ui('Models').disabled, false);
});
