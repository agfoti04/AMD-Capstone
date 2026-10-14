"""Persistent browser controls and demand-driven latest-frame delivery."""
import asyncio
import contextlib
import json
import logging
from urllib.parse import urlparse

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web


def create_app(session, page):
    app = web.Application(client_max_size=2048)
    sockets = set()
    driver = None
    client = None

    def same_origin(request):
        origin = request.headers.get('Origin')
        if origin and urlparse(origin).netloc != request.host:
            raise web.HTTPForbidden(text='Use the local rover dashboard')

    async def startup(app):
        nonlocal client
        client = ClientSession(timeout=ClientTimeout(total=0.25))

    async def command(linear, angular):
        values = session.drive_values(linear, angular)
        async with client.post(session.bridge + '/command', json=values) as response:
            response.raise_for_status()
            await response.read()

    async def cleanup(app):
        for ws in list(sockets):
            await ws.close(code=1001, message=b'Server shutting down')
        with contextlib.suppress(Exception):
            await command(0, 0)
        await client.close()

    async def index(request):
        return web.Response(body=page, content_type='text/html', headers={'Cache-Control': 'no-store'})

    async def status(request):
        return web.json_response(session.snapshot(), headers={'Cache-Control': 'no-store'})

    async def controls(request):
        nonlocal driver
        same_origin(request)
        if session.snapshot()['control_mode'] == 'gazebo':
            raise web.HTTPForbidden(text='Restart the simulator in browser mode')
        if driver is not None:
            raise web.HTTPConflict(text='Another browser tab has the driving controls')
        ws = web.WebSocketResponse(max_msg_size=2048, compress=False)
        driver = ws
        last_seq = -1
        try:
            await ws.prepare(request)
            sockets.add(ws)
            while not session.stop.is_set():
                message = await ws.receive(timeout=0.5)
                if message.type != WSMsgType.TEXT:
                    break
                data = json.loads(message.data)
                seq = data['seq']
                if isinstance(seq, bool) or not isinstance(seq, int) or seq <= last_seq:
                    raise ValueError('Commands must have increasing sequence numbers')
                last_seq = seq
                await command(data['linear'], data['angular'])
                await ws.send_json({'seq': seq, 'ok': True})
        except asyncio.CancelledError:
            pass
        except Exception as error:
            logging.getLogger(__name__).warning('Control connection stopped: %s: %s', type(error).__name__, error)
        finally:
            with contextlib.suppress(Exception):
                await command(0, 0)
            sockets.discard(ws)
            await ws.close()
            if driver is ws:
                driver = None
        return ws

    async def view(request):
        same_origin(request)
        ws = web.WebSocketResponse(max_msg_size=128, compress=False)
        await ws.prepare(request)
        sockets.add(ws)
        version = -1
        try:
            while not session.stop.is_set():
                message = await ws.receive(timeout=5)
                if message.type != WSMsgType.TEXT or message.data != 'next':
                    break
                deadline = asyncio.get_running_loop().time() + 1
                while not session.stop.is_set():
                    with session.condition:
                        current, jpeg = session.version, session.jpeg
                        metadata = session.snapshot()
                    if jpeg is not None and current != version:
                        version = current
                        await ws.send_json(dict(metadata, type='frame'))
                        await ws.send_bytes(jpeg)
                        break
                    if asyncio.get_running_loop().time() >= deadline:
                        await ws.send_json(dict(metadata, type='status'))
                        break
                    await asyncio.sleep(0.005)
                # Request the next frame only after displaying this one: a slow
                # viewer cannot accumulate an old video queue.
        except (Exception, asyncio.CancelledError):
            pass
        finally:
            sockets.discard(ws)
            await ws.close()
        return ws

    app.add_routes([web.get('/', index), web.get('/status', status),
                    web.get('/control', controls), web.get('/view', view)])
    app.on_startup.append(startup)
    app.on_shutdown.append(cleanup)
    return app


def serve(session, page, port):
    web.run_app(create_app(session, page), host='127.0.0.1', port=port,
                print=None, access_log=None, shutdown_timeout=1)
