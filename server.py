import os
import sys
import argparse
import ssl
import multiprocessing

import aiohttp
from aiohttp import web


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    sys.path.append((os.getcwd() + "/DeepFaceLab"))

    from server import *
    face_swapper = FaceSwapper()

    async def index(request):
        content = open(os.path.join(ROOT, "index.html"), "r").read()
        return web.Response(content_type="text/html", text=content)

    async def javascript(request):
        content = open(os.path.join(ROOT, "client.js"), "r").read()
        return web.Response(content_type="application/javascript", text=content)

    async def on_shutdown(app):
        face_swapper.terminate()
        print("shutdown...")

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                frame_base64 = face_swapper.swap(msg.data)
                await ws.send_str(frame_base64)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print('ws connection closed with exception %s' %
                      ws.exception())

        print('websocket connection closed')

        return ws

    parser = argparse.ArgumentParser(
        description="video demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=10001, help="Port for HTTP server"
    )
    args = parser.parse_args()

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.add_routes([web.get("/ws", websocket_handler)])
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
