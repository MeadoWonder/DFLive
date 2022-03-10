import os
import sys
import argparse
import ssl
import multiprocessing
import base64
import json

import aiohttp
from aiohttp import web
import cv2


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    sys.path.append((os.getcwd()))
    sys.path.append((os.getcwd() + "/DeepFaceLab"))

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
    parser.add_argument(
        "--model", help="Filename of your dfm model"
    )
    args = parser.parse_args()

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    from server import *
    face_swapper = FaceSwapper(args.model)
    df_detector = DFDetector()


    async def index(request):
        content = open(os.path.join("./server/index.html"), "r").read()
        return web.Response(content_type="text/html", text=content)

    async def javascript(request):
        content = open(os.path.join("./server/client.js"), "r").read()
        return web.Response(content_type="application/javascript", text=content)

    async def echarts(request):
        content = open(os.path.join("./server/echarts.min.js"), "r").read()
        return web.Response(content_type="application/javascript", text=content)

    async def on_shutdown(app):
        face_swapper.terminate()
        print("shutdown...")

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                frame = face_swapper.swap(msg.data)
                rect = [0, 0, 0, 0]
                df_prob = 0
                if frame.rects is not None and len(frame.rects) > 0:
                    rect = frame.rects[0]
                    df_prob = df_detector.detect(frame.img, rect) * 1000
                frame_base64 = 'data:image/jpeg;base64,' + \
                               base64.b64encode(cv2.imencode('.jpg', frame.img)[1]).decode('ascii')
                img_rect_df_json = {'img': frame_base64, 'rect': rect, 'df_prob': int(df_prob)}
                await ws.send_str(json.dumps(img_rect_df_json))
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print('ws connection closed with exception %s' %
                      ws.exception())

        print('websocket connection closed')

        return ws


    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_get("/echarts.min.js", echarts)
    app.add_routes([web.get("/ws", websocket_handler)])
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
