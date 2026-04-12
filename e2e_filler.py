import asyncio
import uuid
import gzip
import json
import websockets
import os
from dotenv import load_dotenv
import threading
import queue
from audio_manager import get_audio_manager

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "eg") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "eg"))

import protocol

class E2EFillerPlayer:
    def __init__(self):
        load_dotenv()
        self.app_id = os.getenv("DOUBAO_TTS_APP_ID", "")
        self.access_key = os.getenv("DOUBAO_TTS_ACCESS_KEY", "")
        
        # 尝试使用用户请求的复刻音色，如果不被端到端模型支持，再回退
        # 参数必须与 gemini_v2 中的 doubao tts 一致
        self.speaker = os.getenv("DOUBAO_TTS_SPEAKER", "S_XL8NxUsY1")
        self.speed = float(os.getenv("TTS_SPEED", 2.8))
        # emotion/emotion_scale 等暂不在 e2e 接口的文档里常用，但可以尝试传递或保持默认
        self.audio_buffer = bytearray()
        
        self.session_id = str(uuid.uuid4())
        self.ws_url = "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"
        self.headers = {
            "X-Api-App-ID": self.app_id,
            "X-Api-Access-Key": self.access_key,
            "X-Api-Resource-Id": "volc.speech.dialog",
            "X-Api-App-Key": "PlgvMymc7f3tQnJ6",
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        self.audio_queue = queue.Queue()

    async def _run_ws(self, user_question, speaker):
        try:
            async with websockets.connect(self.ws_url, extra_headers=self.headers, ping_interval=None) as ws:
                print("Sending StartConnection")
                start_conn_req = bytearray(protocol.generate_header())
                start_conn_req.extend(int(1).to_bytes(4, 'big'))
                payload_bytes = gzip.compress(b"{}")
                start_conn_req.extend((len(payload_bytes)).to_bytes(4, 'big'))
                start_conn_req.extend(payload_bytes)
                await ws.send(start_conn_req)
                res = await ws.recv()
                print("Received StartConnection response", protocol.parse_response(res))

                # 2. StartSession
                print("Sending StartSession")
                def _doubao_tts_speech_rate_from_speed(speed: float) -> int:
                    # 公式同 gemini_v2
                    v = (float(speed) - 0.5) / 0.05
                    return max(-10, min(100, int(round(v))))
                    
                # 为了让 e2e 垫话放慢，这里我们将语速固定调慢（例如固定传 1.5 对应的 rate，或者给全局 speed 打个折扣）
                # 这里我们将其强制降速，例如设置相当于原速度减慢一倍的值，这里使用 1.0 的慢速
                slow_speed = 1.0
                speech_rate = _doubao_tts_speech_rate_from_speed(slow_speed)
                emotion_scale = float(os.getenv("TTS_EMOTION_SCALE", "5.0"))

                start_session_req = {
                    "tts": {
                        "speaker": speaker,
                        "audio_config": {
                            "channel": 1,
                            "format": "pcm_s16le",
                            "sample_rate": 24000,
                            "speech_rate": speech_rate,
                            "emotion_scale": emotion_scale
                        },
                    },
                    "dialog": {
                        "bot_name": "豆包",
                        "system_role": "1",
                        "character_manifest": "你是一个语音助手。不管用户问什么，你都只回复一句垫话，如“好呢，我这就来分析下当前的对局”、“收到，让我先看看你的阵容，请稍等片刻”等，字数在15个字左右，发言时间控制在3秒左右。不要回答具体问题，只给垫话。",
                        "extra": {
                            "input_mod": "text",
                            "model": "SC2.0"
                        }
                    }
                }
                payload_bytes = gzip.compress(json.dumps(start_session_req).encode('utf-8'))
                start_sess_req = bytearray(protocol.generate_header())
                start_sess_req.extend(int(100).to_bytes(4, 'big'))
                start_sess_req.extend((len(self.session_id)).to_bytes(4, 'big'))
                start_sess_req.extend(self.session_id.encode('utf-8'))
                start_sess_req.extend((len(payload_bytes)).to_bytes(4, 'big'))
                start_sess_req.extend(payload_bytes)
                await ws.send(start_sess_req)
                res = await ws.recv()
                print("Received StartSession response", protocol.parse_response(res))

                data = protocol.parse_response(res)
                if data.get("code", 0) != 0:
                    print(f"[E2E Filler] StartSession error: {data.get('payload_msg')}")
                    return False

                # 3. ChatTextQuery
                print("Sending ChatTextQuery")
                query_payload = {"content": user_question or "分析局势"}
                payload_bytes = gzip.compress(json.dumps(query_payload).encode('utf-8'))
                query_req = bytearray(protocol.generate_header())
                query_req.extend(int(501).to_bytes(4, 'big'))
                query_req.extend((len(self.session_id)).to_bytes(4, 'big'))
                query_req.extend(self.session_id.encode('utf-8'))
                query_req.extend((len(payload_bytes)).to_bytes(4, 'big'))
                query_req.extend(payload_bytes)
                await ws.send(query_req)
                
                # We need to send an empty ChatTextQuery with end=True, or ChatTTSText?
                # According to docs: ChatTextQuery doesn't have start/end fields.
                # Oh wait! We shouldn't use ChatTTSText for end signal, let's just wait for the response of ChatTextQuery!
                # Actually, there's no need to send anything else for text query.
                
                # 4. Receive loop
                print("Entering receive loop")
                first_audio = True
                while True:
                    res = await ws.recv()
                    data = protocol.parse_response(res)
                    if not data: continue
                    
                    if data.get('message_type') == 'SERVER_ACK' and isinstance(data.get('payload_msg'), bytes):
                        if first_audio:
                            print(f"[E2E Filler] Got first audio chunk, length: {len(data['payload_msg'])}")
                            first_audio = False
                        self.audio_queue.put(data['payload_msg'])
                    
                    elif data.get('message_type') == 'SERVER_FULL_RESPONSE':
                        event = data.get('event')
                        print(f"Received SERVER_FULL_RESPONSE with event {event}")
                        if event in [152, 153]: # session finished
                            # Session finished doesn't mean TTS finished, we should wait for TTS ended.
                            # But if the session failed, we might need to break. Let's just log it.
                            print("[E2E Filler] Session finished.")
                            # Don't break here, wait for TTS ended (359)
                        if event == 359: # TTS ended
                            # TTS Ended indicates the server sent all audio chunks
                            # Don't break immediately if there's still audio in queue?
                            # No, the audio is sent via SERVER_ACK, so we just queue None.
                            self.audio_queue.put(None)
                            break
                        if event == 450: # clear cache
                            while not self.audio_queue.empty():
                                try: self.audio_queue.get_nowait()
                                except queue.Empty: break
                    elif data.get('message_type') == 'SERVER_ERROR':
                        print(f"[E2E Filler] SERVER_ERROR: {data}")
                        self.audio_queue.put(None)
                        break

                # 5. Finish
                finish_sess_req = bytearray(protocol.generate_header())
                finish_sess_req.extend(int(102).to_bytes(4, 'big'))
                payload_bytes = gzip.compress(b"{}")
                finish_sess_req.extend((len(self.session_id)).to_bytes(4, 'big'))
                finish_sess_req.extend(self.session_id.encode('utf-8'))
                finish_sess_req.extend((len(payload_bytes)).to_bytes(4, 'big'))
                finish_sess_req.extend(payload_bytes)
                await ws.send(finish_sess_req)
                
                finish_conn_req = bytearray(protocol.generate_header())
                finish_conn_req.extend(int(2).to_bytes(4, 'big'))
                payload_bytes = gzip.compress(b"{}")
                finish_conn_req.extend((len(payload_bytes)).to_bytes(4, 'big'))
                finish_conn_req.extend(payload_bytes)
                await ws.send(finish_conn_req)
                await ws.recv()
                return True

        except Exception as e:
            print(f"[E2E Filler] WebSocket error: {e}")
            self.audio_queue.put(None)
            return False

    def preload(self):
        """
        预加载模式：只发起请求，不自动播放，将音频块全部拉取到队列中缓存。
        """
        def run_async():
            success = asyncio.run(self._run_ws("（用户正在截图并提问，请准备垫话）", self.speaker))
            if not success:
                print(f"[E2E Filler] The speaker {self.speaker} is not supported by the dialog model, falling back to zh_female_suxin_moon_bigtts...")
                self.session_id = str(uuid.uuid4())
                asyncio.run(self._run_ws("（用户正在截图并提问，请准备垫话）", "zh_female_suxin_moon_bigtts"))
            
            # WebSocket 处理完成（正常或异常），通知播放线程退出
            self.audio_queue.put(None)
                
        ws_thread = threading.Thread(target=run_async, daemon=True)
        ws_thread.start()
        
    def play_preloaded(self):
        """兼容旧名字：阻塞直到 WebSocket 完成下载，将数据全部推入全局队列。"""
        am = get_audio_manager()
        self.enqueue_all(am)

    def enqueue_all(self, audio_manager):
        """将队列里的数据推入全局 AudioManager。"""
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None:
                break
            if len(audio_data) > 0:
                audio_manager.play_audio(audio_data)

def preload_filler():
    player = E2EFillerPlayer()
    player.preload()
    return player

if __name__ == "__main__":
    print("Testing E2E filler...")
    player = preload_filler()
    print("Preloading... (waiting 3 seconds)")
    import time
    time.sleep(3)
    print("Playing preloaded audio...")
    player.play_preloaded()
    print("Done.")
