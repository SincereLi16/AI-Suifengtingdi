import argparse
import os
import threading
import time
from pathlib import Path
from dotenv import load_dotenv
import random

# 引入 gemini_v2 中现成的纯 TTS 处理逻辑
from gemini_v2 import TTSPlayer
from audio_manager import get_audio_manager

class E2EFillerPlayerV2:
    def __init__(self):
        load_dotenv()
        
        # 构造虚拟的 args，以便复用 gemini_v2 里面配置好的 TTSPlayer
        args = argparse.Namespace()
        
        # 为了让垫话稍微慢一点，我们给语速打个折扣
        # e.g., 如果全局 speed 是 2.8，这里可能固定成 1.5 左右，或者从环境变量读
        base_speed = float(os.getenv("TTS_SPEED", 2.8))
        args.tts_speed = max(1.0, base_speed - 1.0)  # 垫话减慢 1.0 倍速
        
        # 其他参数默认从环境变量中被 TTSPlayer 读取
        self.tts_player = TTSPlayer(args)
        
        # 由于我们需要推入全局队列，所以将 TTSPlayer 的本地播放静音
        self.tts_player.play = False
        self.tts_player.save = False
        
        self.audio_queue = []
        self.is_preloaded = False

        # 固定的垫话库，随机抽取一句，长度故意大于 15 个字以防过短被某些模型嫌弃
        self.filler_sentences = [
            "niania，我来给你看看。",
            "哈基星你这蠢货，又在捣鼓笋喵幺蛾子。",
            "哈基星你玩个集贸呀，又要老八抬走了。"
        ]

    def preload(self):
        """
        预加载模式：发起普通 HTTP TTS 请求，将音频流块全部拉取到缓存中。
        """
        self.preloaded_sentence = random.choice(self.filler_sentences)
        print(f"[E2E Filler V2] 预加载垫话: {self.preloaded_sentence}")
        
        def run_fetch():
            # 强行指定格式为 pcm，直接拿纯音频流
            self.tts_player.doubao_audio_format = "pcm"
            
            parts = []
            def _on_chunk(chunk: bytes):
                parts.append(chunk)
                
            try:
                # 发起阻塞的 HTTP POST 请求，按 chunk 下载 PCM 数据
                resp = self.tts_player._doubao_tts_request(self.preloaded_sentence, "pcm")
                if resp and resp.status_code < 400:
                    self.tts_player._consume_doubao_tts_stream(resp, on_pcm_chunk=_on_chunk)
                    self.audio_queue = parts
                    self.is_preloaded = True
                    print(f"[E2E Filler V2] 预加载完成，共拉取 {len(parts)} 块音频。")
                else:
                    print(f"[E2E Filler V2] HTTP 请求失败: {resp.status_code if resp else 'None'}")
            except Exception as e:
                print(f"[E2E Filler V2] 预加载异常: {e}")
                
        fetch_thread = threading.Thread(target=run_fetch, daemon=True)
        fetch_thread.start()
        
        # 对于预加载，我们其实应该阻塞等它完成（或者在播放时阻塞），
        # 因为我们是普通 HTTP，速度极快，这里直接 join() 保证一定能拿到音频。
        fetch_thread.join()

    def enqueue_all(self, audio_manager):
        """将缓存里的数据推入全局 AudioManager。"""
        if not self.is_preloaded:
            print("[E2E Filler V2] 警告：尝试推送未预加载完成的音频！")
            
        for audio_data in self.audio_queue:
            if audio_data and len(audio_data) > 0:
                audio_manager.play_audio(audio_data)
        
        # 推送完毕后清空，释放内存
        self.audio_queue.clear()

def preload_filler():
    player = E2EFillerPlayerV2()
    player.preload()
    return player

if __name__ == "__main__":
    print("Testing E2E filler V2...")
    player = preload_filler()
    print("Playing preloaded audio...")
    am = get_audio_manager()
    player.enqueue_all(am)
    print("Done. (Audio is playing in background thread)")
    time.sleep(5)
