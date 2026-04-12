import queue
import threading
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

class AudioManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AudioManager, cls).__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self.q = queue.Queue()
        self.sample_rate = 24000
        self.stream = None
        self.subtitle_callback = None  # 用于通知 UI 更新字幕的回调函数
        if sd is not None:
            try:
                self.stream = sd.RawOutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='int16'
                )
                self.stream.start()
            except Exception as e:
                print(f"[AudioManager] 初始化 sounddevice 失败: {e}")
                
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        while True:
            try:
                item = self.q.get()
                if item is None:
                    break
                
                if isinstance(item, bytes):
                    # Raw PCM data
                    if self.stream:
                        self.stream.write(item)
                elif isinstance(item, dict):
                    cmd = item.get("type")
                    if cmd == "beep":
                        self._play_beep(item.get("duration", 0.1), item.get("freq", 800))
                    elif cmd == "sleep":
                        self._play_silence(item.get("duration", 0.5))
                    elif cmd == "subtitle":
                        # 处理字幕指令，调用 UI 回调
                        text = item.get("text", "")
                        if self.subtitle_callback:
                            try:
                                self.subtitle_callback(text)
                            except Exception as cb_e:
                                print(f"[AudioManager] 字幕回调异常: {cb_e}")
                    elif cmd == "subtitle_clear":
                        if self.subtitle_callback:
                            try:
                                self.subtitle_callback(None)
                            except Exception as cb_e:
                                print(f"[AudioManager] 字幕清空回调异常: {cb_e}")
            except Exception as e:
                print(f"[AudioManager] Worker 播放异常: {e}")

    def _play_beep(self, duration=0.1, freq=800):
        if not self.stream:
            return
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        # 生成正弦波并适当放大音量
        wave = np.sin(freq * 2 * np.pi * t) * 4000
        audio_data = wave.astype(np.int16).tobytes()
        self.stream.write(audio_data)

    def _play_silence(self, duration=0.5):
        if not self.stream:
            return
        zero_data = np.zeros(int(self.sample_rate * duration), dtype=np.int16).tobytes()
        self.stream.write(zero_data)

    def play_audio(self, pcm_data: bytes):
        """播放原始 PCM 音频块"""
        self.q.put(pcm_data)

    def play_beep(self, duration=0.1, freq=800):
        """播放提示音 (Beep)"""
        self.q.put({"type": "beep", "duration": duration, "freq": freq})

    def play_silence(self, duration=0.5):
        """播放静音（精确延时，不阻塞线程）"""
        self.q.put({"type": "sleep", "duration": duration})

    def show_subtitle(self, text: str):
        """将一段字幕压入播放队列，当音频执行到此处时，将通知 UI 更新。"""
        self.q.put({"type": "subtitle", "text": text})

    def clear_subtitle(self):
        """清空当前气泡字幕"""
        self.q.put({"type": "subtitle_clear"})

    def set_subtitle_callback(self, callback):
        """设置字幕回调函数，接收一个 str 参数（None表示清空）"""
        self.subtitle_callback = callback

# 全局单例获取方法
def get_audio_manager():
    return AudioManager()
