# -*- coding: utf-8 -*-
"""
单链路最优速度版：pipeline_v2 (Tkinter 极简无依赖版)
结合 screenshot_agent 的 GUI、pipeline 的分析流程、gemini_v2 的流式语音与 TTS。
"""

import threading
import time
import os
import shutil
import pyautogui
import ctypes
import sys
import queue
import argparse
import tempfile
import json
import math
from pathlib import Path
import numpy as np

import tkinter as tk

# ==== 配置路径 ====
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_manager import get_audio_manager

# ==== 导入业务模块 ====
import pipeline as pl
import player_onnx as pon
import element_recog.equip_column_recog as ecr
import trait_cross_validate as tcv
from element_recog import chess_recog as cr
from project_paths import DEFAULT_OUT_BATTLE_PIPELINE_V3

import gemini_v2
from gemini_v2 import TTSPlayer, run_coach_v2_after_summary, _build_argparser, _asr_ws_stream_run

# 禁用/启用用户输入 (鼠标和键盘) 的 Windows API
user32 = ctypes.windll.user32
pyautogui.PAUSE = 0.01

# ==== 配置区域 ====
OUTPUT_DIR = PROJECT_ROOT / "screen"
CROP_BOX = (152, 80, 2360, 1340)
START_X, START_Y = 305, 740
END_X, END_Y = 305, 315
DRAG_DURATION = 0.2

# ==== 全局模型缓存 ====
_OCR_ENGINE = None
_COL_TEMPLATES = None
_COL_TIER_COUNTS = None
_MODELS_LOADED = False
_MODELS_LOCK = threading.Lock()

def init_models():
    global _OCR_ENGINE, _COL_TEMPLATES, _COL_TIER_COUNTS, _MODELS_LOADED
    with _MODELS_LOCK:
        if _MODELS_LOADED: return
        print("[PipelineV2] 正在加载模型至内存...", flush=True)
        
        # 隐藏 ONNX Runtime C++ 层报错 (ERROR=3, FATAL=4)
        try:
            import onnxruntime as ort
            ort.set_default_logger_severity(4)
        except Exception:
            pass

        pl._ensure_fightboard_mobilenet_runtime()
        _pon_name, pon_kwargs = pl._configure_pipeline_devices()

        _OCR_ENGINE = pon.create_ocr_engine(pon_kwargs)
        _COL_TEMPLATES, _COL_TIER_COUNTS = ecr.build_tiered_templates(PROJECT_ROOT / "equip_gallery")
        _MODELS_LOADED = True
        print("[PipelineV2] 模型加载完毕，系统就绪！", flush=True)

def clear_output_dir():
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                pass
    else:
        os.makedirs(OUTPUT_DIR)

class GuiVoiceSession:
    """自动开始录音，由 GUI 按钮或回车结束。"""
    SR = 16000

    def __init__(self):
        self.segments = []
        self.recording = False
        self._chunks = []
        self._lock = threading.Lock()
        self._audio_q = queue.Queue(maxsize=512)
        self._asr_stop = threading.Event()
        self._asr_ws_thread = None
        self._asr_ws_holder = {}
        self._stream = None
        self._listener = None
        self._submit = threading.Event()
        self._live_log_ts = 0.0

    def _resolve_asr_creds(self):
        return "3491963725", "dqbpiqoB26POmsIca81QFzhWX25N6rdS"

    def _live_asr_log(self, text):
        now = time.perf_counter()
        if now - self._live_log_ts < 0.25:
            return
        self._live_log_ts = now
        t = (text or "").replace("\n", " ")
        if len(t) > 100:
            t = t[:100] + "…"
        sys.stdout.write(f"\r\x1b[2K[听写] {t}")
        sys.stdout.flush()

    def start(self):
        import sounddevice as sd
        from pynput import keyboard

        self._submit.clear()
        self.segments = []
        self.recording = True
        self._chunks.clear()

        def audio_cb(indata, frames, tcb, status):
            if not self.recording: return
            chunk = indata.copy().ravel()
            with self._lock:
                self._chunks.append(chunk)
            if self._asr_ws_thread is not None and self._asr_ws_thread.is_alive():
                try:
                    self._audio_q.put_nowait(chunk)
                except queue.Full:
                    try:
                        self._audio_q.get_nowait()
                    except queue.Empty: pass
                    try:
                        self._audio_q.put_nowait(chunk)
                    except queue.Full: pass

        self._stream = sd.InputStream(
            samplerate=self.SR,
            channels=1,
            dtype="float32",
            callback=audio_cb,
            blocksize=1024,
        )
        self._stream.start()

        def on_press(key):
            if key == keyboard.Key.enter:
                self.stop()

        try:
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
        except Exception as e:
            print(f"[PipelineV2] 键盘监听器启动失败: {e}", flush=True)

        # 启动 WebSocket
        self._asr_stop.clear()
        self._asr_ws_holder.clear()
        app_key, access = self._resolve_asr_creds()
        cluster = "volcengine_streaming_common"

        def _run_ws():
            try:
                self._asr_ws_holder["t"] = _asr_ws_stream_run(
                    audio_q=self._audio_q,
                    stop_ev=self._asr_stop,
                    appid=app_key,
                    token=access,
                    cluster=cluster,
                    uid=app_key[:32],
                    sample_rate=self.SR,
                    chunk_ms=200,
                    workflow="audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate",
                    live_log=self._live_asr_log,
                    verbose=False
                )
            except Exception as e:
                self._asr_ws_holder["e"] = e

        self._asr_ws_thread = threading.Thread(target=_run_ws, daemon=True)
        self._asr_ws_thread.start()
        print("[语音] 已自动开始录音（回车或点击界面结束）...", flush=True)

    def stop(self):
        with self._lock:
            if not self.recording: return
            self.recording = False
        self._asr_stop.set()
        if self._asr_ws_thread is not None and self._asr_ws_thread.is_alive():
            self._asr_ws_thread.join(timeout=10.0)
        self._asr_ws_thread = None

        raw_ws = str(self._asr_ws_holder.get("t") or "").strip()
        if raw_ws:
            try:
                import zhconv
                text_final = zhconv.convert(raw_ws, "zh-cn")
            except ImportError:
                text_final = raw_ws
            self.segments.append(text_final)
        
        self._submit.set()

    def wait_for_result(self):
        self._submit.wait()
        try:
            self._listener.stop()
        except: pass
        if self._stream:
            self._stream.stop()
            self._stream.close()
        
        out = "".join(self.segments).strip()
        print(f"\n[识别] {out}", flush=True)
        return out

# ==== 绘图助手 ====

def cubic_bezier(p0, p1, p2, p3, steps=20):
    points = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        points.extend([x, y])
    return points

def get_arc_points(cx, cy, r, start_deg, extent_deg, steps=10):
    points = []
    for i in range(steps + 1):
        angle = math.radians(start_deg + (extent_deg * i / steps))
        x = cx + r * math.cos(angle)
        y = cy - r * math.sin(angle)
        points.extend([x, y])
    return points

def get_rounded_rect_points(w, h, inset=1.5, tl=30, tr=40, br=20, bl=30):
    pts = []
    pts.extend(get_arc_points(w - inset - tr, inset + tr, tr, 90, -90))
    pts.extend(get_arc_points(w - inset - br, h - inset - br, br, 0, -90))
    pts.extend(get_arc_points(inset + bl, h - inset - bl, bl, 270, -90))
    pts.extend(get_arc_points(inset + tl, inset + tl, tl, 180, -90))
    return pts

def get_cloud_points():
    pts = []
    # 调整为适中的尺寸，比最初的原始尺寸大一点，但比之前 1.2 倍版本小
    # 以 (280, 200) 附近为视觉中心进行收缩
    pts.extend(cubic_bezier((120, 180), (120, 90), (220, 60), (280, 90)))
    pts.extend(cubic_bezier((280, 90), (340, 60), (450, 90), (450, 180)))
    pts.extend(cubic_bezier((450, 180), (500, 210), (460, 290), (400, 290)))
    pts.extend(cubic_bezier((400, 290), (360, 340), (220, 340), (180, 290)))
    pts.extend(cubic_bezier((180, 290), (90, 290), (90, 210), (120, 180)))
    return pts

class CanvasButton:
    def __init__(self, canvas, x, y, w, h, text, radii, font, command=None):
        self.canvas = canvas
        self.command = command
        self.enabled = True
        self.pts = get_rounded_rect_points(w, h, 1.5, *radii)
        self.pts = [p + x if i % 2 == 0 else p + y for i, p in enumerate(self.pts)]
        
        # 增加 splinesteps 参数，默认为 12，提高到 36 进一步提升抗锯齿质量
        self.poly = self.canvas.create_polygon(self.pts, fill="white", outline="black", width=2, smooth=True, splinesteps=36)
        self.text = self.canvas.create_text(x + w/2, y + h/2, text=text, fill="black", font=font, justify="center")
        
        for item in (self.poly, self.text):
            self.canvas.tag_bind(item, "<Enter>", self.on_enter)
            self.canvas.tag_bind(item, "<Leave>", self.on_leave)
            self.canvas.tag_bind(item, "<Button-1>", self.on_click)
            self.canvas.tag_bind(item, "<ButtonRelease-1>", self.on_release)

    def set_fill(self, color):
        self.canvas.itemconfig(self.poly, fill=color)

    def on_enter(self, event):
        if self.enabled:
            self.set_fill("#f0f0f0")
            self.canvas.config(cursor="hand2")

    def on_leave(self, event):
        if self.enabled:
            self.set_fill("white")
            self.canvas.config(cursor="")

    def on_click(self, event):
        if self.enabled:
            self.set_fill("#e0e0e0")

    def on_release(self, event):
        if self.enabled:
            self.set_fill("white")
            if self.command:
                self.command()

    def set_state(self, state):
        self.enabled = state
        if not state:
            self.set_fill("#e0e0e0")
        else:
            self.set_fill("white")

    def hide(self):
        self.canvas.itemconfig(self.poly, state="hidden")
        self.canvas.itemconfig(self.text, state="hidden")

    def show(self):
        self.canvas.itemconfig(self.poly, state="normal")
        self.canvas.itemconfig(self.text, state="normal")

class PipelineAgentGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("随风听笛 - 鹅鹅姐")
        
        # 添加代码级别开关，控制多轮对话是否播放垫话
        self.ENABLE_MULTI_TURN_FILLER = False
        
        # --- 桌面宠物化配置 ---
        self.geometry("550x400")
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        
        self.trans_color = '#abcdef'
        self.config(bg=self.trans_color)
        self.attributes('-transparentcolor', self.trans_color)

        self.canvas = tk.Canvas(self, width=550, height=400, bg=self.trans_color, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # 绘制云朵背景，开启平滑参数，提高 splinesteps 到 36 让边缘更细腻
        self.canvas.create_polygon(get_cloud_points(), fill="white", outline="black", width=3, smooth=True, splinesteps=36)

        # 拖拽状态
        self.canvas.bind("<Button-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.do_drag)
        
        # 加载图片，使用 subsample(2, 2) 将图片缩小一半
        png_path = PROJECT_ROOT / "鹅鹅姐" / "鹅鹅姐透明.png"
        gif_path = PROJECT_ROOT / "鹅鹅姐" / "鹅鹅姐透明.gif"
        
        self.frames = []
        if gif_path.exists():
            try:
                i = 0
                while True:
                    frame = tk.PhotoImage(file=str(gif_path), format=f"gif -index {i}")
                    # 缩小一半
                    self.frames.append(frame.subsample(2, 2))
                    i += 1
            except tk.TclError:
                pass
                
        if png_path.exists():
            img = tk.PhotoImage(file=str(png_path))
            self.idle_pixmap = img.subsample(2, 2)
        else:
            self.idle_pixmap = self.frames[0] if self.frames else None

        # 居中在云朵中间，再向下移动一点，从 Y=170 移动到 Y=190
        self.pet_image_item = self.canvas.create_image(280, 190, image=self.idle_pixmap)
        self.animating = False

        # 打字机文字气泡
        self.typewriter_timer_active = False
        self.typewriter_text = ""
        self.typewriter_idx = 0
        
        # 调试模式
        self.debug_var = tk.BooleanVar(value=False)
        self.debug_chk = tk.Checkbutton(
            self.canvas, text="Debug Mode", variable=self.debug_var, 
            bg="white", fg="#888888", font=("Microsoft YaHei", 8, "bold"), 
            activebackground="white", activeforeground="black", highlightthickness=0, bd=0
        )
        # 将复选框往左边挪一点，X 从 240 调整为 220
        self.canvas.create_window(220, 260, window=self.debug_chk, anchor="nw")

        # 按钮
        font_go = ("幼圆", 16, "bold")
        font_small = ("幼圆", 12, "bold")
        # Go 按钮向右上角移动，从 (120, 160) 移动到 (380, 100) 附近
        self.btn_go = CanvasButton(self.canvas, 380, 100, 60, 60, "Go", (30,30,30,30), font_go, command=self.on_btn_click)
        # 将 Go on 和 Quit 向中间聚拢，Go on 往右移到 130，Quit 往左移到 350
        self.btn_continue = CanvasButton(self.canvas, 130, 160, 70, 50, "Go on", (25,15,25,15), font_small, command=self.on_continue_click)
        self.btn_quit = CanvasButton(self.canvas, 350, 160, 70, 50, "Quit", (15,25,15,25), font_small, command=self.on_end_click)
        
        self.btn_continue.hide()
        self.btn_quit.hide()

        self.state = "IDLE"
        self.voice_session = None
        self.question = ""
        self.summary_path = None
        self.pipeline_done = threading.Event()
        self.voice_done = threading.Event()
        self.multi_turn_event = threading.Event()
        
        # 线程安全的 UI 更新队列
        self.ui_queue = queue.Queue()
        self.after(100, self.process_ui_queue)
        
        # 预加载模型
        threading.Thread(target=init_models, daemon=True).start()

    def process_ui_queue(self):
        try:
            while True:
                func, args, kwargs = self.ui_queue.get_nowait()
                func(*args, **kwargs)
        except queue.Empty:
            pass
        self.after(100, self.process_ui_queue)

    def run_in_main_thread(self, func, *args, **kwargs):
        self.ui_queue.put((func, args, kwargs))

    def start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def do_drag(self, event):
        x = self.winfo_x() + (event.x - self._drag_x)
        y = self.winfo_y() + (event.y - self._drag_y)
        self.geometry(f"+{x}+{y}")

    def trigger_bounce(self):
        x = self.winfo_x()
        y = self.winfo_y()
        self.geometry(f"+{x}+{y-10}")
        self.after(100, lambda: self.geometry(f"+{x}+{y+5}"))
        self.after(200, lambda: self.geometry(f"+{x}+{y}"))

    def start_gif_animation(self):
        if len(self.frames) > 1 and not self.animating:
            self.animating = True
            self.frame_idx = 0
            self._play_gif()

    def _play_gif(self):
        if self.animating:
            self.frame_idx = (self.frame_idx + 1) % len(self.frames)
            self.canvas.itemconfig(self.pet_image_item, image=self.frames[self.frame_idx])
            # 加快帧率，30fps，约 33ms 一帧
            self.gif_timer = self.after(33, self._play_gif)

    def stop_gif_animation(self):
        self.animating = False
        if hasattr(self, 'gif_timer'):
            self.after_cancel(self.gif_timer)
        self.canvas.itemconfig(self.pet_image_item, image=self.idle_pixmap)

    def show_fixed_speech(self, text):
        # 停止打字机
        self.typewriter_timer_active = False
        if hasattr(self, 'typewriter_timer_id'):
            self.after_cancel(self.typewriter_timer_id)
        
        if not text.strip():
            self.hide_speech()
            return
            
        self._update_bubble_geometry(text)

    def show_speech(self, text):
        # 恢复使用内部的打字机特效，不再依赖底层同步回调
        if hasattr(self, 'typewriter_timer_id'):
            self.after_cancel(self.typewriter_timer_id)
        self.typewriter_text = text
        self.typewriter_idx = 0
        
        if not text.strip():
            self.hide_speech()
            return
            
        self._update_bubble_geometry("")
        self._typewriter_step()

    def _append_speech_ui(self, text_chunk):
        # 修复分句消失太快的问题：
        # 大模型的文字下发速度远快于 TTS 的语音播放速度（和人眼阅读速度）。
        # 如果简单地按标点符号截断 `typewriter_text`，就会导致刚收到前两句就瞬间被清空，
        # 直接跳到了最后一句。
        #
        # 正确做法：
        # 我们只负责把收到的 `text_chunk` 追加到一个“待显示队列/完整文本”中，
        # 让打字机 `_typewriter_step` 自己根据速度慢慢打字。
        # 当打字机打到一个标点符号时，暂停一下，如果气泡字数过多，在“打字机”层面去清屏。
        
        self.typewriter_text += text_chunk

        if not self.typewriter_timer_active:
            if hasattr(self, 'bubble_poly'):
                self.canvas.itemconfig(self.bubble_poly, state="normal")
                self.canvas.itemconfig(self.bubble_text, state="normal")
            # 增加一个 0.8 秒的启动延迟，让字幕比语音稍微晚一点点出来
            self.typewriter_timer_active = True
            self.after(800, self._typewriter_step)

    def _update_bubble_geometry(self, current_text):
        if not current_text:
            return
        lines = current_text.count('\n') + 1
        chars = len(current_text)
        # 增加每行字数容忍度，放大气泡
        estimated_lines = max(lines, chars // 18 + 1)
        
        # 加大高度与内边距，提升松弛感
        new_height = max(70, estimated_lines * 26 + 30)
        w = 380  # 加宽气泡
        h = new_height
        x = 80   # 整体向左微调，居中展示
        y = 80 - h
        
        pts = get_rounded_rect_points(w, h, 1.5, 30, 40, 20, 30)
        pts = [p + x if i % 2 == 0 else p + y for i, p in enumerate(pts)]
        
        if hasattr(self, 'bubble_poly'):
            self.canvas.coords(self.bubble_poly, *pts)
            self.canvas.coords(self.bubble_text, x + w/2, y + h/2)
            self.canvas.itemconfig(self.bubble_text, text=current_text, width=w-40)
            self.canvas.itemconfig(self.bubble_poly, state="normal")
            self.canvas.itemconfig(self.bubble_text, state="normal")
        else:
            self.bubble_poly = self.canvas.create_polygon(pts, fill="white", outline="black", width=3, smooth=True, splinesteps=36)
            self.bubble_text = self.canvas.create_text(x + w/2, y + h/2, text=current_text, fill="black", font=("Microsoft YaHei", 12, "bold"), justify="center", width=w-40)

    def _typewriter_step(self):
        self.typewriter_timer_active = True
        if self.typewriter_idx < len(self.typewriter_text):
            next_char = self.typewriter_text[self.typewriter_idx]
            self.typewriter_idx += 1
            
            # 不再清屏，永远从 0 开始截取，实现整段文字的打字机累加效果
            current_text = self.typewriter_text[0:self.typewriter_idx].strip()
            
            if current_text:
                self._update_bubble_geometry(current_text)
            
            # 打字速度适中，不再对标点做极长停顿，因为是整个气泡展示
            delay = 120
            if next_char in ['，', ',', '。', '！', '？', '.', '!', '?', '；', ';', '\n']:
                delay = 400
            
            self.typewriter_timer_id = self.after(delay, self._typewriter_step)
        else:
            self.typewriter_timer_active = False

    def hide_speech(self):
        # 如果打字机还在运行，不要立刻隐藏，而是推迟隐藏
        if getattr(self, 'typewriter_timer_active', False):
            # 500ms 后再次尝试隐藏
            self.after(500, self.hide_speech)
            return
            
        if hasattr(self, 'bubble_poly'):
            self.canvas.itemconfig(self.bubble_poly, state="hidden")
            self.canvas.itemconfig(self.bubble_text, state="hidden")

    def force_hide_speech(self):
        """强制隐藏气泡，不管打字机是否完成"""
        self.typewriter_timer_active = False
        if hasattr(self, 'typewriter_timer_id'):
            self.after_cancel(self.typewriter_timer_id)
        if hasattr(self, 'bubble_poly'):
            self.canvas.itemconfig(self.bubble_poly, state="hidden")
            self.canvas.itemconfig(self.bubble_text, state="hidden")

    def update_btn(self, text, bg_color="#4CAF50", state=True):
        self.run_in_main_thread(self.show_fixed_speech, text)
        self.run_in_main_thread(self._update_btn_ui, text, state)
        
    def _update_btn_ui(self, text, state):
        if not state:
            self.btn_go.hide()
            self.btn_continue.hide()
            self.btn_quit.hide()
        else:
            self.btn_go.show()
            self.btn_continue.hide()
            self.btn_quit.hide()

    def show_multi_turn_btns(self):
        self.run_in_main_thread(self._show_multi_turn_btns_ui)
        
    def _show_multi_turn_btns_ui(self):
        self.stop_gif_animation()
        self.show_speech("要问鹅鹅什么？")
        self.btn_go.hide()
        self.btn_continue.show()
        self.btn_quit.show()

    def _sync_subtitle_callback(self, text):
        """恢复打字机入口：AudioManager 触发时，将文本推入打字机并开始播放"""
        if text is None:
            self.run_in_main_thread(self.hide_speech)
        else:
            # AudioManager 收到下一段字幕前，先强制清理上一次可能没打完的旧状态
            self.run_in_main_thread(self.force_hide_speech)
            self.run_in_main_thread(self.show_speech, text)

    def append_speech(self, text_chunk):
        # 兼容接口，实际不再使用
        pass

    def on_continue_click(self):
        self.trigger_bounce()
        self.state = "MULTI_RECORDING"
        self.update_btn("鹅鹅听你讲...", bg_color="#F44336", state=True)
        self.run_in_main_thread(self.show_fixed_speech, "鹅鹅正在听...")
        
        self.btn_go.hide()
        self.btn_continue.hide()
        self.btn_quit.hide()
        
        # 预加载 e2e 垫话，多轮对话也需要！
        # 为了不阻塞主线程（避免 UI 卡顿），将其放到后台线程预加载
        def preload_multi_filler():
            try:
                # 默认使用纯 TTS 版垫话引擎，音色绝对稳定
                import e2e_filler_v2
                self.multi_filler_player = e2e_filler_v2.preload_filler()
            except Exception as e:
                print(f"[PipelineV2] 多轮预加载 V2 垫话失败，尝试回退 V1: {e}")
                try:
                    import e2e_filler
                    self.multi_filler_player = e2e_filler.preload_filler()
                except Exception as inner_e:
                    print(f"[PipelineV2] 多轮预加载 V1 垫话失败: {inner_e}")
                    self.multi_filler_player = None
                    
        threading.Thread(target=preload_multi_filler, daemon=True).start()
            
        self.voice_done.clear()
        self.voice_session = GuiVoiceSession()
        self.voice_session.start()
        threading.Thread(target=self.multi_voice_worker, daemon=True).start()

    def on_end_click(self):
        self.trigger_bounce()
        self.state = "IDLE"
        self.update_btn("Go", bg_color="#4CAF50", state=True)
        self.stop_gif_animation()
        self.hide_speech()
        self.multi_turn_event.set()

    def on_btn_click(self):
        self.trigger_bounce()
        if self.state == "REPLYING":
            self.state = "IDLE"
            self.multi_turn_event.set()
            self.update_btn("请再次点击出击", bg_color="#4CAF50", state=True)
            self.hide_speech()
            return

        if self.state == "IDLE":
            self._played_beep = False
            self.state = "SCREENSHOTTING"
            self.update_btn("鹅鹅出击中...", bg_color="#FF9800", state=False)
            threading.Thread(target=self.run_workflow, daemon=True).start()
        elif self.state == "RECORDING" or self.state == "MULTI_RECORDING":
            self.state = "STOPPING_RECORD"
            self.update_btn("正在整理思路...", bg_color="#F44336", state=False)
            self.start_gif_animation()
            if self.voice_session:
                self.voice_session.stop()

    def multi_voice_worker(self):
        self.question = self.voice_session.wait_for_result()
        self.voice_done.set()
        
        self.run_in_main_thread(self.hide_speech)
        self.run_in_main_thread(self.start_gif_animation)
        self.update_btn("鹅鹅构思战术...", bg_color="#9C27B0", state=False)
        
        # ASR语音输入完成，交由 AudioManager 统一管理
        def play_multi_filler():
            am = get_audio_manager()
            am.set_subtitle_callback(self._sync_subtitle_callback)
            am.play_silence(0.5)
            print("[PipelineV2] 多轮播放第一声 Beep")
            am.play_beep(freq=800)
            am.play_silence(1.0)
            
            # beep声完毕后，根据开关决定是否播放e2e的tts回复，并同步显示垫话字幕
            if self.ENABLE_MULTI_TURN_FILLER:
                if hasattr(self, 'multi_filler_player') and self.multi_filler_player:
                    if hasattr(self.multi_filler_player, 'preloaded_sentence'):
                        am.show_subtitle(self.multi_filler_player.preloaded_sentence)
                    self.multi_filler_player.enqueue_all(am)
            else:
                print("[PipelineV2] 开关已关闭，多轮对话跳过垫话播报")
                
            self.multi_turn_event.set()
            
        # 注意：这里我们不再让 play_multi_filler 直接 set event 并自己跑完
        # 因为我们后续逻辑依赖于 filler 垫话排队完成。
        # 但我们发现多轮的 follow_up 是在 custom_reader 里 wait 的。
        self.multi_filler_thread = threading.Thread(target=play_multi_filler, daemon=True)
        self.multi_filler_thread.start()

    def run_workflow(self):
        # 预加载首轮垫话（放在后台线程避免卡顿主线程 UI）
        self.first_turn_filler_player = None
        def preload_first_filler():
            try:
                import e2e_filler_v2
                self.first_turn_filler_player = e2e_filler_v2.preload_filler()
            except Exception as e:
                print(f"[PipelineV2] 预加载 V2 垫话语音失败，尝试回退 V1: {e}")
                try:
                    import e2e_filler
                    self.first_turn_filler_player = e2e_filler.preload_filler()
                except Exception as inner_e:
                    print(f"[PipelineV2] 预加载 V1 垫话语音失败: {inner_e}")
                    
        threading.Thread(target=preload_first_filler, daemon=True).start()

        try:
            time.sleep(0.1)
            clear_output_dir()
            
            if self.debug_var.get():
                debug_img_dir = Path(r"C:\Users\Administrator\Desktop\随风听笛6\对局截图")
                if debug_img_dir.exists():
                    for f in debug_img_dir.glob("*.png"):
                        shutil.copy2(f, OUTPUT_DIR)
                    for f in debug_img_dir.glob("*.jpg"):
                        shutil.copy2(f, OUTPUT_DIR)
                print(f"[PipelineV2] 调试模式：已从 {debug_img_dir} 读取图片", flush=True)
            else:
                try: user32.BlockInput(True)
                except: pass

                img_a = pyautogui.screenshot()
                cropped_a = img_a.crop(CROP_BOX)
                save_path_a = os.path.join(OUTPUT_DIR, "01-a.png")
                cropped_a.save(save_path_a)
                
                pyautogui.moveTo(START_X, START_Y, duration=0)
                time.sleep(0.01)
                pyautogui.mouseDown(button='left')
                pyautogui.moveTo(END_X, END_Y, duration=0.15)
                pyautogui.mouseUp(button='left')
                
                time.sleep(0.5)
                
                img_b = pyautogui.screenshot()
                cropped_b = img_b.crop(CROP_BOX)
                save_path_b = os.path.join(OUTPUT_DIR, "01-b.png")
                cropped_b.save(save_path_b)
                
                time.sleep(0.05)
                pyautogui.moveTo(END_X, END_Y, duration=0)
                time.sleep(0.01)
                pyautogui.mouseDown(button='left')
                pyautogui.moveTo(START_X, START_Y, duration=0.15)
                pyautogui.mouseUp(button='left')

                try: user32.BlockInput(False)
                except: pass

            self.state = "RECORDING"
            self.update_btn("鹅鹅听你讲...", bg_color="#F44336", state=True)
            self.run_in_main_thread(self.show_fixed_speech, "鹅鹅正在听...")
            # 在这里不要调用 start_gif_animation
            
            self.voice_done.clear()
            self.pipeline_done.clear()
            self.voice_session = GuiVoiceSession()
            self.voice_session.start()

            threading.Thread(target=self.voice_worker, daemon=True).start()
            threading.Thread(target=self.pipeline_worker, daemon=True).start()

            # 等待语音输入完成（回车）
            self.voice_done.wait()
            
            # 语音输入完成，隐藏固定气泡，开始播放 GIF
            self.run_in_main_thread(self.hide_speech)
            self.run_in_main_thread(self.start_gif_animation)
            
            def play_preloaded_filler():
                am = get_audio_manager()
                am.set_subtitle_callback(self._sync_subtitle_callback)
                am.play_silence(0.5)
                print("[PipelineV2] 播放第一声 Beep")
                am.play_beep(freq=800)
                am.play_silence(1.0)
                
                # beep声完毕后，推入e2e的tts回复，并同步显示垫话字幕
                if self.first_turn_filler_player:
                    if hasattr(self.first_turn_filler_player, 'preloaded_sentence'):
                        am.show_subtitle(self.first_turn_filler_player.preloaded_sentence)
                    self.first_turn_filler_player.enqueue_all(am)
            
            filler_thread = threading.Thread(target=play_preloaded_filler, daemon=True)
            filler_thread.start()

            if self.state == "STOPPING_RECORD" or self.state == "RECORDING":
                self.state = "ANALYZING"
                self.update_btn("鹅鹅冥思苦想中...", bg_color="#2196F3", state=False)
            
            self.pipeline_done.wait()
            
            if not self.summary_path:
                raise RuntimeError("战术快报生成失败，请检查终端日志。")

            try:
                debug_dir = PROJECT_ROOT / "debug_history" / time.strftime("%Y%m%d_%H%M%S")
                debug_dir.mkdir(parents=True, exist_ok=True)
                for f in os.listdir(OUTPUT_DIR):
                    shutil.copy2(os.path.join(OUTPUT_DIR, f), debug_dir)
                for f in os.listdir(DEFAULT_OUT_BATTLE_PIPELINE_V3):
                    if f.endswith(".json") or f.endswith(".png"):
                        shutil.copy2(os.path.join(DEFAULT_OUT_BATTLE_PIPELINE_V3, f), debug_dir)
                print(f"[PipelineV2] 本回合截图与分析数据已备份至: {debug_dir}")
            except Exception as e:
                print(f"[PipelineV2] 备份调试数据失败: {e}")

            self.state = "THINKING"
            self.update_btn("鹅鹅构思战术...", bg_color="#9C27B0", state=False)

            if not self.question:
                self.question = "（分析局势并给出建议）"

            ap = _build_argparser()
            args = ap.parse_args([])
            args.summary_json = self.summary_path
            
            tts = TTSPlayer(args)

            def tts_callback(text_chunk, turn, is_first_chunk=True):
                if is_first_chunk:
                    # 等待 e2e 垫话完全压入音频队列
                    filler_thread.join()
                    
                    if self.state == "THINKING":
                        self.state = "REPLYING"
                self.append_speech(text_chunk)
                tts.speak(text_chunk, turn, is_first_chunk)
            tts_callback.__self__ = tts

            def custom_reader():
                self.show_multi_turn_btns()
                self.multi_turn_event.clear()
                self.multi_turn_event.wait()
                
                if self.state == "IDLE":
                    return "q"
                    
                if self.state == "MULTI_RECORDING" or self.state == "STOPPING_RECORD":
                    self.voice_done.wait()
                    # 确保多轮的垫话语音（e2e）已经完全被压入音频队列
                    if hasattr(self, 'multi_filler_thread') and self.multi_filler_thread.is_alive():
                        self.multi_filler_thread.join()
                    
                    self.update_btn("鹅鹅构思战术...", bg_color="#9C27B0", state=False)
                    return self.question
                    
                return "q"

            run_coach_v2_after_summary(
                args,
                self.summary_path,
                self.question,
                threading.Lock(),
                follow_up_reader=custom_reader,
                on_answer=tts_callback,
            )

        except Exception as e:
            try: user32.BlockInput(False)
            except: pass
            print(f"执行失败：{e}")
        finally:
            self.state = "IDLE"
            self.update_btn("Go", bg_color="#4CAF50", state=True)
            self.run_in_main_thread(self.stop_gif_animation)

    def voice_worker(self):
        self.question = self.voice_session.wait_for_result()
        self.voice_done.set()

    def pipeline_worker(self):
        try:
            out_root = DEFAULT_OUT_BATTLE_PIPELINE_V3
            if out_root.exists():
                shutil.rmtree(out_root, ignore_errors=True)
            out_root.mkdir(parents=True, exist_ok=True)
            
            import sys
            old_argv = sys.argv
            sys.argv = [
                "pipeline.py", 
                "--img-dir", str(OUTPUT_DIR), 
                "--out", str(out_root)
            ]
            
            init_models()
            
            original_create_ocr = pon.create_ocr_engine
            original_build_templates = ecr.build_tiered_templates
            pon.create_ocr_engine = lambda kwargs: _OCR_ENGINE
            ecr.build_tiered_templates = lambda p: (_COL_TEMPLATES, _COL_TIER_COUNTS)
            
            try:
                import repo_sys_path
                pl.main()
            finally:
                pon.create_ocr_engine = original_create_ocr
                ecr.build_tiered_templates = original_build_templates
                sys.argv = old_argv
            
            self.summary_path = list(out_root.glob("*_summary.json"))[0]
        except Exception as e:
            print(f"[PipelineV2] 分析失败: {e}")
        finally:
            self.pipeline_done.set()

if __name__ == "__main__":
    app = PipelineAgentGUI()
    app.geometry("+500+200")
    app.mainloop()
