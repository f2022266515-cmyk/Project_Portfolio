"""
DMS Alerts — Voice + Beep alert system.

FIX v2:
  - Global speech queue with a single persistent worker thread.
    Previously a new pyttsx3 engine was created on EVERY alert, which caused:
      • Audio glitching when multiple alerts fired simultaneously
      • Memory accumulation (engines not fully destroyed)
      • Threading races between concurrent pyttsx3 instances
  - All alert messages now go through speak_queued() → single engine speaks them FIFO.
  - winsound.Beep() still fires immediately (synchronous, no thread needed).
  - Queue has maxsize=3: if 3 alerts are already queued, new ones are dropped
    (prevents a speech backlog from a burst of detections).
"""
import winsound
import pyttsx3
import threading
import queue

# ─── GLOBAL SPEECH QUEUE ───────────────────────────────────────────────────────
_speech_queue   = queue.Queue(maxsize=3)
_speech_thread  = None
_speech_lock    = threading.Lock()


def _speech_worker():
    """
    Single background worker that reads from the queue and speaks each message
    sequentially using ONE pyttsx3 engine instance.
    The engine is recreated if it fails (pyttsx3 can crash on backend errors).
    """
    engine = None
    while True:
        try:
            text = _speech_queue.get(timeout=30)   # Wait up to 30s for a message
            if text is None:                        # Poison pill — shutdown signal
                break

            # Lazy-init / re-init engine
            if engine is None:
                engine = pyttsx3.init()
                engine.setProperty('rate', 165)
                engine.setProperty('volume', 1.0)

            print(f"🔊 Speaking: {text}")
            engine.say(text)
            engine.runAndWait()
            _speech_queue.task_done()

        except queue.Empty:
            # No messages for 30 s — dispose engine to free resources
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
                engine = None

        except Exception as e:
            print(f"❌ Speech worker error: {e}")
            # Reset engine on any error
            engine = None
            try:
                _speech_queue.task_done()
            except Exception:
                pass


def _ensure_speech_thread():
    """Start the speech worker thread if it is not running."""
    global _speech_thread
    with _speech_lock:
        if _speech_thread is None or not _speech_thread.is_alive():
            _speech_thread = threading.Thread(target=_speech_worker, daemon=True, name="DMS-Speech")
            _speech_thread.start()


def speak_queued(text: str):
    """
    Queue a speech message. Non-blocking.
    If queue is full (3 messages pending), the new message is silently dropped
    to avoid a speech backlog.
    """
    _ensure_speech_thread()
    try:
        _speech_queue.put_nowait(text)
    except queue.Full:
        print(f"⚠️  Speech queue full — dropped: {text}")


# ─── PUBLIC API ────────────────────────────────────────────────────────────────

def play_alert_sound(alert_type: str, name: str = ""):
    """
    Fire the appropriate beep + voice alert for the given alert type.

    Supported types:
      LOGIN, FATIGUE, SMOKING, PHONE, DISTRACTION, SEATBELT,
      UNAUTHORIZED, YAWN
    """
    try:
        if alert_type == "LOGIN":
            # Friendly welcome — no beep
            speak_queued(f"Welcome, {name}. Drive safely.")

        elif alert_type == "FATIGUE":
            winsound.Beep(1000, 600)
            speak_queued("Warning! Driver drowsiness detected. Please wake up.")

        elif alert_type == "YAWN":
            winsound.Beep(800, 300)
            speak_queued("You appear fatigued. Consider taking a break.")

        elif alert_type == "SMOKING":
            winsound.Beep(800, 400)
            speak_queued("Warning! Smoking detected inside the vehicle.")

        elif alert_type == "PHONE":
            winsound.Beep(1200, 500)
            speak_queued("Warning! Mobile phone usage detected. Please focus on the road.")

        elif alert_type == "DISTRACTION":
            winsound.Beep(1500, 300)
            speak_queued("Warning! Please focus on the road ahead.")

        elif alert_type == "SEATBELT":
            winsound.Beep(900, 700)
            speak_queued("Warning! Please fasten your seatbelt immediately.")

        elif alert_type == "UNAUTHORIZED":
            winsound.Beep(500, 1000)
            speak_queued("Warning! Unauthorized driver detected.")

        else:
            print(f"⚠️  Unknown alert type: {alert_type}")

    except Exception as e:
        print(f"❌ Alert error ({alert_type}): {e}")
