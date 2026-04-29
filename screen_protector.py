# screen_protector.py
# ============================================================================
# PURPOSE:
#   Displays a full-screen blurred overlay when a shoulder-surfing threat is
#   detected. A "spotlight" circle follows the cursor so the USER can still
#   see and use the area right around their mouse, while the rest of the
#   screen is obscured from anyone looking over their shoulder.
#
# HOW TO USE (from main_system.py):
#   from screen_protector import ScreenProtector
#   protector = ScreenProtector()
#   protector.start()                 # create the tkinter window (call once)
#
#   protector.activate_protection()   # show blur → call when THREAT detected
#   protector.deactivate_protection() # hide blur → call when SAFE again
#
# PRESS 'q' in the YOLO window to quit — it calls protector.stop() for you.
# ============================================================================

import sys
import platform
import threading
import tkinter as tk
from tkinter import Canvas

from PIL import Image, ImageFilter, ImageTk, ImageDraw
import numpy as np

# Optional: faster screenshot library.  pip install mss
# Falls back to PIL ImageGrab if mss is not installed.
try:
    import mss
    _HAS_MSS = True
except ImportError:
    _HAS_MSS = False

# Windows-only: ctypes lets us call Win32 API functions to make the window
# transparent AND click-through.
if platform.system() == "Windows":
    import ctypes
    import ctypes.wintypes


# ============================================================================
# CONSTANTS
# ============================================================================

BLUR_RADIUS     = 25     # Gaussian blur strength (higher = more obscured)
SPOTLIGHT_RADIUS = 80    # Radius (px) of the clear circle around the cursor
POLL_INTERVAL_MS = 50    # How often (ms) to update cursor position
FALLBACK_BANNER_H = 60   # Height of the red warning banner (fallback mode)


# ============================================================================
# ScreenProtector CLASS
# ============================================================================

class ScreenProtector:
    """
    Full-screen blurred overlay with a spotlight that follows the cursor.

    Modes:
        HIDDEN    → overlay invisible, user sees normal screen
        PROTECTED → blurred screenshot displayed; spotlight reveals cursor area

    The window is always-on-top and click-through on Windows (the user can
    still click their actual applications through the overlay).  On systems
    where click-through cannot be applied, a fallback red warning banner is
    shown at the top of the screen instead.
    """

    def __init__(self):
        # Current state: "hidden" or "protected"
        self.state = "hidden"

        # The main tkinter root window (created in start())
        self.root  = None

        # Canvas widget — we draw everything here
        self.canvas = None

        # Tkinter-compatible image reference (must be kept alive or it gets GC'd)
        self.tk_image = None

        # Screen dimensions (filled in during start())
        self.screen_w = 0
        self.screen_h = 0

        # Whether click-through succeeded on this machine
        self.click_through_ok = False

        # ID returned by root.after() so we can cancel the cursor-polling loop
        self._poll_job = None

        # Thread lock — tkinter must only be touched from the main thread,
        # but main_system.py might call activate/deactivate from its loop thread.
        self._lock = threading.Lock()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def start(self):
        """
        Creates the tkinter window.  MUST be called from the MAIN thread.
        Call this once at startup, before the YOLO main loop begins.
        """
        self.root = tk.Tk()

        # ---- Detect screen dimensions ----
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        # ---- Configure the window ----
        # Remove the title bar and border (we want a bare overlay)
        self.root.overrideredirect(True)

        # Size and position: cover the entire screen starting at (0, 0)
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")

        # Always stay on top of every other window
        self.root.wm_attributes("-topmost", True)

        # Start fully invisible — we reveal it on activate_protection()
        self.root.wm_attributes("-alpha", 0.0)

        # ---- Create the canvas (the drawing surface) ----
        # bg="black" is the fallback if no image is loaded yet
        self.canvas = Canvas(
            self.root,
            width=self.screen_w, height=self.screen_h,
            bg="black", highlightthickness=0
        )
        self.canvas.pack()

        # ---- Apply click-through (Windows) ----
        self._apply_click_through()

        # ---- Platform-specific transparency ----
        if platform.system() == "Darwin":   # macOS
            self.root.wm_attributes("-transparent", True)
            self.root.config(bg="systemTransparent")

        print(f"[ScreenProtector] Window ready: {self.screen_w}x{self.screen_h}")
        print(f"[ScreenProtector] Click-through: {self.click_through_ok}")

    def activate_protection(self):
        """
        Captures the screen, applies blur, displays the overlay, and starts
        tracking the cursor so the spotlight follows it.
        """
        if self.root is None:
            print("[ScreenProtector] ERROR: call start() first.")
            return

        print("[ScreenProtector] Activating protection...")
        self.state = "protected"

        # Capture + build the overlay image on a background thread so we don't
        # stall the YOLO main loop while PIL does the blur work.
        t = threading.Thread(target=self._build_and_show_overlay, daemon=True)
        t.start()

    def deactivate_protection(self):
        """
        Hides the overlay — user sees their normal screen again.
        """
        self.state = "hidden"

        # Cancel the cursor-polling loop if it is running
        if self._poll_job is not None:
            self.root.after_cancel(self._poll_job)
            self._poll_job = None

        # Make the window invisible (alpha=0 means 100% transparent)
        self.root.after(0, lambda: self.root.wm_attributes("-alpha", 0.0))
        print("[ScreenProtector] Protection deactivated.")

    def update_spotlight(self, cursor_x, cursor_y):
        """
        Moves the spotlight circle to the given screen coordinates.
        Called automatically by the cursor-polling loop; you can also
        call it manually if you already know where the cursor is.

        Parameters:
            cursor_x, cursor_y (int): absolute screen pixel coordinates
        """
        if self.state != "protected" or self._base_blurred is None:
            return

        # Schedule the redraw on the tkinter main thread
        # (Canvas operations must always happen on the main thread)
        self.root.after(0, lambda: self._redraw_spotlight(cursor_x, cursor_y))

    def stop(self):
        """
        Destroys the tkinter window. Call this when the program exits.
        """
        if self.root is not None:
            self.root.after(0, self.root.destroy)
            self.root = None

    def update(self):
        """
        Drives the tkinter event loop for one tick.
        Call this from your main loop every frame:
            protector.update()
        This must be called from the SAME thread that called start().
        """
        if self.root is not None:
            try:
                self.root.update()
            except tk.TclError:
                pass   # Window was destroyed — ignore

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _apply_click_through(self):
        """
        Makes the window click-through on Windows using the Win32 API.

        HOW DOES CLICK-THROUGH WORK ON WINDOWS?
          Every window has a set of "extended style flags" stored by Windows.
          Two flags are relevant:
            WS_EX_LAYERED     → enables transparency effects for this window
            WS_EX_TRANSPARENT → makes mouse clicks pass THROUGH the window
                                 to whatever is underneath it

          We retrieve the current flags, OR in the new ones, then write them
          back using SetWindowLong().

          GetWindowLong / SetWindowLong are Win32 functions. We call them via
          Python's ctypes library, which lets Python call C functions in DLLs.
        """
        if platform.system() != "Windows":
            return

        try:
            # Constants from the Windows SDK headers
            GWL_EXSTYLE       = -20           # Index for extended window style
            WS_EX_LAYERED     = 0x00080000    # Required for transparency
            WS_EX_TRANSPARENT = 0x00000020    # Makes clicks pass through

            # Get the native window handle (HWND) for our tkinter window.
            # winfo_id() returns it as an integer.
            hwnd = self.root.winfo_id()

            # Read the current extended style flags
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)

            # Add LAYERED and TRANSPARENT flags using bitwise OR
            new_style = style | WS_EX_LAYERED | WS_EX_TRANSPARENT

            # Write the updated style back to Windows
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

            self.click_through_ok = True
            print("[ScreenProtector] Click-through applied (WS_EX_TRANSPARENT).")

        except Exception as e:
            # If this fails (e.g. permissions, wrong Python build), fall back
            # to the warning banner mode instead.
            self.click_through_ok = False
            print(f"[ScreenProtector] Click-through FAILED: {e}")
            print("[ScreenProtector] Will use fallback banner mode.")

    def _capture_screen(self):
        """
        Takes a screenshot of the entire screen and returns it as a PIL Image.

        We prefer `mss` (pip install mss) because it is much faster than
        PIL ImageGrab (especially on Windows with multiple monitors).
        Falls back to PIL ImageGrab if mss is not installed.
        """
        if _HAS_MSS:
            with mss.mss() as sct:
                # sct.monitors[0] is the "all monitors combined" virtual screen.
                # sct.monitors[1] is the primary monitor only — use that.
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                # mss returns raw bytes in BGRA format; convert to PIL RGB
                img = Image.frombytes(
                    "RGB",
                    (screenshot.width, screenshot.height),
                    screenshot.rgb   # rgb property strips the alpha channel
                )
        else:
            # PIL ImageGrab.grab() is simpler but slower (~3× slower on Windows)
            from PIL import ImageGrab
            img = ImageGrab.grab()

        return img

    def _build_and_show_overlay(self):
        """
        Runs in a background thread:
          1. Captures the current screen
          2. Creates a heavily blurred version
          3. Hands the result to the main thread for display

        We store two images:
          self._base_original : unblurred (used to create the clear spotlight)
          self._base_blurred  : blurred   (the default view outside spotlight)
        """
        # Step 1: Capture
        original = self._capture_screen()
        original = original.resize((self.screen_w, self.screen_h), Image.LANCZOS)

        # Step 2: Blur
        # ImageFilter.GaussianBlur(radius) — higher radius = more blur.
        # radius=25 is very strong: text becomes completely unreadable.
        blurred = original.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

        # Store both for later use in update_spotlight()
        self._base_original = original
        self._base_blurred  = blurred

        # Step 3: Hand off to the main thread via root.after(0, ...)
        # root.after(delay_ms, func) schedules func to run on the main thread.
        # delay=0 means "as soon as possible".
        self.root.after(0, self._show_overlay_on_main_thread)

    def _show_overlay_on_main_thread(self):
        """
        Called on the main tkinter thread after the blur is ready.
        Displays the blurred image and starts cursor tracking.
        """
        if self.state != "protected":
            return   # User deactivated before the thread finished

        if self.click_through_ok or platform.system() == "Darwin":
            # Full-screen blurred overlay — click-through works
            self._redraw_spotlight(
                self.root.winfo_pointerx(),
                self.root.winfo_pointery()
            )
            # Make the window visible (alpha=1.0 = fully opaque)
            self.root.wm_attributes("-alpha", 1.0)
        else:
            # Fallback: click-through failed → show a red banner only
            self._show_fallback_banner()
            self.root.wm_attributes("-alpha", 0.85)

        # Start the cursor-tracking loop
        self._start_cursor_poll()

    def _redraw_spotlight(self, cursor_x, cursor_y):
        """
        Redraws the full overlay with a clear circle at (cursor_x, cursor_y).

        HOW THE SPOTLIGHT WORKS:
          We start with the blurred image as the base. Then we "punch a hole"
          by compositing in the unblurred original just within the circle.

          To make the edge soft (gradual transition, not hard cut), we create
          a circular mask that is:
            - Pure white (255) inside the spotlight → show original pixels
            - Pure black (0) far outside the circle → show blurred pixels
            - Smooth gradient at the boundary

          PIL Image.composite(img1, img2, mask) blends two images using the
          mask: white pixels show img1 (original), black pixels show img2 (blurred).

        Parameters:
            cursor_x, cursor_y: absolute screen coords of the cursor
        """
        if self._base_blurred is None or self._base_original is None:
            return

        w, h = self.screen_w, self.screen_h
        r    = SPOTLIGHT_RADIUS

        # ---- Build the spotlight mask ----
        # A greyscale image — white circle on black background
        mask = Image.new("L", (w, h), 0)   # Start all black
        draw = ImageDraw.Draw(mask)

        # Draw a white-filled ellipse (circle) at the cursor position
        draw.ellipse(
            [cursor_x - r, cursor_y - r,
             cursor_x + r, cursor_y + r],
            fill=255
        )

        # Soften the edge of the mask with a blur — this creates the gradient
        # transition between the spotlight and the blurred region.
        # A small blur (radius=18) gives a gentle, natural-looking fade.
        mask = mask.filter(ImageFilter.GaussianBlur(radius=18))

        # ---- Composite original + blurred using the mask ----
        # Where mask=255 (white, inside spotlight): show original
        # Where mask=0   (black, outside):          show blurred
        composite = Image.composite(self._base_original, self._base_blurred, mask)

        # ---- Convert to tkinter-compatible format and display ----
        # ImageTk.PhotoImage wraps a PIL image so tkinter's Canvas can use it.
        # We MUST keep a reference (self.tk_image) — if it gets garbage collected,
        # the image disappears from the canvas even though the canvas item stays.
        self.tk_image = ImageTk.PhotoImage(composite)

        # Delete the old canvas image item (if any) and draw the new one
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def _show_fallback_banner(self):
        """
        Fallback for when click-through cannot be applied.
        Shows a semi-transparent red banner at the top of the screen instead
        of the full-screen blur, so the user can still interact with their apps.
        """
        self.canvas.delete("all")
        bh = FALLBACK_BANNER_H
        w  = self.screen_w

        # Red background rectangle
        self.canvas.create_rectangle(0, 0, w, bh, fill="#cc0000", outline="")

        # Warning text
        self.canvas.create_text(
            w // 2, bh // 2,
            text="⚠  THREAT DETECTED — Shoulder Surfing Alert!  ⚠",
            fill="white",
            font=("Helvetica", 18, "bold")
        )

        # Resize window to show only the banner (not full screen)
        self.root.geometry(f"{w}x{bh}+0+0")

    def _start_cursor_poll(self):
        """
        Polls the cursor position every POLL_INTERVAL_MS milliseconds and
        redraws the spotlight to follow it.

        WHY NOT USE A MOUSEMOVE EVENT?
          tkinter's bind("<Motion>") only fires when the mouse moves over
          the tkinter window itself. Since our window is click-through, the
          mouse events go to other windows — tkinter never sees them.
          Instead, we use winfo_pointerx/y to read the cursor position from
          the OS directly, and schedule repeated calls with root.after().
        """
        def poll():
            if self.state != "protected":
                return   # Protection was deactivated — stop polling

            # Read cursor position in absolute screen coordinates
            x = self.root.winfo_pointerx()
            y = self.root.winfo_pointery()

            self._redraw_spotlight(x, y)

            # Schedule the next poll after POLL_INTERVAL_MS milliseconds
            # Store the job ID so we can cancel it in deactivate_protection()
            self._poll_job = self.root.after(POLL_INTERVAL_MS, poll)

        # Kick off the first poll immediately
        self._poll_job = self.root.after(0, poll)


# ============================================================================
# STANDALONE DEMO — runs when you execute this file directly
# ============================================================================
# Tests the overlay in isolation (without YOLO / MediaPipe).
# Press ENTER in the terminal to toggle protection on/off.
# Close the terminal (Ctrl+C) to exit.

if __name__ == "__main__":
    import time

    print("=" * 55)
    print("  ScreenProtector — Standalone Demo")
    print("  Press ENTER to toggle protection ON/OFF")
    print("  Press Ctrl+C to exit")
    print("=" * 55)

    protector = ScreenProtector()
    protector.start()

    # Run a simple toggle loop in a background thread so we can read input
    # while tkinter's event loop runs on the main thread.
    is_on = False

    def input_loop():
        global is_on
        while True:
            input()   # Block until the user presses ENTER
            if is_on:
                protector.deactivate_protection()
                print("Protection OFF — screen visible")
            else:
                protector.activate_protection()
                print("Protection ON  — screen blurred")
            is_on = not is_on

    t = threading.Thread(target=input_loop, daemon=True)
    t.start()

    # Run the tkinter event loop on the main thread
    try:
        while True:
            protector.update()
            time.sleep(0.01)   # ~100 fps for the overlay UI
    except KeyboardInterrupt:
        protector.stop()
        print("\nDemo exited.")
