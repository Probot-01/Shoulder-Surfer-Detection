# run_demo.py
# ============================================================================
# AI SHOULDER SURFING PREVENTION SYSTEM — Demo Launcher
# ============================================================================
# This is the ONLY file the examiner needs to run:
#
#   python run_demo.py
#
# It checks that everything is in place, then launches main_system.py.
# ============================================================================

import sys
import os
import time
import cv2


# ============================================================================
# REQUIRED FILES — edit these if your filenames differ
# ============================================================================

REQUIRED_FILES = {
    "best_head_pose_model.pth" : "Trained head pose model   (run biwi_train.py in Colab first)",
    "yolov8n.pt"               : "YOLOv8n weights           (run download_yolo.py to get it)",
    "head_pose_predictor.py"   : "Head pose predictor module",
    "decision_engine.py"       : "Threat decision engine module",
    "face_crop_utils.py"       : "Face crop utility module",
    "main_system.py"           : "Main system script",
}


# ============================================================================
# COLOUR CODES (works in most terminals on Windows 10+, macOS, Linux)
# ============================================================================

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {RED}{msg}{RESET}")
def warn(msg): print(f"  {YELLOW}!{RESET}  {YELLOW}{msg}{RESET}")
def info(msg): print(f"     {msg}")


# ============================================================================
# CHECKS
# ============================================================================

def check_python_version():
    """Python 3.8+ required for all dependencies."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        fail(f"Python 3.8+ required. You have {major}.{minor}.")
        return False
    ok(f"Python {major}.{minor} detected")
    return True


def check_required_files():
    """Verify every required file exists in the current directory."""
    all_ok = True
    for filename, description in REQUIRED_FILES.items():
        if os.path.isfile(filename):
            size_kb = os.path.getsize(filename) / 1024
            ok(f"{filename:<35} ({size_kb:,.0f} KB)")
        else:
            fail(f"{filename} NOT FOUND")
            info(f"→ {description}")
            all_ok = False
    return all_ok


def check_dependencies():
    """Try importing every key library and report any that are missing."""
    deps = {
        "cv2"         : "pip install opencv-python",
        "torch"       : "pip install torch torchvision",
        "torchvision" : "pip install torchvision",
        "ultralytics" : "pip install ultralytics",
        "mediapipe"   : "pip install mediapipe",
        "PIL"         : "pip install Pillow",
        "numpy"       : "pip install numpy",
    }
    all_ok = True
    for module, install_cmd in deps.items():
        try:
            __import__(module)
            ok(f"{module}")
        except ImportError:
            fail(f"{module} not installed  →  {install_cmd}")
            all_ok = False
    return all_ok


def check_webcam():
    """Open and immediately release the default webcam to confirm it works."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        fail("Webcam (device 0) could not be opened")
        info("→ Check that your webcam is plugged in and not used by another app")
        info("→ Try closing Zoom, Teams, or any browser with camera access")
        return False

    # Read one test frame to make sure video data is actually flowing
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        fail("Webcam opened but could not read a frame")
        info("→ Your webcam driver may need reinstalling")
        return False

    h, w = frame.shape[:2]
    ok(f"Webcam connected  ({w}×{h} resolution)")
    return True


def check_model_loadable():
    """
    Attempt to instantiate HeadPosePredictor to confirm the .pth file is valid.
    This catches corrupted or incompatible model files before the demo starts.
    """
    try:
        # Import here (not at top) so a missing dependency gives a clear message
        from head_pose_predictor import HeadPosePredictor
        predictor = HeadPosePredictor("best_head_pose_model.pth")
        ok("Head pose model loaded and verified")
        return True
    except FileNotFoundError:
        fail("best_head_pose_model.pth not found (see above)")
        return False
    except Exception as e:
        fail(f"Model failed to load: {e}")
        info("→ The .pth file may be corrupted or from an incompatible architecture")
        return False


def check_yolo():
    """Load YOLOv8n from disk (does NOT trigger a download if file exists)."""
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")
        ok("YOLOv8n model ready")
        return True
    except Exception as e:
        fail(f"YOLO failed to load: {e}")
        info("→ Run:  python download_yolo.py")
        return False


# ============================================================================
# BANNER
# ============================================================================

def print_banner():
    banner = f"""
{CYAN}{BOLD}
  ==========================================
    AI SHOULDER SURFING PREVENTION SYSTEM
    Version 1.0  |  College Project Demo
  ==========================================
{RESET}"""
    print(banner)


def print_section(title):
    print(f"\n{BOLD}{title}{RESET}")
    print("  " + "─" * 45)


# ============================================================================
# LAUNCH
# ============================================================================

def launch_main():
    """
    Imports and calls main() from main_system.py directly.
    This avoids spawning a subprocess, so all output stays in one terminal
    and Ctrl+C works cleanly.
    """
    import importlib
    try:
        main_module = importlib.import_module("main_system")
        if hasattr(main_module, "main"):
            main_module.main()
        else:
            # main_system.py runs its code at module level (no main() function)
            # Importing it is enough to execute it.
            pass
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n{RED}Runtime error in main_system.py: {e}{RESET}")
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():

    print_banner()

    all_passed = True

    # ---- Check 1: Python version ----
    print_section("System Check")
    if not check_python_version():
        all_passed = False

    # ---- Check 2: Required files ----
    print_section("Required Files")
    if not check_required_files():
        all_passed = False

    # ---- Check 3: Dependencies ----
    print_section("Python Dependencies")
    if not check_dependencies():
        all_passed = False

    # ---- Check 4: Webcam ----
    print_section("Hardware")
    if not check_webcam():
        all_passed = False

    # ---- Check 5: Model and YOLO (only if files exist) ----
    print_section("AI Models")
    if os.path.isfile("best_head_pose_model.pth"):
        if not check_model_loadable():
            all_passed = False
    else:
        warn("Skipping model load check (file missing — see above)")

    if os.path.isfile("yolov8n.pt"):
        if not check_yolo():
            all_passed = False
    else:
        warn("Skipping YOLO load check (file missing — see above)")

    # ---- Final decision ----
    print()
    print("  " + "─" * 45)

    if not all_passed:
        print(f"\n{RED}{BOLD}  PRE-FLIGHT CHECK FAILED{RESET}")
        print(f"  {RED}Fix the issues above and run again.{RESET}\n")
        sys.exit(1)

    print(f"\n{GREEN}{BOLD}  ALL CHECKS PASSED — System is ready{RESET}")
    print()

    # ---- Countdown ----
    for i in range(3, 0, -1):
        print(f"  Starting in {BOLD}{i}{RESET}...", end="\r")
        time.sleep(1)

    print(f"  {GREEN}{BOLD}Launching...{RESET}                     ")
    print()

    # ---- Launch ----
    launch_main()


if __name__ == "__main__":
    main()
