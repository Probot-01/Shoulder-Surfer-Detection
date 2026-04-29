# decision_engine.py
# ============================================================================
# PURPOSE:
#   Converts per-frame model predictions into stable, reliable threat alerts.
#   The key technique used here is called "temporal smoothing".
#
# WHAT IS TEMPORAL SMOOTHING?
#   A neural network predicts on EACH individual frame independently.
#   At 20+ FPS, even a perfect model will occasionally misclassify a frame
#   due to motion blur, lighting changes, or a person briefly looking away.
#
#   Without smoothing:
#       Frame 1: SAFE → Frame 2: THREAT → Frame 3: SAFE → Frame 4: THREAT
#       This causes rapid flickering alerts — useless and annoying.
#
#   With temporal smoothing:
#       We only DECLARE a threat after N consecutive THREAT frames.
#       We only CLEAR a threat after M consecutive SAFE frames.
#       This filters out momentary glitches while staying responsive to real threats.
#
#   Think of it like a doctor: one sneeze doesn't mean you're sick.
#   But 10 sneezes in a row? That's worth paying attention to.
#
# USAGE:
#   from decision_engine import ThreatDecisionEngine
#   engine = ThreatDecisionEngine(threat_threshold=10, cooldown_frames=30)
#   state  = engine.update(num_persons=2, observer_results=[...])
# ============================================================================


class ThreatDecisionEngine:
    """
    Converts raw per-frame detection results into a stable SAFE/THREAT state.

    Two parameters control sensitivity:
        threat_threshold  : consecutive THREAT frames needed to trigger an alert
        cooldown_frames   : consecutive SAFE frames needed to clear the alert

    Making threat_threshold smaller  → faster to trigger (more sensitive, more false alarms)
    Making cooldown_frames  larger   → slower to clear  (persistent alerts)
    """

    def __init__(self, threat_threshold=10, cooldown_frames=30):
        """
        Initialises the engine in a safe, neutral state.

        Parameters:
            threat_threshold (int): How many consecutive THREAT frames must
                                    occur before flipping to THREAT state.
                                    Default 10 ≈ 0.5 seconds at 20 FPS.

            cooldown_frames  (int): How many consecutive SAFE frames must
                                    occur before returning to SAFE state.
                                    Default 30 ≈ 1.5 seconds at 20 FPS.
                                    Longer cooldown = alert lingers longer
                                    after the observer looks away.
        """

        # Store the thresholds for use in update()
        self.threat_threshold = threat_threshold
        self.cooldown_frames  = cooldown_frames

        # Minimum confidence a "LOOKING" prediction must have to count as a threat.
        # Raised to 80 to reduce false positives from uncertain predictions.
        self.confidence_threshold = 80.0

        # --- Internal counters (the heart of temporal smoothing) ---
        # These increment/reset every frame to track how long each state has held

        # How many consecutive frames the GLOBAL threat condition has been met
        self.threat_frame_count = 0

        # How many consecutive frames since the last THREAT frame
        self.safe_frame_count   = 0

        # The publicly visible current state: "SAFE" or "THREAT"
        self.current_state = "SAFE"

        # --- Per-observer consecutive LOOKING streak counters ---
        # Key  : observer index (0 = first observer, 1 = second, etc.)
        # Value: how many consecutive frames that observer has been LOOKING
        #
        # WHY PER-OBSERVER instead of one global counter?
        #   A global counter triggers if ANY observer is LOOKING _in enough frames_,
        #   but those frames don't have to be from the SAME observer.
        #   Example: observer A looks once, observer B looks 9 times → old code
        #   would trigger THREAT even though neither sustained a real stare.
        #   Per-observer streaks require ONE person to stare continuously.
        self.observer_looking_streak = {}   # {observer_index: consecutive_frame_count}

        # Track how many observers were in the previous frame so we can
        # clear streaks when an observer leaves the scene.
        self.prev_observer_count = 0

    # =========================================================================
    def update(self, num_persons, observer_results):
        """
        Called ONCE PER FRAME with the latest detection results.
        Updates internal counters and returns the current system state.

        Parameters:
            num_persons      (int) : total persons detected in the frame
                                     (includes the user + all observers)
            observer_results (list): list of prediction dicts for each OBSERVER.
                                     Each dict: {"label", "confidence", "is_looking"}
                                     This should NOT include the user's own result.

        Returns:
            str: "SAFE" or "THREAT"
        """

        # -----------------------------------------------------------------
        # STEP 1: Classify THIS individual frame as SAFE or THREAT
        #         using per-observer consecutive streak counters
        # -----------------------------------------------------------------

        if num_persons <= 1:
            # Only one person visible (or nobody) — no observers possible.
            # Reset all per-observer streaks since the scene is clear.
            this_frame = "SAFE"
            self.observer_looking_streak = {}
            self.prev_observer_count     = 0

        else:
            # Multiple people detected.

            # ----------------------------------------------------------------
            # Change 4: If the number of observers DECREASED, an observer has
            # left the frame. Clear all streaks to avoid carrying over stale
            # counts from a person who is no longer there.
            # ----------------------------------------------------------------
            current_observer_count = len(observer_results)
            if current_observer_count < self.prev_observer_count:
                self.observer_looking_streak = {}
            self.prev_observer_count = current_observer_count

            # ----------------------------------------------------------------
            # Change 1: Filter out UNKNOWN and UNCERTAIN results.
            #
            # UNKNOWN  = face crop failed (bad detection, too small, etc.)
            # UNCERTAIN = model could not confidently decide either way
            #             (gap between LOOKING and NOT_LOOKING probabilities
            #              was too small — see head_pose_predictor.py)
            #
            # Neither of these should contribute to a threat decision.
            # -----------------------------------------------------------------
            SKIP_LABELS = ("UNKNOWN", "UNCERTAIN")

            # ----------------------------------------------------------------
            # Change 3: Per-observer consecutive streak tracking
            #
            # For each observer, count how many frames IN A ROW they have been
            # looking with high confidence. Only when a SINGLE observer's streak
            # reaches 5 continuous frames do we classify this frame as THREAT.
            #
            # This is much harder to false-trigger than a global frame counter:
            #   - A person glancing for 1-2 frames: streak never reaches 5
            #   - A genuine shoulder-surfer staring for 5+ frames: THREAT
            # ----------------------------------------------------------------
            this_frame = "SAFE"   # Default; any observer's streak can flip this

            for i, result in enumerate(observer_results):

                label      = result.get("label",      "UNKNOWN")
                confidence = result.get("confidence", 0.0)
                is_looking = result.get("is_looking", False)

                # Change 2: confidence threshold raised to 80
                if (label not in SKIP_LABELS
                        and is_looking
                        and confidence > self.confidence_threshold):
                    # Observer i has a valid, high-confidence LOOKING result
                    # → extend their individual streak
                    self.observer_looking_streak[i] = (
                        self.observer_looking_streak.get(i, 0) + 1
                    )
                else:
                    # Observer i is NOT looking (or result is uncertain/unknown)
                    # → reset their streak back to zero
                    self.observer_looking_streak[i] = 0

                # 5 consecutive frames of confirmed LOOKING = real threat
                # (at ~20 FPS this is ≈ 0.25 seconds of sustained staring)
                if self.observer_looking_streak[i] >= 5:
                    this_frame = "THREAT"
                    break   # One confirmed streaking observer is enough

        # -----------------------------------------------------------------
        # STEP 2: Update the running counters based on this frame's result
        # -----------------------------------------------------------------
        if this_frame == "THREAT":
            self.threat_frame_count += 1
            self.safe_frame_count   = 0
        else:
            self.safe_frame_count   += 1
            self.threat_frame_count = 0

        # -----------------------------------------------------------------
        # STEP 3: State transition logic
        # -----------------------------------------------------------------
        if self.threat_frame_count >= self.threat_threshold:
            self.current_state = "THREAT"

        if self.safe_frame_count >= self.cooldown_frames:
            self.current_state = "SAFE"

        return self.current_state

    # =========================================================================
    def get_status_info(self):
        """
        Returns a dictionary with detailed status information for display
        on the video overlay or console.

        Returns:
            dict with keys:
                "state"           (str)  : "SAFE" or "THREAT"
                "threat_progress" (float): 0.0 – 100.0, how close we are
                                           to triggering a THREAT alert
                "message"         (str)  : human-readable status description
        """

        # threat_progress shows how far along the threat-triggering streak is
        # 0%  = no recent threat frames
        # 100% = threshold reached, alert active
        threat_progress = min(
            (self.threat_frame_count / self.threat_threshold) * 100.0,
            100.0     # Cap at 100% — don't go above even if counter exceeds threshold
        )

        # Build a human-readable message based on current state
        if self.current_state == "THREAT":
            message = "⚠️  SHOULDER SURFING DETECTED — observer is watching your screen!"
        elif threat_progress > 0:
            # We're not in THREAT yet but a streak is building
            message = f"⚡ Suspicious activity — monitoring... ({threat_progress:.0f}% to alert)"
        else:
            message = "✅ All clear — no observers detected."

        return {
            "state"           : self.current_state,
            "threat_progress" : round(threat_progress, 1),
            "message"         : message
        }

    # =========================================================================
    def reset(self):
        """
        Resets all internal counters and returns the engine to the initial
        SAFE state. Call this when starting a new session or after an alert
        has been acknowledged by the user.
        """
        self.threat_frame_count      = 0
        self.safe_frame_count        = 0
        self.current_state           = "SAFE"
        self.observer_looking_streak = {}
        self.prev_observer_count     = 0
        print("[ThreatDecisionEngine] Reset to SAFE state.")


# =============================================================================
# SIMULATION TEST — runs only when this file is executed directly
# =============================================================================

if __name__ == "__main__":

    print("=" * 65)
    print("ThreatDecisionEngine — 50-Frame Simulation")
    print("=" * 65)
    print()

    # Create the engine with tighter settings for a shorter demo
    # threat_threshold=5  → needs 5 consecutive threat frames to alert
    # cooldown_frames=8   → needs 8 consecutive safe frames to clear
    engine = ThreatDecisionEngine(threat_threshold=5, cooldown_frames=8)

    # --- Define alternating scenarios ---
    # Each scenario is a tuple:
    #   (num_persons, observer_results, scenario_label)

    # Helper to build an observer prediction dict quickly
    def obs(is_looking, confidence):
        label = "LOOKING" if is_looking else "NOT_LOOKING"
        return {"label": label, "confidence": confidence, "is_looking": is_looking}

    # 50-frame script: mix of SAFE and THREAT situations
    frames = []

    # Frames 1–8: only 1 person visible (user alone) → should always be SAFE
    for _ in range(8):
        frames.append((1, [], "User alone"))

    # Frames 9–16: 2 people, observer NOT looking → should stay SAFE
    for _ in range(8):
        frames.append((2, [obs(False, 88.0)], "Observer NOT looking"))

    # Frames 17–24: 2 people, observer IS looking with high confidence
    # → should trigger THREAT after 5 consecutive frames
    for _ in range(8):
        frames.append((2, [obs(True, 91.5)], "Observer LOOKING (high conf)"))

    # Frames 25–28: observer IS looking but LOW confidence  → should be SAFE
    # (below 65% threshold — treated as uncertain, not a threat)
    for _ in range(4):
        frames.append((2, [obs(True, 55.0)], "Observer LOOKING (low conf)"))

    # Frames 29–36: observer IS looking again — threat should re-trigger
    for _ in range(8):
        frames.append((2, [obs(True, 78.0)], "Observer LOOKING again"))

    # Frames 37–50: observer stops looking → should cool down to SAFE
    for _ in range(14):
        frames.append((2, [obs(False, 83.0)], "Observer stopped looking"))

    # --- Run the simulation ---
    print(f"{'Frame':>5}  {'Scenario':<35}  {'Frame Result':<12}  {'State':<8}  {'Progress':>8}  Message")
    print("-" * 100)

    prev_state = "SAFE"

    for frame_num, (num_persons, observer_results, scenario) in enumerate(frames, start=1):

        # Determine what this single frame would be classified as (for display)
        # (mirrors the logic inside engine.update())
        if num_persons <= 1:
            raw = "SAFE"
        else:
            valid = [o for o in observer_results
                     if o.get("label") not in ("UNKNOWN", "UNCERTAIN")]
            if len(valid) == 0:
                raw = "SAFE"
            elif any(o["is_looking"] and o["confidence"] > 80 for o in valid):
                raw = "THREAT"
            else:
                raw = "SAFE"

        # Feed into the engine
        state = engine.update(num_persons, observer_results)
        info  = engine.get_status_info()

        # Mark state changes clearly
        changed = " ◄ CHANGED" if state != prev_state else ""
        prev_state = state

        # Colour-code in terminal using ANSI codes (works in most terminals and Colab)
        state_display = f"\033[91m{state}\033[0m" if state == "THREAT" else f"\033[92m{state}\033[0m"

        print(f"{frame_num:>5}  {scenario:<35}  {raw:<12}  {state:<8}  {info['threat_progress']:>7.1f}%  {info['message'][:40]}{changed}")

    print()
    print("--- Testing reset() ---")
    engine.reset()
    info = engine.get_status_info()
    print(f"State after reset: {info['state']}  |  Progress: {info['threat_progress']}%")
    print()
    print("Simulation complete!")
