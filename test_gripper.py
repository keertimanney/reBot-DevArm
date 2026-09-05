#!/usr/bin/env python3
"""Gripper motor test using FORCE_POS mode — the correct mode for the B601 gripper.

Range: 0.0 rad (closed) → -4.71 rad (-270°, fully open).

Run alone:
    python release_arm.py
    python test_gripper.py
"""
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "reBotArm_control_py"))

from motorbridge import Controller, Mode, CallError

CHANNEL     = "/dev/tty.usbmodem00000000050C1"
BAUD        = 921600
MOTOR_ID    = 0x07
FEEDBACK_ID = 0x17
MODEL       = "4310"

VLIM        = 2.0   # rad/s velocity limit
RATIO       = 0.1   # torque ratio (10 % of max)


def read_state(ctrl, mot):
    try:
        mot.request_feedback()
        ctrl.poll_feedback_once()
        time.sleep(0.005)
        st = mot.get_state()
        if st:
            return st.pos, st.vel, st.torq, st.status_code
    except Exception as e:
        print(f"  [read error] {e}")
    return None, None, None, None


def continuous_sweep(ctrl, mot, targets, hold_s=2.0):
    """100 Hz FORCE_POS loop, sweep through targets."""
    target_box = [targets[0]]
    running    = [True]

    def loop():
        dt = 0.01
        while running[0]:
            t0 = time.perf_counter()
            try:
                mot.send_force_pos(target_box[0], VLIM, RATIO)
                mot.request_feedback()
                ctrl.poll_feedback_once()
            except Exception as e:
                print(f"  [loop err] {e}")
            rem = dt - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

    t = threading.Thread(target=loop, daemon=True)
    t.start()

    try:
        prev_pos = None
        for target in targets:
            target_box[0] = target
            print(f"\n  target={target:+.3f} rad  ({target * 180/3.14159:+.1f}°)  hold {hold_s}s ...")
            deadline = time.monotonic() + hold_s
            while time.monotonic() < deadline:
                pos, vel, torq, status = read_state(ctrl, mot)
                moved = ""
                if prev_pos is not None and pos is not None and abs(pos - prev_pos) > 0.005:
                    moved = f"  ← MOVED Δ={pos-prev_pos:+.4f}"
                if pos is not None:
                    print(f"    pos={pos:+.4f} ({pos*180/3.14159:+.1f}°)  vel={vel:+.4f}  torq={torq:+.4f}  status={status}{moved}")
                prev_pos = pos
                time.sleep(0.3)
    finally:
        running[0] = False
        t.join(timeout=1.0)


def main():
    print(f"Connecting to {CHANNEL} ...")
    ctrl = Controller.from_dm_serial(CHANNEL, BAUD)
    mot  = ctrl.add_damiao_motor(MOTOR_ID, FEEDBACK_ID, MODEL)
    print(f"Motor: id={MOTOR_ID:#04x}  feedback={FEEDBACK_ID:#04x}")

    print("\nClearing fault (red LED = fault state) ...")
    try:
        mot.clear_error()
        print("  clear_error OK")
    except Exception as e:
        print(f"  clear_error: {e}")
    time.sleep(0.2)

    print("\nEnabling ...")
    try:
        ctrl.enable_all()
        time.sleep(0.5)
    except Exception as e:
        print(f"  enable_all: {e}")

    print("\nInitial state after clear+enable:")
    for _ in range(3):
        pos, vel, torq, status = read_state(ctrl, mot)
        print(f"  pos={pos:+.4f}  vel={vel:+.4f}  torq={torq:+.4f}  status={status}")

    print("\nSwitching to FORCE_POS mode ...")
    for attempt in range(10):
        try:
            mot.ensure_mode(Mode.FORCE_POS, 2000)
            print(f"  FORCE_POS mode OK (attempt {attempt+1})")
            break
        except Exception as e:
            print(f"  attempt {attempt+1}: {e}")
            time.sleep(0.05)
    time.sleep(0.2)

    print("\n── FORCE_POS sweep: closed→open→closed ─────────────────────────────")
    # 0.0 = closed, -4.71 = fully open (-270°)
    targets = [0.0, -1.0, -2.0, -3.0, -4.0, -3.0, -2.0, -1.0, 0.0]
    continuous_sweep(ctrl, mot, targets, hold_s=2.0)

    print("\nDisabling ...")
    try:
        ctrl.disable_all()
        time.sleep(0.3)
    except Exception:
        pass
    try:
        ctrl.shutdown()
        ctrl.close()
    except Exception:
        pass
    print("Done.")


if __name__ == "__main__":
    main()
