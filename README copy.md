# reBot Setup

**Hardware**: reBot Arm 102 (leader) + Seeed B601 (follower, Damiao motors)  
**OS**: macOS (arm64) · **Env**: `rebot` conda

---

## One-time setup

```bash
conda activate rebot

# MacCAN PCBUSB (CAN on macOS) — already installed if /usr/local/lib/libPCBUSB.dylib exists
# If blocked by Gatekeeper:
sudo xattr -d com.apple.quarantine /usr/local/lib/libPCBUSB.dylib /usr/local/lib/libPCBUSB.0.13.dylib

# Install packages (if not already)
cd lerobot && pip install -e . && cd ..
cd lerobot-robot-seeed-b601 && pip install -e . && cd ..
cd lerobot-teleoperator-rebot-arm-102 && pip install -e . && cd ..
pip install pyyaml
```

---

## Configure ports

Edit [`dev_config.yaml`](dev_config.yaml) — just the two port lines:

```bash
# Find your devices after plugging in
ls /dev/cu.usbserial-*    # leader (reBot Arm 102 UART)
ls /dev/cu.usbmodem*      # follower (Damiao USB2CAN)
```

| Field | Value |
|---|---|
| `leader.port` | UART port for reBot Arm 102 |
| `follower.port` | USB port for Damiao adapter |
| `follower.can_adapter` | `damiao` (serial bridge) or `socketcan` (PCAN) |

---

## Run

```bash
conda activate rebot
python teleop.py
```

First run will trigger calibration — move arm to zero pose, press Enter.

---

## Troubleshooting

**`PCBUSB loaded: False`**  
→ Run the `xattr` command above, then retry.

**Port not found / permission denied**  
→ Check `ls /dev/cu.*` with hardware plugged in. On macOS, Damiao adapter shows as `cu.usbmodem*`.

**Motor doesn't respond**  
→ Check CAN IDs match physical wiring (defaults: send `0x01–0x07`, recv `0x11–0x17`).

**Verify PCBUSB loads**:
```bash
conda run -n rebot python -c "from motorbridge.platform_hints import can_load_pcbusb; print(can_load_pcbusb())"
```
