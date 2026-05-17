# Reachy + reBot Interaction Agent Prototype

This folder contains a small, hardware-safe orchestration layer for using Reachy
Mini as the interaction agent and the reBot arm as the manipulation backend.

The first version runs in mock mode:

```bash
python -m inference.interaction_agent.run_mock
```

Try commands like:

```text
move the block from top left to center
move from centre to bottom right
what do you see
stop
go home
```

## OpenAI Realtime Text Loop

Set your API key:

```bash
export OPENAI_API_KEY="..."
```

Then run:

```bash
python -m inference.interaction_agent.run_realtime_text
```

This connects to the OpenAI Realtime API over WebSocket, exposes local tools to
the model, and executes tool calls against the current mock arm executor.

The model can call:

| Tool | Purpose |
| --- | --- |
| `move_block` | Move a block from one named position to another |
| `get_scene` | Report calibrated positions and tracked block state |
| `go_home` | Return the arm to home |
| `stop` | Stop arm motion |

Example prompts:

```text
move the block from top left to center
what positions are available?
send the arm home
stop
```

The default model is `gpt-realtime-2`. You can override it:

```bash
python -m inference.interaction_agent.run_realtime_text --model gpt-realtime-2
```

## Architecture

```text
Reachy audio/video
    -> interaction loop
    -> scene perception
    -> intent parser / planner
    -> typed skill call
    -> reBot arm executor
    -> spoken response
```

Reachy owns the human interaction. The arm only receives validated skill calls
such as `move_block`, `go_home`, or `stop`. The agent should never emit raw
motor commands directly.

## Files

| File | Purpose |
| --- | --- |
| `models.py` | Shared dataclasses for objects, poses, decisions, and skill results |
| `perception.py` | Scene provider interface plus a mock five-position tabletop scene |
| `skills.py` | Skill registry, validation, and high-level slot-to-slot manipulation skills |
| `arm_executor.py` | Arm executor interface plus a mock reBot executor |
| `voice.py` | Speech interface, console speech, and optional ElevenLabs HTTP adapter |
| `agent.py` | Rule-based first-pass interaction agent |
| `run_mock.py` | Console demo loop |
| `realtime_client.py` | OpenAI Realtime WebSocket client with local tool-call execution |
| `run_realtime_text.py` | Text console using the Realtime API |

## Next Integration Points

1. Calibrate the five named positions to real table coordinates.
2. Replace `MockArmExecutor` with a real reBot executor backed by the Python SDK,
   ROS2/MoveIt2, or the existing policy loop.
3. Add microphone input and audio playback on top of the Realtime client.
4. Replace `ConsoleSpeaker` with `ElevenLabsSpeaker` where external TTS is still preferred.
