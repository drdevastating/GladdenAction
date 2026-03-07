# GladdenAction# GladdenAction — AI-Powered Desktop Automation Agent

<p align="center">
  <strong>An intelligent, LLM-driven desktop agent that controls your computer through natural language commands.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/LLM-Llama%203.3%2070B-orange?logo=meta&logoColor=white" alt="Llama 3.3">
  <img src="https://img.shields.io/badge/Inference-Groq-purple" alt="Groq">
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white" alt="Windows">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [API Keys & Configuration](#api-keys--configuration)
- [Installation & Setup](#installation--setup)
- [Running the Agent](#running-the-agent)
- [Usage Examples](#usage-examples)
- [Security Model](#security-model)
- [Technology Stack](#technology-stack)
- [Future Scope & Advancements](#future-scope--advancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**GladdenAction** is an AI-powered desktop automation agent that interprets natural language instructions and executes real actions on your computer. It leverages **Llama 3.3 70B** (via the **Groq** inference API) as its reasoning engine and a modular tool system to perform:

- **UI Automation** — Open applications (Notepad, VS Code, Chrome, WhatsApp), type content, save files, send emails, and send messages — all via visible, on-screen interaction.
- **File Management** — Create, write, and manage files directly on disk.
- **System Control** — Monitor processes, inspect CPU/memory/disk usage, manage the filesystem, and control running applications — all within strict security boundaries.

Think of it as a **Jarvis-like assistant** that doesn't just talk — it *acts* on your desktop.

---

## Architecture

GladdenAction follows a clean, layered architecture with strict separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                     User (REPL)                         │
│              Natural Language Input                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Agent Layer                            │
│  • Builds structured prompts with tool metadata         │
│  • Sends instruction to Llama 3.3 via Groq API         │
│  • Parses JSON tool-call decision from LLM response     │
│  • Biases selection toward UI automation tools           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 Executor Layer                           │
│  • Resolves tool name → registered BaseTool instance    │
│  • Validates inputs against tool schema                 │
│  • Executes tool & wraps result in ToolResult           │
│  • Emits structured execution events                    │
│  • Catches all exceptions (never leaks raw errors)      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Tool Layer                              │
│  ┌──────────────────┐ ┌──────────────────┐              │
│  │  UIAutomation    │ │  FileCreation    │              │
│  │  Tool            │ │  Tool            │              │
│  │  • Notepad       │ │  • Direct file   │              │
│  │  • VS Code       │ │    write to disk │              │
│  │  • Gmail/Chrome  │ │                  │              │
│  │  • WhatsApp      │ │                  │              │
│  └──────────────────┘ └──────────────────┘              │
│  ┌──────────────────┐                                   │
│  │  SystemControl   │                                   │
│  │  Tool            │                                   │
│  │  • Process mgmt  │                                   │
│  │  • System metrics│                                   │
│  │  • Filesystem ops│                                   │
│  └──────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

**Data flow:** User → Agent (LLM reasoning) → Executor (validation + dispatch) → Tool (action) → ToolResult → User

---

## Features

### 1. UI Automation (`ui_automation` tool)

| Workflow | Description |
|---|---|
| `create_file_notepad` | Opens Notepad, types content, saves file using Save As dialog |
| `create_file_vscode` | Writes file to disk, opens it in VS Code automatically |
| `send_email_browser` | Opens Gmail in Chrome with pre-filled compose URL, sends via Ctrl+Enter with intelligent fallback |
| `send_whatsapp_desktop` | Launches WhatsApp Desktop, searches contact, types and sends message |

### 2. File Creation (`file_creation` tool)

- Silent/direct file creation on disk
- Supports relative and absolute paths
- Overwrite protection with optional override
- Automatic parent directory creation

### 3. System Control (`system_control` tool)

| Domain | Actions | Description |
|---|---|---|
| **Process** | `list`, `inspect`, `kill` | List top processes by CPU/memory, inspect a process by name/PID, safely terminate processes |
| **System** | `cpu_usage`, `memory_usage`, `disk_usage`, `uptime` | Real-time system metrics and monitoring |
| **Filesystem** | `list_directory`, `file_info`, `create_directory`, `rename_file`, `delete_file` | Controlled filesystem operations with path safety enforcement |

---

## Project Structure

```
GladdenAction/
├── backend/
│   ├── main.py                             # Entry point — interactive REPL
│   ├── requirements.txt                    # Python dependencies
│   ├── .gitignore                          # Git ignore rules
│   ├── .env                                # Environment variables (create manually)
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   └── agent.py                        # LLM reasoning layer (Groq + Llama 3.3)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── base.py                     # BaseTool abstract class + ToolResult
│   │       ├── registry.py                 # ToolRegistry — centralized tool management
│   │       ├── ui_automation_tool.py       # UI automation workflows (Notepad, VS Code, Gmail, WhatsApp)
│   │       ├── file_creation_tool.py       # Direct file creation tool
│   │       └── system_control_tool.py      # Secure OS capability engine
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   └── executor.py                     # ToolExecutor — validation + dispatch gateway
│   │
│   └── gladden/                            # Python virtual environment (auto-generated)
│       ├── Lib/
│       ├── Scripts/
│       └── pyvenv.cfg
│
└── README.md                               # This file
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Operating System** | Windows 10/11 (UI automation relies on Windows-specific apps) |
| **Python** | 3.11 or higher |
| **Groq API Key** | Free key from [console.groq.com](https://console.groq.com) |
| **Google Chrome** | Required for Gmail email sending workflow |
| **Gmail Account** | Must be logged into Gmail in Chrome |
| **VS Code** | Required for `create_file_vscode` workflow. Ensure `code` is in your system PATH |
| **WhatsApp Desktop** | Required for `send_whatsapp_desktop` workflow (Microsoft Store or standalone) |
| **Screen Resolution** | Minimum 1920×1080 recommended (UI automation uses pixel coordinates) |

---

## API Keys & Configuration

### 1. Groq API Key (Required)

GladdenAction uses the **Groq Cloud** inference API to run **Llama 3.3 70B Versatile** — a powerful open-source LLM.

**Steps to obtain:**

1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up / Log in with your account
3. Navigate to **API Keys** section
4. Click **Create API Key**
5. Copy the generated key

**Groq Free Tier Limits:**
- 30 requests per minute
- 14,400 requests per day
- No credit card required

### 2. Environment Variable Setup

Create a `.env` file inside the `backend/` directory:

```bash
# backend/.env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

**Alternatively**, set it as a system environment variable:

```powershell
# PowerShell (temporary — current session only)
$env:GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"

# PowerShell (permanent — user-level)
[Environment]::SetEnvironmentVariable("GROQ_API_KEY", "gsk_your_actual_groq_api_key_here", "User")
```

```bash
# Linux / macOS
export GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 3. Application Dependencies

| Application | Required For | How to Verify |
|---|---|---|
| **Google Chrome** | `send_email_browser` | The tool auto-detects Chrome in standard installation paths |
| **VS Code** | `create_file_vscode` | Run `code --version` in terminal. If not found, add to PATH during installation |
| **WhatsApp Desktop** | `send_whatsapp_desktop` | Install from Microsoft Store or [whatsapp.com/download](https://www.whatsapp.com/download) |
| **Notepad** | `create_file_notepad` | Pre-installed on all Windows systems |

---

## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/drdevastating/GladdenAction.git
cd GladdenAction
```

### Step 2: Create & Activate Virtual Environment

```powershell
# Windows (PowerShell)
cd backend
python -m venv gladden
.\gladden\Scripts\Activate.ps1
```

```bash
# Linux / macOS
cd backend
python3 -m venv gladden
source gladden/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**

| Package | Purpose |
|---|---|
| `groq>=0.9.0` | Groq SDK — LLM inference (Llama 3.3 70B) |
| `fastapi>=0.111.0` | Web framework (for future API endpoints) |
| `uvicorn[standard]>=0.29.0` | ASGI server (for future web deployment) |
| `pyautogui>=0.9.54` | Desktop UI automation (mouse/keyboard control) |
| `pyperclip>=1.8.2` | Clipboard operations (for reliable text pasting) |
| `psutil>=5.9.0` | System metrics and process management |
| `python-dotenv>=1.0.0` | Load environment variables from `.env` file |
| `typing-extensions>=4.11.0` | Typing backports for Python compatibility |

### Step 4: Configure Environment Variables

```bash
# Create .env file in the backend directory
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### Step 5: Verify Setup

```bash
python -c "from groq import Groq; print('Groq SDK OK')"
python -c "import pyautogui; print('PyAutoGUI OK')"
python -c "import psutil; print('psutil OK')"
```

---

## Running the Agent

```powershell
cd backend
.\gladden\Scripts\Activate.ps1   # Activate virtual environment (if not already)
python main.py
```

On successful startup, you'll see:

```
╔══════════════════════════════════════════════════════════════════════╗
║         AI Agent — Jarvis Mode  (Groq + Secure OS Control)          ║
╠══════════════════════════════════════════════════════════════════════╣
║  UI Automation                                                       ║
║    "Open Notepad and write a shopping list"                          ║
║    "Create a C++ Hello World in VS Code"                             ║
║    "Send an email to you@example.com saying hello"                   ║
║                                                                      ║
║  Commands:  tools | quit | exit                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### REPL Commands

| Command | Action |
|---|---|
| `tools` | List all registered tools |
| `quit` / `exit` | Exit the agent |
| Any natural language | Processed by the LLM and executed |

---

## Usage Examples

### UI Automation

```
You › Open Notepad and write a shopping list with milk, eggs, and bread
You › Create a Python hello world program in VS Code
You › Send an email to alice@example.com about the project deadline being Friday
You › Send a WhatsApp message to John saying "Meeting at 5pm today"
```

### Process Management

```
You › Show top 5 processes by RAM usage
You › Inspect chrome.exe
You › Kill notepad.exe
You › List processes sorted by CPU
```

### System Metrics

```
You › What is my CPU usage?
You › Check memory stats
You › Show disk usage for C:
You › How long has the PC been running?
```

### Filesystem Operations

```
You › List files in my Documents folder
You › Get info about report.pdf
You › Create a folder called MyProject on the Desktop
You › Rename notes.txt to todo.txt
You › Delete temp.log
```

---

## Security Model

GladdenAction implements a **6-layer security model** to prevent accidental or malicious system damage:

| Layer | Protection | Details |
|---|---|---|
| **Layer 1** | Domain Whitelist | Only `process`, `system`, and `filesystem` domains are accepted |
| **Layer 2** | Action Whitelist | Each domain has a strict set of allowed actions |
| **Layer 3** | Protected Processes | Critical system processes (`explorer.exe`, `csrss.exe`, `lsass.exe`, `smss.exe`, etc.) and special PIDs (0, 4, current Python PID) cannot be killed |
| **Layer 4** | Protected Paths | `C:\Windows`, `Program Files`, `System32`, `AppData`, and root drive paths are blocked from mutating operations |
| **Layer 5** | No Shell Execution | All operations use Python stdlib + `psutil` only — no `os.system()` or `subprocess.run(shell=True)` |
| **Layer 6** | Security Events | All violations emit structured `security_violation_detected` events for audit |

### Protected Filesystem Paths

```
C:\Windows\*
C:\Program Files\*
C:\Program Files (x86)\*
*\AppData\*
/ (root drives)
/etc, /bin, /sbin, /usr/bin, /usr/lib, /boot, /sys, /proc (Linux/macOS)
```

### Protected Processes

```
system, explorer.exe, wininit.exe, csrss.exe, services.exe,
lsass.exe, smss.exe, systemd, launchd, init, kernel_task
```

---

## Technology Stack

| Component | Technology | Role |
|---|---|---|
| **LLM** | Llama 3.3 70B Versatile | Natural language understanding & tool selection |
| **Inference API** | Groq Cloud | Ultra-fast LLM inference (~200 tokens/sec) |
| **Language** | Python 3.11+ | Core implementation |
| **UI Automation** | PyAutoGUI + PyPerclip | Mouse/keyboard control & clipboard operations |
| **System Metrics** | psutil | Process management, CPU/memory/disk monitoring |
| **Web Framework** | FastAPI + Uvicorn | Prepared for future web API / WebSocket interface |
| **Env Management** | python-dotenv | Secure API key loading |

---

## Future Scope & Advancements

### Short-Term Enhancements

| Feature | Description |
|---|---|
| **Multi-Turn Conversations** | Add conversation history and context memory so the agent can handle follow-up instructions (e.g., "Now rename that file to report_v2.txt") |
| **WebSocket / SSE Streaming** | Use FastAPI + WebSockets to stream real-time execution events to a web dashboard instead of console only |
| **Web UI Dashboard** | Build a React/Next.js frontend with a chat interface, live event timeline, and system metrics dashboard |
| **Voice Input Integration** | Add speech-to-text (e.g., OpenAI Whisper, Google Speech API) so users can speak commands instead of typing |
| **Multi-Step Execution Plans** | Allow the LLM to decompose complex tasks into multiple sequential tool calls (e.g., "Create a Python project with a README, main.py, and requirements.txt") |

### Medium-Term Advancements

| Feature | Description |
|---|---|
| **Browser Automation (Playwright/Selenium)** | Replace pixel-based Gmail/Chrome automation with proper browser automation for reliability across screen resolutions |
| **Cross-Platform Support** | Extend UI automation to macOS (AppleScript) and Linux (xdotool) |
| **Custom Tool Plugin System** | Allow users to define and register their own tools via a plugin YAML/JSON spec without modifying core code |
| **Task Scheduling** | Integrate a scheduler (APScheduler / Celery) to run automated tasks at specified times (e.g., "Send a report email every Monday at 9am") |
| **Error Recovery & Retry Logic** | Implement intelligent retry mechanisms for failed UI automations with screenshot-based state detection |
| **Screen Understanding (Vision)** | Integrate vision models (e.g., GPT-4 Vision, LLaVA) to understand what's currently on screen and adapt actions dynamically |

### Long-Term Vision

| Feature | Description |
|---|---|
| **Agentic Workflows** | Multi-agent collaboration where specialized agents handle different domains (e.g., a "Code Agent" + "Email Agent" + "Research Agent") working together on complex tasks |
| **RAG Integration** | Connect to local/cloud knowledge bases (PDFs, Notion, Confluence) so the agent has context about the user's projects |
| **Learning from User Feedback** | Record successful/failed executions and fine-tune tool selection based on user corrections |
| **Enterprise Deployment** | Multi-user support with role-based access control, audit logging, and compliance-ready security |
| **Mobile Companion App** | Control your desktop agent remotely from a mobile device |
| **Integration Hub** | Native connectors for Slack, Teams, Jira, GitHub, Google Calendar, Trello, and other productivity tools |
| **Autonomous Research Agent** | Web browsing + summarization capabilities — "Research the latest Python 3.13 features and create a summary document" |
| **Local LLM Support** | Add support for locally-hosted models via Ollama or llama.cpp to eliminate API dependency and enable fully offline operation |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `GROQ_API_KEY not set` | Ensure `.env` file exists in `backend/` with your key, or set the environment variable |
| `VS Code not found` | Install VS Code and ensure `code` is in your PATH (`code --version` to verify) |
| `PyAutoGUI FailSafe` | Moving your mouse to the top-left corner triggers PyAutoGUI's failsafe — this is intentional. Avoid moving the mouse during automation |
| `Gmail send failed` | Ensure you're logged into Gmail in Chrome. The agent uses pre-filled compose URLs |
| `WhatsApp launch failed` | Install WhatsApp Desktop from Microsoft Store or the official website. Ensure you're logged in |
| `Module not found` | Activate the virtual environment: `.\gladden\Scripts\Activate.ps1` |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-feature`
3. Commit your changes: `git commit -m "Add my new feature"`
4. Push to the branch: `git push origin feature/my-new-feature`
5. Open a Pull Request

---

## License

This project is open-source. See the repository for license details.

---

<p align="center">
  Built with ❤️ using Llama 3.3 + Groq + Python
</p>