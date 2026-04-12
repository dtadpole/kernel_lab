# Wave Context

- **Mode:** {{ wave.mode }}
- **Current Direction:** {{ wave.direction_json }}
- **Direction file:** {{ wave.direction_path }}
- **Transcript:** {{ wave.transcript_path }}
- **Events:** {{ wave.events_path }}

## Recent Events
{{ wave.recent_events }}

---

## Permission Request
- **Tool:** {{ permission.tool_name }}
- **Input:** {{ permission.tool_input }}

---

## Judgment Criteria
- Is the tool call within the Solver's reasonable scope of work?
- Is the target path safe? (Only modifications under ~/kernel_lab_kb/runs/<run_tag>/ are allowed)
- Is the command destructive? (rm -rf, modifying system files → DENY)
- Does the operation align with the current task?

## Response Format
Your first line MUST be exactly one of:
- ALLOW
- DENY

Second line onward: your reasoning.
