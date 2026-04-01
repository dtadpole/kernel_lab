# Service Skill Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate all 5 service commands work correctly on both hosts, including nuke + redeploy cycles, then merge to main.

**Architecture:** Python CLI (`plugins/cuda/deploy/cli.py`) manages remote cuda_exec via SSH. Host config in `conf/hosts/default.yaml`.

**Tech Stack:** Python 3.12, SSH, systemd --user, rsync, uv

---

### Task 1: Full nuke + redeploy cycle on _one

- [ ] **Step 1: Nuke _one with --data**

Run: `python plugins/cuda/deploy/cli.py nuke _one --data`
Expected: Service stopped, service dir removed, data dir removed.

- [ ] **Step 2: Verify _one is clean**

Run: `ssh _one 'ls ~/.cuda_exec_service 2>&1; ls ~/.cuda_exec 2>&1; systemctl --user is-active cuda-exec 2>&1'`
Expected: All "No such file or directory", service "inactive" or "not found".

- [ ] **Step 3: Fresh deploy to _one**

Run: `python plugins/cuda/deploy/cli.py deploy _one`
Expected: All 5 steps complete, "Deploy complete" message.

- [ ] **Step 4: Start _one**

Run: `python plugins/cuda/deploy/cli.py start _one`
Expected: "Running. Health: OK"

- [ ] **Step 5: Status _one**

Run: `python plugins/cuda/deploy/cli.py status _one`
Expected: Service active, health OK, GPU info shown.

---

### Task 2: Deploy + verify _two

- [ ] **Step 1: Deploy to _two**

Run: `python plugins/cuda/deploy/cli.py deploy _two`

- [ ] **Step 2: Start _two**

Run: `python plugins/cuda/deploy/cli.py start _two`
Expected: "Running. Health: OK"

- [ ] **Step 3: Status --all**

Run: `python plugins/cuda/deploy/cli.py status --all`
Expected: Both hosts show active + healthy.

---

### Task 3: Stop + start cycle on _two

- [ ] **Step 1: Stop _two**

Run: `python plugins/cuda/deploy/cli.py stop _two`
Expected: "Stopped."

- [ ] **Step 2: Start _two (without redeploy)**

Run: `python plugins/cuda/deploy/cli.py start _two`
Expected: "Running. Health: OK"

---

### Task 4: Nuke _two + redeploy with --rebuild

- [ ] **Step 1: Nuke _two**

Run: `python plugins/cuda/deploy/cli.py nuke _two`
Expected: Service removed, data kept.

- [ ] **Step 2: Deploy with --rebuild**

Run: `python plugins/cuda/deploy/cli.py deploy _two --rebuild`

- [ ] **Step 3: Start + verify**

Run: `python plugins/cuda/deploy/cli.py start _two`
Expected: "Running. Health: OK"

---

### Task 5: Final status + push

- [ ] **Step 1: Status --all**

Run: `python plugins/cuda/deploy/cli.py status --all`
Expected: Both hosts active and healthy.

- [ ] **Step 2: Push plugins branch and merge to main**

```bash
git push origin plugins
cd /home/centos/kernel_lab && git fetch origin plugins && git merge origin/plugins --no-edit && git push origin main
```
