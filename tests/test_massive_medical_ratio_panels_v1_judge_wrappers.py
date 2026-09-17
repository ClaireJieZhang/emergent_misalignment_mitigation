"""Offline shell boundary tests; never reach SSH or a paid SDK boundary."""

import os
import json
import fcntl
from pathlib import Path
import pty
import select
import subprocess
import sys
import tempfile
import termios
import time
import unittest


REPO = Path(__file__).resolve().parents[1]
FINALIZER = REPO / "scripts/finalize_massive_medical_ratio_panels_v1_judge_tillicum.sh"
STAGER = REPO / "scripts/stage_massive_medical_ratio_panels_v1_judge_tillicum.sh"


def acknowledgments(stage):
    calls, cap = ("1", "0.003072") if stage == "canary" else ("624", "1.916928")
    return [stage, "--ack-calls", calls, "--ack-max-cost-usd", cap,
            "--ack-total-cap-usd", "1.920000", "--ack-no-retry-resume"]


def clean_environment():
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("OPENAI_") or key in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"}:
            environment.pop(key)
    return environment


class WrapperBoundaryTests(unittest.TestCase):
    def run_script(self, script, arguments=(), environment=None):
        return subprocess.run(["bash", str(script), *arguments],
                              input="", text=True, capture_output=True,
                              env=environment or clean_environment(), timeout=10)

    def test_all_three_wrappers_pass_shell_syntax(self):
        for name in ("stage", "finalize", "status"):
            script = REPO / f"scripts/{name}_massive_medical_ratio_panels_v1_judge_tillicum.sh"
            result = subprocess.run(["bash", "-n", str(script)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_or_unknown_stage_never_reaches_terminal_prompt(self):
        for args in ([], ["resume"], ["restart"], ["all"]):
            result = self.run_script(FINALIZER, args)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("OpenAI API key:", result.stderr)

    def test_extra_reserved_or_duplicate_arguments_rejected_before_prompt(self):
        for extra in (["--stage", "continuation"], ["--manifest", "/private/tmp/not-a-manifest"],
                      ["--owner-token", "a" * 64], ["--ack-calls", "624"]):
            result = self.run_script(FINALIZER, acknowledgments("canary") + extra)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("OpenAI API key:", result.stderr)

    def test_incorrect_stage_cap_is_rejected_before_prompt(self):
        args = acknowledgments("canary")
        args[4] = "1.916928"
        self.assertEqual(self.run_script(FINALIZER, args).returncode, 2)

    def test_both_valid_stages_require_private_interactive_terminal(self):
        for stage in ("canary", "continuation"):
            result = self.run_script(FINALIZER, acknowledgments(stage))
            self.assertEqual(result.returncode, 3)
            self.assertIn("private interactive terminal", result.stderr)

    def test_inherited_key_rejected_and_not_echoed(self):
        environment = clean_environment()
        environment["OPENAI_API_KEY"] = "offline-dummy-secret-do-not-echo"
        for script, arguments in ((FINALIZER, acknowledgments("canary")), (STAGER, [])):
            result = self.run_script(script, arguments, environment)
            self.assertEqual(result.returncode, 3)
            self.assertNotIn(environment["OPENAI_API_KEY"], result.stdout + result.stderr)

    def test_finalizer_cannot_be_sourced_into_parent_shell(self):
        result = subprocess.run(["bash", "-c", 'source "$1"; exit "$?"', "test", str(FINALIZER)],
                                input="", text=True, capture_output=True,
                                env=clean_environment(), timeout=10)
        self.assertEqual(result.returncode, 3)
        self.assertIn("do not source", result.stderr)

    def test_readiness_and_freshness_precede_hidden_key_prompt(self):
        source = FINALIZER.read_text()
        self.assertLess(source.index("set +a"), source.index("IFS= read"))
        self.assertLess(source.index("unset OPENAI_API_KEY"), source.index("IFS= read"))
        self.assertLess(source.index('"$runner" audit-stage'), source.index("IFS= read"))
        self.assertLess(source.index('"$runner" validate-sdk-serialization'), source.index("IFS= read"))
        self.assertGreater(source.index('"$runner" authorize'), source.index("IFS= read"))
        self.assertGreater(source.index("export OPENAI_API_KEY"), source.index('"$runner" authorize'))
        self.assertLess(source.index("export OPENAI_API_KEY"), source.index('"$runner" run'))

    def test_allexport_invocation_keeps_authorization_keyless_and_run_child_only(self):
        # Replace only the fixed filesystem root in a temporary wrapper copy.
        # Both Git and Python are local stubs: no core, SSH, or SDK executes.
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary).resolve()
            repo = task_root / "projects/subliminal-mitigate-mmu-ratio-panel-judge-v1"
            repo.mkdir(parents=True)
            output = task_root / "outputs/massive_medical_ratio_panels_v1_judge"
            (output / "control").mkdir(parents=True)
            (output / "control/PREP.json").write_text("{}")
            (output / "logs").mkdir()
            event_path = task_root / "events.json"
            interpreter = task_root / "envs/subliminal-mitigate-py311/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text(f'''#!{sys.executable}
import json,os,pathlib,sys
a=sys.argv[1:]
if a[0]=="-B": a=a[1:]
if a[0]=="-c":
    print("a"*64); sys.exit(0)
command=a[1]
p=pathlib.Path({str(event_path)!r})
events=json.loads(p.read_text()) if p.exists() else []
present="OPENAI_API_KEY" in os.environ
events.append({{"command":command,"key_present":present}})
p.write_text(json.dumps(events))
if present != (command=="run"): sys.exit(9)
print("OFFLINE_STUB_OK")
''')
            interpreter.chmod(0o755)
            binaries = task_root / "bin"
            binaries.mkdir()
            git_stub = binaries / "git"
            git_stub.write_text("#!/bin/bash\nexit 0\n")
            git_stub.chmod(0o755)
            wrapper = task_root / "wrapper.sh"
            wrapper.write_text(FINALIZER.read_text().replace(
                "/gpfs/projects/stf/claizhan/subliminal-mitigate", str(task_root)))
            environment = clean_environment()
            environment["PATH"] = str(binaries) + os.pathsep + environment["PATH"]
            master, slave = pty.openpty()
            def controlling_terminal():
                os.setsid()
                fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
            process = subprocess.Popen(["bash", "-a", str(wrapper), *acknowledgments("canary")],
                                       stdin=slave, stdout=slave, stderr=slave, env=environment,
                                       preexec_fn=controlling_terminal)
            os.close(slave)
            captured = b""
            entered = False
            deadline = time.monotonic() + 10
            try:
                while time.monotonic() < deadline:
                    ready, _, _ = select.select([master], [], [], 0.1)
                    if ready:
                        try:
                            chunk = os.read(master, 8192)
                        except OSError:
                            break
                        if not chunk:
                            break
                        captured += chunk
                    if not entered and b"OpenAI API key: " in captured:
                        # Bash can print the prompt just before disabling echo.
                        # The test waits for the terminal's actual silent state.
                        if not termios.tcgetattr(master)[3] & termios.ECHO:
                            os.write(master, b"offline-dummy-hidden-key\n")
                            entered = True
                    elif process.poll() is not None:
                        break
                self.assertEqual(process.wait(timeout=1), 0, captured.decode())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                os.close(master)
            self.assertTrue(entered)
            self.assertNotIn(b"offline-dummy-hidden-key", captured)
            events = json.loads(event_path.read_text())
            self.assertFalse(next(e["key_present"] for e in events if e["command"] == "authorize"))
            self.assertTrue(next(e["key_present"] for e in events if e["command"] == "run"))
            self.assertTrue(all(not e["key_present"] for e in events if e["command"] != "run"))


if __name__ == "__main__":
    unittest.main()
