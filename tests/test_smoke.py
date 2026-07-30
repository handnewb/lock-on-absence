"""Smoke tests — pegam AttributeError/NameError de refactor sem webcam."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


class TestImportsAndInstantiation:
    """P0-1, P0-3: tudo que importa e instancia sem câmera."""

    def test_body_detector_instantiates(self):
        """Pega AttributeError da property sem setter (P0-1)."""
        import face_utils

        face_utils.BodyDetector()

    def test_logger_instantiates(self):
        import face_utils

        face_utils.Logger()

    def test_keep_awake_instantiates(self):
        import face_utils

        face_utils.KeepAwake()

    def test_event_logger_instantiates(self):
        import face_utils

        face_utils.EventLogger(False, None)

    def test_presence_state_machine_instantiates(self):
        from presence_state_machine import PresenceStateMachine, Config

        psm = PresenceStateMachine(Config())
        assert psm is not None


class TestCLI:
    """--help exercita argparse + imports de topo."""

    def test_lock_help(self):
        r = subprocess.run(
            [sys.executable, str(REPO / "lock-on-absence.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, r.stderr

    def test_enroll_help(self):
        r = subprocess.run(
            [sys.executable, str(REPO / "enroll.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, r.stderr


class TestNoUndefinedNames:
    """pyflakes — pega NameError como user_data/samples_per (P0-2)."""

    @staticmethod
    def _check(path: str) -> list[str]:
        r = subprocess.run(
            [sys.executable, "-m", "pyflakes", path],
            capture_output=True, text=True, timeout=30,
        )
        errors = []
        for line in r.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Allowed: unused-import of YUNetDetector etc. in lock (they're conditionally used)
            if "imported but unused" in stripped and (
                "YUNetDetector" in stripped
                or "create_detector" in stripped
                or "download_yunet" in stripped
                or "camera_available" in stripped
                or "detect_faces_dnn" in stripped
            ):
                continue
            if "undefined name" in stripped or "imported but unused" in stripped:
                errors.append(stripped)
        return errors

    def test_face_utils_no_undefined(self):
        errs = self._check("face_utils.py")
        assert not errs, "\n".join(errs)

    def test_lock_no_undefined(self):
        errs = self._check("lock-on-absence.py")
        assert not errs, "\n".join(errs)

    def test_enroll_no_undefined(self):
        errs = self._check("enroll.py")
        assert not errs, "\n".join(errs)

    def test_state_machine_no_undefined(self):
        errs = self._check("presence_state_machine.py")
        assert not errs, "\n".join(errs)


class TestStateMachineIntegration:
    """P0-6: state machine precisa estar no caminho de produção."""

    def test_state_machine_is_imported(self):
        src = (REPO / "lock-on-absence.py").read_text()
        assert "PresenceStateMachine" in src, (
            "state machine não integrada ao loop principal"
        )


class TestDocumentedFlagsAreImplemented:
    """P0-7: flags documentadas precisam ter implementação no mesmo arquivo."""

    def test_purge_flag_implemented_in_enroll(self):
        src = (REPO / "enroll.py").read_text()
        assert "args.purge" in src, (
            "--purge declarado em enroll.py mas argumento nunca lido"
        )

    def test_no_consent_flag_implemented_in_enroll(self):
        src = (REPO / "enroll.py").read_text()
        assert "args.no_consent" in src, (
            "--no-consent declarado em enroll.py mas argumento nunca lido"
        )
