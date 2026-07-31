# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Use GitHub's private reporting: **Security → Advisories → Report a vulnerability**
on this repository. That channel is private until we publish an advisory together.

Include, as far as you can:

- version (`lock-on-absence --version`) and OS
- what an attacker gains, and what access they need to start
- reproduction steps, ideally as a `--scenario` JSONL that the replay harness
  can run — that makes the finding a regression test immediately

I maintain this in my spare time. Expect an acknowledgement within 7 days and an
assessment within 30. If you have heard nothing in 30 days, escalate by opening a
public issue that says only "awaiting response on a private report" — no details.

## Scope

In scope:

- bypassing a lock decision (making the agent believe the owner is present)
- disabling the agent or the watchdog from an unprivileged local process
- accepting a non-owner face as the owner
- anything in `lock_on_absence/` that fails open — an unlocked screen when the
  policy says locked

Out of scope, because they are documented properties rather than defects:

- **Photo and video spoofing.** There is no liveness detection. A printed photo
  or a phone screen can defeat recognition. `--anti-spoof-timeout` is a weak
  movement heuristic, off by default, and is not liveness.
- **`--any-face` mode.** It accepts any detected face by design and prints a
  warning saying so. It exists for people without an enrolled model.
- **An attacker who already has your login session.** Tampering with
  `face_model.yml`, `face_model.json` or `watchdog_heartbeat.txt` requires code
  execution as your user. At that point the screen lock is not your problem.
  The SHA-256 model digest is tamper *detection*, not prevention.
- **LBPH recognition accuracy.** LBPH is a 2006 algorithm and a determined
  lookalike may pass. Migration to SFace embeddings is on the roadmap.

## Security properties this project tries to hold

1. **Fail closed.** If presence cannot be proven, lock. No camera signal for
   `--camera-fail-grace` seconds locks the screen in `--mode security`.
2. **Never suppress the OS lock while blind.** Keep-awake is only engaged on a
   fresh, verified presence. During cooldown, pause, or any failure, the native
   idle timeout is allowed to do its job.
3. **A failed lock is reported, not assumed.** `lock_screen()` checks the exit
   status of every mechanism. A failure emits `lock_failed` (1005) and never the
   cause event, so the audit trail cannot show a lock that did not happen.
4. **Refuse rather than pretend.** No model and no `--any-face` exits 2. A model
   that fails its digest exits 3.

## Privacy

Everything stays local. No network calls except the optional one-time YuNet model
download from the OpenCV Zoo. No telemetry, no images stored — only an LBPH
histogram template.

`face_model.yml` is a biometric template. Under LGPD Art. 5 II and GDPR Art. 9
that is sensitive personal data. `enroll` asks for consent, restricts file
permissions, and `--purge` deletes both files. If you deploy this on anyone
else's machine, that is your legal responsibility, not the tool's: get informed
consent in writing and check your local rules (BIPA in Illinois is strict).

## Supported versions

| Version | Supported |
|---|---|
| 5.1.x | yes |
| 5.0.x | upgrade to 5.1 |
| < 5.0 | no — pre-package, known fail-open bugs |
