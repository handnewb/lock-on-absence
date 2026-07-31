# Splunk integration

## Why JSON and not CEF/LEEF

Splunk ingests JSON natively and it is the preferred format — `INDEXED_EXTRACTIONS
= json` gives you typed fields at search time with no regex. CEF is ArcSight's
wire format; LEEF is QRadar's. Converting this project's JSON Lines into either
one for Splunk would flatten nested objects into `key=value` pairs and lose
fidelity for nothing.

What actually makes events usable in Splunk Enterprise Security is **CIM
compliance**, not the wire format. `props.conf` maps the events into the
`Authentication` and `Change` data models so ES correlation searches can see them.

Build a CEF or LEEF serializer only when a client shows up running ArcSight or
QRadar.

## Install

```bash
mkdir -p $SPLUNK_HOME/etc/apps/lock_on_absence/{local,lookups}
cp props.conf transforms.conf $SPLUNK_HOME/etc/apps/lock_on_absence/local/
cp loa_events.csv             $SPLUNK_HOME/etc/apps/lock_on_absence/lookups/
$SPLUNK_HOME/bin/splunk restart
```

`inputs.conf` on the endpoint (or via a Universal Forwarder):

```ini
[monitor:///var/log/lock-on-absence/siem.jsonl]
sourcetype = lock_on_absence:json
index      = security
```

Agent side:

```bash
lock-on-absence --siem /var/log/lock-on-absence/siem.jsonl --event-log
```

## Verify the mapping

```
index=security sourcetype=lock_on_absence:json
| table _time dest action signature signature_id severity lock_reason is_dry_run
```

## Searches worth saving

**A lock that failed — the highest-signal event in the whole system.** It means
policy said lock and the OS did not comply, so the session may be sitting
unlocked right now.

```
index=security sourcetype=lock_on_absence:json event_id=1005
| stats count min(_time) AS first max(_time) AS last BY dest, lock_reason
```

**Agent went dark.** No heartbeat-equivalent event for 15 minutes on a host that
normally reports. Pair it with the external watchdog, not instead of it.

```
index=security sourcetype=lock_on_absence:json
| stats latest(_time) AS last_seen BY dest
| eval minutes_silent=round((now()-last_seen)/60, 1)
| where minutes_silent > 15
```

**Intruder events clustered in time** — someone repeatedly approaching a
workstation that is not theirs.

```
index=security sourcetype=lock_on_absence:json event_id=1001
| bin _time span=1h | stats count BY _time, dest | where count >= 3
```

**Exclude rehearsals from every dashboard.** A `--no-lock` run emits the same
events with `dry_run=true`.

```
index=security sourcetype=lock_on_absence:json is_dry_run=0
```

## Caveat on the MITRE column

`loa_events.csv` carries a `mitre_technique` field. Treat it as a hint for
triage, not as a mapping anyone has reviewed. A screen-lock failure is *evidence
consistent with* T1562 Impair Defenses; it is far more often a broken
`loginctl`. Do not build alerting that assumes malice.
