# Migração para v5.0 — Bloco 4 (arquitetura)

Isto é o que muda, por quê, e o que fazer no repositório. Tudo abaixo foi
**executado e validado** antes de ser entregue: 71 testes, `ruff` limpo,
`pyflakes` limpo, `pip install -e .` num venv virgem, e os 4 entry points
resolvendo.

---

## 1. Comandos git para aplicar

```bash
# arquivos que saem
git rm run_hidden.bat                 # substituído por Scheduled Task
git rm __init__.py                    # a raiz não é mais um pacote
git rm presence_state_machine.py      # virou lock_on_absence/state_machine.py
git rm face_utils.py                  # virou lock_on_absence/face_utils.py
git rm test_state_machine.py          # virou tests/test_state_machine.py

# copiar tudo do zip por cima do repo, depois:
git add -A
git commit -m "v5.0: state machine full replacement, FAR/FRR harness, package structure"
```

`lock-on-absence.py`, `enroll.py` e `watchdog.py` **continuam existindo na raiz**
como shims de 8 linhas. Isso é deliberado: `install.sh`, `install.bat`, a unit
systemd e cada comando do README continuam funcionando. O `tests/test_replay.py`
tem um teste que falha se algum shim desaparecer.

---

## 2. B4-1 — Substituição completa da máquina de estados

### O problema

O loop importava a PSM e depois **redecidia** com variáveis paralelas
(`last_face_time`, `static_since`, `prev_face_center`, `_body_detect_active`),
mutando `psm_state` de fora. Duas fontes de verdade, e as duas já divergiam:
`max_body_only` era 20 s na PSM e 60 s na CLI; o anti-spoof travava inteiramente
fora da máquina.

### O que foi feito

`step()` deixou de retornar `(Decision, Reason)` e passou a retornar um
**`Verdict`**:

```python
@dataclass
class Verdict:
    decision: Decision          # KEEP | WARN | LOCK | PAUSE
    reason: Reason
    keep_awake: bool            # o adaptador só obedece
    message: str | None         # só preenchido na TRANSIÇÃO de fase
    detail: dict                # payload estruturado pro SIEM
```

O campo `message` é o que eliminou `_body_detect_active`: a máquina tem memória
de fase (`State._phase`) e emite a mensagem uma única vez. O adaptador não tem
mais nenhuma variável de deduplicação de log.

O campo `keep_awake` é o que corrigiu o bug do cooldown: `KEEP` durante cooldown
agora vem com `keep_awake=False`, então o agente nunca mais suprime o sleep do SO
enquanto a tela está travada. Tem teste dedicado.

Absorvido para dentro da PSM: anti-spoof, falha de câmera, pausa de reunião,
`startup_grace`, cooldown, `mode`, e o modo `--any-face` (via
`Observation.has_recognizer`).

### Correções de comportamento que vieram junto

**`Decision.PAUSE` + `Reason.CAMERA_BUSY`.** O `--meeting-pause` era código morto
(`camera_available` importado e nunca chamado, `camera_busy_until` atribuído e
nunca lido). Com `--on-camera-failure lock` no default, uma call do Teams
disparava trava a cada ~20 s. Agora `camera_is_busy()` pergunta ao SO **quem**
detém `/dev/videoN` via `fuser`/`lsof` — em vez de abrir a câmera para testar,
que acende o LED no Windows e dá falso negativo no Linux porque o V4L2 permite
múltiplos `open()`. A pausa tem orçamento (`meeting_pause_max`, 15 min) para não
virar esconderijo indefinido.

**Janela deslizante em vez de streak consecutivo.** `intruder_hits` é uma lista
de timestamps filtrada por `intruder_window`. E — isto foi um bug que eu
introduzi e os testes pegaram — a lista **não** é limpa num frame sem rosto.
Limpar ali era exatamente o que fazia o flicker de detecção derrotar a trava por
intruso: visto → perdido → visto nunca acumulava.

**Teto absoluto `max_without_face`.** `body_only_start` era reiniciado a cada
rosto detectado, então a janela body-only podia se renovar para sempre. Agora
`last_proof_of_presence` impõe um limite de 90 s desde o último rosto
*verificado*, independente de quantas vezes a janela reiniciou. Tem teste que
alterna body-only com rosto não reconhecido e exige que a trava aconteça.

**`mode` deixou de ser decorativo.** `Config.__post_init__` força
`on_camera_failure="lock"` em `security` e `"warn"` em `convenience`. Tem teste
que verifica que pedir `warn` em modo `security` é ignorado.

**`Observation.__post_init__` rejeita entrada incoerente** (`owner_recognized`
com `faces=0`, `faces` negativo). Erro de adaptador vira exceção, não decisão
silenciosa errada.

### A guarda arquitetural

`tests/test_replay.py::test_agent_contains_no_presence_logic` lê `agent.py` e
falha se qualquer uma destas voltar a aparecer: `absence_start`,
`intruder_streak`, `static_since`, `prev_face_center`, `_body_detect_active`,
`last_face_time`, `locked_until`, `body_only_duration`. É o que impede a
divergência de acontecer de novo.

E `test_every_reason_is_reachable` monta 8 configurações diferentes e falha se
algum valor de `Reason` nunca for retornado por nenhum caminho. Foi assim que
`Reason.SPOOF` deixou de ser vocabulário morto.

---

## 3. B4-2 — Harness FAR/FRR

`lock-on-absence-replay`. Três modos:

```bash
# 1. sintético — sem vídeo, sem câmera, roda em CI
lock-on-absence-replay --synthetic --repeat 8 --far-window 12

# 2. vídeo real com rótulos
lock-on-absence-replay --video mesa.mp4 --labels mesa.csv --model face_model.yml

# 3. gravar cenário: roda a visão UMA vez, itera na PSM sem redecodificar
lock-on-absence-replay --video mesa.mp4 --record mesa.jsonl
lock-on-absence-replay --scenario mesa.jsonl --labels mesa.csv --sweep absence_delay=5,10,20
```

Timestamps vêm de `frame_index / fps`, **nunca** do relógio de parede, então um
clipe de 30 min avalia em segundos e sempre igual. Tem teste de determinismo.

### Definições (explícitas, porque cada um usa FAR/FRR com sentido diferente)

| Métrica | Definição usada aqui |
|---|---|
| **FAR** | fração dos intervalos `intruder` em que a tela **não** travou dentro de `--far-window` |
| **FRR** | fração dos intervalos `owner` em que a tela travou |
| **TTL** | segundos do início de um intervalo `absent` até a trava (mediana e p90) |
| spurious/h | travas durante `owner` por hora de presença do dono |

Formato dos rótulos (`--labels`):

```csv
start_sec,end_sec,truth
0,60,owner
60,90,absent
170,182,intruder
280,330,body_only
```

### Números medidos agora (default, cenário sintético, repeat=8)

```
FAR  intruder missed   0.0%   (0/8 intervals)
FRR  owner rejected    0.0%   (0/32 intervals)
spurious locks/hour   0.00
time-to-lock median  11.0s
time-to-lock p90     11.5s
```

E o tradeoff real que o projeto nunca tinha, com 60% de perda de detecção
(Haar perdendo o rosto em ângulo):

```
sweep intruder_count
     value      FAR      FRR   TTL med
       1.0     0.0%     0.0%     11.0s
       2.0    25.0%     0.0%     11.0s
       3.0    37.5%     0.0%     11.0s
```

Investigando os 2 intervalos perdidos com `intruder_count=2`: tiveram **1
detecção em 8 ticks**. Um hit nunca fecha uma contagem de 2. Ou seja, o gargalo
sob detecção ruim é o `intruder_count`, **não** a janela — variar
`intruder_window` de 1,6 s a 12 s não muda nada. Esse é o tipo de conclusão que
não se obtém por leitura de código.

**Não use isso como validação do produto.** É um cenário sintético; mede a
máquina de estados, não a visão computacional. O número que importa vem de vídeo
real seu, com seus rótulos, na sua iluminação. O harness é a ferramenta; o
dataset é seu trabalho de casa.

### Gate de CI

```bash
lock-on-absence-replay --synthetic --repeat 8 --far-window 12 \
  --fail-if-far-above 0.0 --fail-if-frr-above 0.0
```

Já está no `.github/workflows/ci.yml`, com um segundo gate mais tolerante a 40%
de flicker. Qualquer mudança futura que faça a máquina perder um intruso ou
travar na cara do dono quebra o build. Tem teste que verifica que o gate
**consegue** falhar (`--intruder-count 999` → exit 1), porque gate que não falha
não é gate.

---

## 4. B4-3 — Estrutura de pacote

**Decisões que você pediu que fossem tomadas:**

| | Escolha | Motivo |
|---|---|---|
| Nome de distribuição | `lock-on-absence` | igual ao repo |
| Pacote de import | `lock_on_absence` | o hífen no nome do arquivo antigo impedia `import`, então o módulo era intestável por construção |
| Versão | `lock_on_absence/__init__.py` | fonte única, lida pelo `pyproject` via `dynamic`, pelos `--version` e (deve ser) pelos badges. O drift README-vs-`__init__` aconteceu 3 rodadas seguidas |
| Entry points | 4 nomes longos e explícitos | zero risco de colisão; `loa` é curto mas é o tipo de nome que colide |

```
lock-on-absence            -> lock_on_absence.agent:main
lock-on-absence-enroll     -> lock_on_absence.enroll:main
lock-on-absence-watchdog   -> lock_on_absence.watchdog:main
lock-on-absence-replay     -> lock_on_absence.replay:main
```

`python -m lock_on_absence` também funciona.

`[tool.setuptools] packages = ["lock_on_absence"]` é explícito de propósito, para
o autodiscovery nunca confundir os shims da raiz com módulos do pacote.

`opencv-contrib-python` sozinho na dependência — nunca junto com `opencv-python`,
que entrega o mesmo `cv2` e sobrescreve, fazendo `cv2.face` desaparecer conforme
a ordem de instalação.

---

## 5. O que eu corrigi de quebra (estava no caminho)

**`watchdog.py` reescrito.** Não tinha argparse, então `--help` caía no
`while True` — foi assim que o smoke test o pegou. Além do CLI, corrigi os três
problemas que apontei na revisão:

- **latch**: trava uma vez por episódio de staleness. O original travava a cada
  30 s para sempre depois de qualquer crash do agente, então o usuário não
  conseguia logar para consertar exatamente o problema que causou aquilo.
- **clamp de relógio**: timestamp no futuro agora é `TAMPERED`, não "muito
  fresco". Antes, `echo 9999999999 > watchdog_heartbeat.txt` desligava o
  fail-closed inteiro, e qualquer correção de NTP ou resume de notebook fazia o
  mesmo por acidente.
- **verificação de estado**: `session_is_locked()` via `LockedHint` no Linux e
  `LogonUI.exe` no Windows, para não re-travar sessão já travada.
- `--print-unit` e `--print-task` cospem as definições prontas para instalar.

**Heartbeat gravado em toda iteração**, não só com dono presente. Heartbeat prova
que o **agente** está vivo, não que o **usuário** está lá. Gravar só na presença
fazia o watchdog disparar durante ausências legítimas e correr com o usuário
logo depois de ele voltar. Escrita atômica (`.tmp` + `os.replace`).

**`sd_notify` implementado** (`READY=1` + `WATCHDOG=1` por iteração) e a unit
mudou para `Type=notify`. `Type=simple` com `WatchdogSec=30` e sem notificação
fazia o systemd matar e reiniciar o serviço a cada 30 s, para sempre.

**`safe_face_roi()`** clampa todo recorte. YuNet emite coordenada negativa em
rosto na borda; o slice virava array vazio, `cv2.resize` levantava, e o
`except Exception` genérico matava o agente. Tem teste com 5 casos.

**Calibração do `BodyDetector` movida para `sample_noise()`**, chamado só com dono
confirmado. Antes as amostras eram colhidas dentro de `present()`, que só roda
quando **não** há rosto — a baseline de "micro-movimento normal" era medida
durante ausência. `complete_calibration()` agora exige
`calibration_samples` amostras (antes uma bastava) e retorna `bool` para o
adaptador logar só na transição.

**`EventLogger.lock(reason, ok, **detail)`** unificado. Antes, quando a trava
falhava, o SIEM recebia `lock_failed` **e** `intruder_lock` — um analista veria
um bloqueio bem-sucedido que nunca aconteceu. Agora `ok=False` emite só
`lock_failed`. Todo evento leva `dry_run`.

**`pmset displaysleepnow` removido da cadeia de lock no macOS.** Retornava 0 e
apenas apagava a tela, então `lock_screen()` devolvia `True` com a sessão aberta.
Era o C2 reintroduzido.

**Threshold do `face_model.json` validado** contra `[20, 100]`. O arquivo é
gravável pelo usuário; sem faixa, qualquer processo local escrevia
`{"threshold": 999999}` e todo rosto virava o dono, em silêncio.

**Crash handler falha fechado**: `except Exception` no loop agora trava a tela
antes de sair, em vez de morrer deixando a sessão aberta.

**`--stealth` e `--anti-spoof-timeout` com aviso honesto no `--help`.** O
anti-spoof por movimento continua disponível mas desligado por default e
descrito como heurística fraca, não como liveness. A docstring de
`_check_static_face` explica por que não detecta foto.

---

## 6. Validação executada

```
pip install -e ".[dev]"  (venv virgem)   OK
lock-on-absence --version                lock-on-absence 5.0.0
4 entry points resolvem                  OK
ruff check                               All checks passed
pyflakes                                 (vazio)
pytest                                   71 passed
harness --synthetic --repeat 8           FAR 0.0%  FRR 0.0%  TTL 11.0s
gate consegue falhar                     exit 1 com --intruder-count 999
```

---

## 7. O que ainda falta (não está no Bloco 4)

1. **Dataset real.** O harness sem vídeo seu é um chassi sem motor. Grave 30–60
   min rotulados e publique a tabela no README. Só então o threshold 65 deixa de
   ser chute — hoje ele é um valor herdado de uma configuração LBPH diferente
   (`radius=2, grid 6x6` mudou a escala das distâncias chi-quadrado).
2. **Trocar Haar+LBPH por YuNet+SFace.** `cv2.FaceRecognizerSF` está no OpenCV
   padrão, tem ponto de operação publicado (cosseno ~0,363), e os 5 landmarks do
   YuNet dão yaw real. Isso mata a categoria inteira de bug de threshold e o
   `estimate_angle`, que ainda mede posição no quadro em vez de pose.
3. **Permissão do modelo no Windows.** `chmod 600` funciona no Linux/macOS; no
   Windows precisa de `icacls`.
4. **README.** Não reescrevi — precisa refletir v5.0, os novos comandos, o
   `--mode`, o harness, e remover a afirmação sobre `--meeting-pause` que estava
   invertida. Sincronize o badge de versão com `__init__.py`.
5. **HMAC no `face_model.yml`.** O arquivo de modelo é a fronteira de confiança
   do produto; hoje é um arquivo comum.

---

## 8. Correções aplicadas na auditoria (pós-entrega)

A auditoria profunda do zip encontrou e corrigiu o seguinte antes do commit:

1. **Omissões no §1.** `git rm` também precisa de `tests/test_smoke.py`
   (importava `face_utils` da raiz — que sai — e a suíte nova coletaria ele e
   quebraria com ImportError; sua função foi absorvida por
   `tests/test_replay.py`) e `requirements.txt` (substituído pelo
   `pyproject.toml`; manter seria duas fontes de verdade de dependências).
2. **`--camera-fail-grace` re-adicionado ao CLI.** Existia no v4.1 e sumiu do
   argparse na migração (o `Config` mantinha o default 20.0, então o
   comportamento em default era o mesmo, mas qualquer comando antigo com a flag
   quebrava com `unrecognized arguments`).
3. **API de recovery na máquina.** O bloco de recuperação de câmera do
   `agent.py` mutava `st.camera_fail_streak` / `st.first_camera_fail` direto —
   violava "a máquina é dona do State". Agora é
   `psm.reset_camera_failure(st)`. A guarda arquitetural foi estendida para
   banir **qualquer** mutação direta de `st.<attr> =` no adaptador (regex
   `\bst\.[a-z_]+\s*=(?!=)`), não só os nomes legados.

Validação final executada com esses patches: 71 testes, ruff limpo, pyflakes
limpo, 4 entry points resolvendo, gates FAR/FRR passando (e falhando quando
devem).
