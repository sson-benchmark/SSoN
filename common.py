"""
common.py -- shared protocol, data and model helpers for the odd-one-out pipeline.

Everything that must be IDENTICAL between the baseline evaluation, training and
the post-fine-tuning evaluation lives here: the prompt, the message layout, the
image budget, the answer parser and the SFT target construction.
"""
import hashlib
import json
import os
import re
from pathlib import Path

# --------------------------------------------------------------------------
# Protocol (separate_v1): one user turn, prompt text then images A, B, C, D.
# --------------------------------------------------------------------------
PROMPT = (
    "You will be shown four images labeled A, B, C, and D (in that order). "
    "Each image shows a scene from one camera viewpoint.\n\n"
    "Three of the four images are mutually consistent views of one and the same scene, "
    "taken from different camera positions and angles. One image is inconsistent with that "
    "scene — it cannot be a view of the same scene as the other three. Identify which one.\n\n"
    "Reason step by step, then end your response with ONLY this JSON on the final line:\n"
    '{"odd_one_out": "<A|B|C|D>"}'
)
PROMPT_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()

LETTERS = ["A", "B", "C", "D"]
PROTOCOL = "separate_v1_local"
VERSION = "v3"

# Teacher-forced prefix for the forced-choice metric.
ANSWER_PREFIX = '{"odd_one_out": "'

# Output budget of the protocol (greedy decoding).
PROTOCOL_MAX_TOKENS = 16384

# --------------------------------------------------------------------------
# Image budget: native resolution.
# The bank renders at 1024x1024 (iTHOR/ProcTHOR, RoboTHOR, SAGE, TDW),
# 1100x825 (3D-FRONT) and 1024x576 (Infinigen). MAX_PIXELS is above all of
# them, so the Qwen2.5-VL processor only snaps each side to a multiple of 28 px
# and never downscales (<= 1369 visual tokens per 1024x1024 panel).
# --------------------------------------------------------------------------
MAX_PIXELS = 1_200_000
MIN_PIXELS = 3136
MAX_SEQ = 16384          # guard on the tokenised input length

DEFAULT_BASE = "Qwen/Qwen2.5-VL-7B-Instruct"


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent


def find_bank():
    """BANK_dataset root: $ODDOUT_BANK, else next to / above the code."""
    if os.environ.get("ODDOUT_BANK"):
        return os.environ["ODDOUT_BANK"]
    for cand in (CODE_DIR / "BANK_dataset", CODE_DIR.parent / "BANK_dataset"):
        if cand.is_dir():
            return str(cand)
    return None


BANK_ROOT = find_bank()


def resolve_images(item):
    """The item's four panel paths (A, B, C, D in the item's current order)."""
    if BANK_ROOT and item.get("rel_images"):
        paths = [os.path.join(BANK_ROOT, r) for r in item["rel_images"]]
        if all(os.path.exists(p) for p in paths):
            return paths
    if all(os.path.exists(p) for p in item["images"]):
        return item["images"]
    raise FileNotFoundError("%s: panels not found; set ODDOUT_BANK to the "
                            "BANK_dataset directory" % item.get("qid"))


def load_images(image_paths):
    """RGB PIL images at their original size; resizing is the processor's job.
    Only A..D are opened (question.png / answer_key.png would leak the answer)."""
    from PIL import Image

    out = []
    for p in image_paths:
        with Image.open(p) as im:
            out.append(im.convert("RGB"))
    return out


# --------------------------------------------------------------------------
# Model / processor
# --------------------------------------------------------------------------
def load_processor(base):
    """AutoProcessor at native resolution. Returns (processor, budget dict)."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(base, min_pixels=MIN_PIXELS,
                                         max_pixels=MAX_PIXELS)
    ip = proc.image_processor
    ppt = ip.patch_size ** 2 * ip.merge_size ** 2        # 784 for Qwen2.5-VL
    budget = {"mode": "native", "pixels_per_token": ppt,
              "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS,
              "max_img_tokens": MAX_PIXELS // ppt, "max_seq": MAX_SEQ}
    return proc, budget


def load_model(base, attn="sdpa", device_map=None):
    """Base model in bf16."""
    import torch
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation=attn,
        device_map=device_map if device_map is not None else {"": 0})


def patch_vision_varlen():
    """Run the Qwen2.5-VL ViT window attention as ONE varlen flash-attention
    call per layer instead of one sdpa call per window (torch >= 2.10,
    torch.nn.attention.varlen). Same computation -- on our checks the ViT
    output and the answer logits were bitwise identical -- but ~1.8x faster
    forward+backward when the flash_attn wheel is not installed."""
    import torch
    from torch.nn.attention.varlen import varlen_attn
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as M

    cls = M.Qwen2_5_VLVisionAttention
    if getattr(cls, "_varlen_patched", False):
        return
    orig = cls.forward

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None,
                max_seqlen=None, **kwargs):
        if (M.is_flash_attention_requested(self.config)
                or not hidden_states.is_cuda
                or hidden_states.dtype not in (torch.bfloat16, torch.float16)):
            return orig(self, hidden_states, cu_seqlens,
                        position_embeddings=position_embeddings,
                        max_seqlen=max_seqlen, **kwargs)
        n = hidden_states.shape[0]
        q, k, v = (self.qkv(hidden_states).reshape(n, 3, self.num_heads, -1)
                   .permute(1, 0, 2, 3).unbind(0))
        cos, sin = position_embeddings
        q, k = M.apply_rotary_pos_emb_vision(q, k, cos, sin)
        cu = cu_seqlens.to(torch.int32)
        mx = int((cu[1:] - cu[:-1]).max())
        out = varlen_attn(q.contiguous(), k.contiguous(), v.contiguous(),
                          cu, cu, mx, mx, scale=self.scaling)
        return self.proj(out.reshape(n, -1))

    cls.forward = forward
    cls._varlen_patched = True


def letter_token_ids(tokenizer, prefix=""):
    """Token id of each answer letter (prefix=" ": as in running text)."""
    ids = []
    for L in LETTERS:
        enc = tokenizer.encode(prefix + L, add_special_tokens=False)
        assert len(enc) == 1, "letter %r is not a single token" % L
        ids.append(enc[0])
    return ids


def check_seq(input_ids, qid=""):
    n = int(input_ids.shape[-1])
    assert n <= MAX_SEQ, "%s: %d input tokens > MAX_SEQ=%d" % (qid, n, MAX_SEQ)
    return n


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------
def build_messages(image_paths, target=None):
    """One user turn (prompt text, then the four images in A, B, C, D order);
    if `target` is given, the assistant turn used for SFT. No system prompt is
    supplied by the caller (the Qwen2.5-VL chat template inserts its default
    one, identically for training and evaluation)."""
    assert len(image_paths) == 4
    content = [{"type": "text", "text": PROMPT}]
    for p in image_paths:
        content.append({"type": "image", "image": os.path.abspath(p)})
    msgs = [{"role": "user", "content": content}]
    if target is not None:
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": target}]})
    return msgs


# --------------------------------------------------------------------------
# Answer parsing. parse_strict is the protocol metric: the last non-empty line
# must itself be the JSON object; anything else scores as wrong. The two looser
# parsers are diagnostics only.
# --------------------------------------------------------------------------
_JSON_RE = re.compile(r'\{\s*"odd_one_out"\s*:\s*"?([ABCD])"?\s*\}')
_LENIENT_RE = re.compile(r'\b([ABCD])\b')


def parse_strict(text):
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    if not lines:
        return None
    try:
        obj = json.loads(lines[-1].strip())
    except ValueError:
        return None
    v = obj.get("odd_one_out") if isinstance(obj, dict) else None
    return v if v in LETTERS else None


def parse_fenced(text):
    """Strict, but tolerating a markdown fence and an unquoted letter."""
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    for last in reversed(lines):
        last = last.strip("`").strip()
        if not last:
            continue
        try:
            v = str(json.loads(last).get("odd_one_out", "")).strip().upper()
            return v if v in LETTERS else None
        except Exception:
            m = _JSON_RE.fullmatch(last)
            return m.group(1) if m else None
    return None


def parse_lenient(text):
    """Last JSON answer anywhere, else the last standalone letter."""
    if not text:
        return None
    m = list(_JSON_RE.finditer(text))
    if m:
        return m[-1].group(1)
    m = list(_LENIENT_RE.finditer(text))
    return m[-1].group(1) if m else None


# --------------------------------------------------------------------------
# Panel-order views
# --------------------------------------------------------------------------
def cyclic_views(item):
    """The item in its 4 cyclic rotations: view k shows original panels
    (k, k+1, k+2, k+3) mod 4, so the odd image sits at each of A-D once.
    View 0 is the canonical item (same qid). perm[j] = original panel index
    shown at slot j."""
    out = []
    for k in range(4):
        perm = [(j + k) % 4 for j in range(4)]
        v = dict(item)
        for key in ("images", "rel_images", "image_sha256"):
            if item.get(key):
                v[key] = [item[key][i] for i in perm]
        v["answer"] = LETTERS[perm.index(LETTERS.index(item["answer"]))]
        v["src_qid"] = item.get("src_qid", item["qid"])
        v["view"] = k
        v["perm"] = perm
        v["qid"] = item["qid"] if k == 0 else "%s#c%d" % (item["qid"], k)
        out.append(v)
    return out


def item_perm(item):
    """perm[j] = original panel index at slot j (identity for canonical items)."""
    if item.get("perm"):
        return list(item["perm"])
    m = re.search(r"#p([0-3]{4})$", item.get("qid", ""))
    return [int(c) for c in m.group(1)] if m else [0, 1, 2, 3]


# --------------------------------------------------------------------------
# SFT targets
#
# item["sup"] is the generator's own record of the edit (from the per-dataset
# manifest CSV), in CANONICAL panel letters. build_target turns it into the
# rationale target and rewrites every panel letter for the item's permutation.
# --------------------------------------------------------------------------
_VIEW_SPAN = re.compile(
    r"\b(views?|panels?|images?)(\s+)"
    r"([A-D](?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*|\s*/\s*)[A-D])*)\b",
    re.IGNORECASE)
_LETTER = re.compile(r"[A-D]")
_RUNG = re.compile(r"\s*\([^)]*\brung\b[^)]*\)")


def _join_letters(ls):
    return ls[0] if len(ls) == 1 else "%s and %s" % (", ".join(ls[:-1]), ls[-1])


def remap_panel_letters(text, letter_of):
    """Rewrite the letters inside "view B" / "views A and C" / "views A, B and
    D" spans with letter_of(canonical_index); lists are re-sorted."""
    if not text:
        return text

    def sub(m):
        new = sorted(letter_of(LETTERS.index(c))
                     for c in _LETTER.findall(m.group(3)))
        return m.group(1) + m.group(2) + _join_letters(new)
    return _VIEW_SPAN.sub(sub, text)


def panel_letter_leftovers(text):
    """Capital B-D (or non-article A) tokens outside view spans, i.e. letters
    remap_panel_letters would not rewrite."""
    if not text:
        return []
    rest = _VIEW_SPAN.sub(" ", text)
    return [m.group(0) for m in re.finditer(r"\b[B-D]\b|\bA\b(?!\s+[a-z])", rest)]


def target_text(letter):
    return '{"odd_one_out": "%s"}' % letter


def build_target(item, mode="answer"):
    """Assistant text for SFT.

      answer     {"odd_one_out": "X"}
      rationale  Edit type / Edited object / What changed / Reason lines,
                 then the protocol JSON as the final line.
    """
    if mode == "answer":
        return target_text(item["answer"])

    perm = item_perm(item)

    def letter_of(i):                 # canonical panel index -> current letter
        return LETTERS[perm.index(i)]

    s = item.get("sup") or {}
    lines = []
    if s.get("edit_type"):
        lines.append("Edit type: %s" % s["edit_type"])
    if s.get("object"):
        lines.append("Edited object: %s" % remap_panel_letters(s["object"], letter_of))
    if s.get("edit_description"):
        description = remap_panel_letters(s["edit_description"], letter_of)
        lines.append("What changed: %s" % description.rstrip(".") + ".")
    reason = remap_panel_letters(s.get("gt_reason"), letter_of)
    if not reason:
        reason = "View %s is inconsistent with the other three views." % item["answer"]
    lines.append("Reason: %s" % reason)
    lines.append(target_text(item["answer"]))
    return "\n".join(lines)


def supervision_from_row(row):
    """item["sup"] from one row of a per-dataset manifest CSV (canonical letters)."""
    g = lambda k: (row.get(k) or "").strip()
    obj = g("edited_objects_display")
    if not obj:
        raw = g("edited_objects")
        obj = "the whole scene" if raw == "WholeScene" else raw
    obj = " and ".join(x.strip() for x in obj.split("|") if x.strip())   # swaps
    dv = g("distractor_view").upper()
    if not dv:
        roles = dict(x.split(":", 1) for x in g("panel_roles").split("|") if ":" in x)
        dv = next((k for k, v in roles.items() if v == "distractor"), "")
    return {
        "edit_type": g("edit_type") or g("tier"),
        "object": obj,
        # "(half rung of a 1.572 m move)" is generator bookkeeping, not scene content
        "edit_description": _RUNG.sub("", g("edit_description")),
        "gt_reason": g("gt_reason"),
        "distractor": LETTERS.index(dv) if dv in LETTERS else None,
        "distractor_type": g("distractor_type"),
        "accepted_solutions": g("accepted_solutions"),
    }


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_key(qid, version, model_tag, protocol):
    return hashlib.sha256("|".join([qid, version, model_tag, protocol])
                          .encode()).hexdigest()[:16]


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
