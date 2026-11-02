#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import hashlib
import json
import shutil
import secrets
import time

import torch

import gradio as gr

from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_CSS, build_emoji_palette
from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    download_hf_checkpoint,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    save_wav,
)
from irodori_tts.speaker_inversion import is_speaker_inversion_safetensors_path

MAX_GRADIO_CANDIDATES = 32
GRADIO_AUDIO_COLS_PER_ROW = 8
SHARE_MODE = False


def _default_checkpoint() -> str:
    candidates = sorted(
        [
            *Path(".").glob("**/checkpoint_*.pt"),
            *(
                path
                for path in Path(".").glob("**/checkpoint_*.safetensors")
                if not is_speaker_inversion_safetensors_path(path)
            ),
        ]
    )
    if not candidates:
        return "Aratako/Irodori-TTS-v4-Large-Quantized/int4-weight-only"
    return str(candidates[-1])


def _default_model_device() -> str:
    return default_runtime_device()


def _default_codec_device() -> str:
    return default_runtime_device()


def _precision_choices_for_device(device: str) -> list[str]:
    return list_available_runtime_precisions(device)


def _on_model_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _on_codec_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _on_t_schedule_mode_change(mode: str) -> object:
    return gr.update(interactive=str(mode).strip().lower() == "sway")


def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a float or blank.") from exc


def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an int or blank.") from exc


def _parse_optional_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "null", "off", "disable", "disabled", "base"}:
        return None
    return text


def _format_timings(stage_timings: list[tuple[str, float]], total_to_decode: float) -> str:
    lines = [
        "[timing] ---- request ----",
        *[f"[timing] {name}: {sec * 1000.0:.1f} ms" for name, sec in stage_timings],
        f"[timing] total_to_decode: {total_to_decode:.3f} s",
    ]
    return "\n".join(lines)


def _coerce_gradio_file_path(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("path", "name"):
            candidate = value.get(key)
            if candidate is not None and str(candidate).strip():
                return str(candidate)
        return None
    candidate = getattr(value, "name", None)
    if candidate is not None and str(candidate).strip():
        return str(candidate)
    text = str(value).strip()
    return text or None


def _resolve_ref_wavs(uploaded_audio: object) -> list[str]:
    if uploaded_audio is None:
        return []
    values = uploaded_audio if isinstance(uploaded_audio, (list, tuple)) else [uploaded_audio]
    paths = [_coerce_gradio_file_path(value) for value in values]
    return [path for path in paths if path is not None]


def _resolve_speaker_embedding(
    uploaded_embedding: object,
    speaker_embedding_path_raw: str | None,
) -> str | None:
    uploaded_path = _coerce_gradio_file_path(uploaded_embedding)
    raw_path = None
    if speaker_embedding_path_raw is not None and str(speaker_embedding_path_raw).strip():
        raw_path = str(speaker_embedding_path_raw).strip()
    if uploaded_path is not None and raw_path is not None:
        raise ValueError("Use either speaker embedding upload or speaker embedding path, not both.")
    return uploaded_path if uploaded_path is not None else raw_path


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")

    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        return checkpoint

    resolved = download_hf_checkpoint(checkpoint)
    print(f"[gradio] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
    return str(resolved)


def _build_runtime_key(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        compile_model=False,
        compile_dynamic=False,
    )


def _load_model(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
) -> str:
    if SHARE_MODE and str(checkpoint).strip() != str(_default_checkpoint()).strip():
        raise ValueError("共有モードでは既定Checkpoint以外を指定できません。")
    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
    )
    runtime, reloaded = get_cached_runtime(runtime_key)
    if reloaded:
        status = "loaded model into memory"
    else:
        status = "model already loaded; reused existing runtime"
    status_text = (
        f"{status}\n"
        f"checkpoint: {runtime_key.checkpoint}\n"
        f"model_device: {runtime_key.model_device}\n"
        f"model_precision: {runtime_key.model_precision}\n"
        f"codec_device: {runtime_key.codec_device}\n"
        f"codec_precision: {runtime_key.codec_precision}"
    )
    return status_text



def _split_overlong_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split long text near a natural pause, avoiding tiny trailing fragments."""
    remaining = sentence.strip()
    pieces: list[str] = []
    separators = "、，, "
    while len(remaining) > max_chars:
        lower = max(1, max_chars // 2)
        candidates = [
            (remaining.rfind(mark, lower, max_chars + 1), mark)
            for mark in separators
        ]
        candidates = [(pos, mark) for pos, mark in candidates if pos >= lower]
        if candidates:
            cut, mark = max(candidates, key=lambda item: item[0])
            end = cut + (1 if mark != " " else 0)
        else:
            end = max_chars

        # Avoid leaving a tiny final chunk when the whole remainder barely exceeds
        # the limit. Rebalance that one split instead of producing a short tail.
        min_tail = max(10, max_chars // 3)
        tail_len = len(remaining) - end
        if 0 < tail_len < min_tail:
            target_end = max(lower, len(remaining) - min_tail)
            earlier = [
                (remaining.rfind(mark, lower, target_end + 1), mark)
                for mark in separators
            ]
            earlier = [(pos, mark) for pos, mark in earlier if pos >= lower]
            if earlier:
                cut, mark = max(earlier, key=lambda item: item[0])
                end = cut + (1 if mark != " " else 0)
            else:
                end = target_end

        piece = remaining[:end].strip()
        if not piece:
            end = max_chars
            piece = remaining[:end].strip()
        pieces.append(piece)
        remaining = remaining[end:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _split_long_text(text: str, max_chars: int) -> list[tuple[str, bool]]:
    raw = "" if text is None else str(text).strip()
    if not raw:
        return []
    max_chars = max(20, int(max_chars))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    chunks: list[tuple[str, bool]] = []
    for pi, paragraph in enumerate(paragraphs):
        sentences = [
            x.strip()
            for x in re.findall(r".+?(?:[。！？!?]+[」』】）)]*|$)", paragraph, flags=re.S)
            if x.strip()
        ] or [paragraph]
        paragraph_chunks: list[str] = []
        current = ""
        for sentence in sentences:
            for piece in _split_overlong_sentence(sentence, max_chars):
                if current and len(current) + len(piece) > max_chars:
                    paragraph_chunks.append(current.strip())
                    current = piece
                else:
                    current += piece
        if current.strip():
            paragraph_chunks.append(current.strip())
        for ci, chunk in enumerate(paragraph_chunks):
            chunks.append((chunk, ci == len(paragraph_chunks) - 1 and pi < len(paragraphs) - 1))
    return chunks


def _fingerprint_path(raw_path: str | Path | None) -> str | None:
    """Hash file contents (or all files in a directory) for resume identity."""
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if not path.exists():
        return f"missing:{str(path.resolve())}"
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _cleanup_old_resume_jobs(resume_root: Path, keep_days: int = 30) -> int:
    """Remove only inactive job directories directly beneath the resume root."""
    if not resume_root.exists():
        return 0
    cutoff = time.time() - keep_days * 86400
    removed = 0
    root_resolved = resume_root.resolve()
    for candidate in resume_root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            resolved = candidate.resolve()
            if resolved.parent != root_resolved or candidate.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(candidate)
            removed += 1
        except OSError as exc:
            print(f"[large-long] 古い再開データを削除できません: {candidate} ({exc})", flush=True)
    return removed


def _chunk_cache_key(chunk_text: str, generation_settings: dict) -> str:
    """Stable cache key for one chunk and every setting that affects its audio."""
    payload = {
        "text": str(chunk_text),
        "generation_settings": generation_settings,
        "cache_format_version": 1,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _on_long_mode_change(enabled: bool):
    return gr.update(value=1, interactive=not bool(enabled))


def _preview_long_text(text: str, max_chars: int) -> str:
    chunks=_split_long_text(text,int(max_chars))
    if not chunks: return "Text is empty."
    out=[f"{len(chunks)} chunk(s)"]
    for i,(chunk,pb) in enumerate(chunks,1):
        out.append(f"[{i:02d}] {len(chunk)} chars{' [paragraph break]' if pb else ''}\n{chunk}")
    return "\n\n".join(out)

def _run_long_generation(
    checkpoint, model_device, model_precision, codec_device, codec_precision,
    text, uploaded_audio, uploaded_speaker_embedding, speaker_embedding_path_raw,
    num_steps, num_candidates, seed_raw, seconds_raw, duration_scale,
    t_schedule_mode, sway_coeff, cfg_guidance_mode, cfg_scale_text,
    cfg_scale_speaker, cfg_scale_raw, cfg_min_t, cfg_max_t, context_kv_cache,
    truncation_factor_raw, rescale_k_raw, rescale_sigma_raw,
    speaker_kv_scale_raw, speaker_kv_min_t_raw, speaker_kv_max_layers_raw,
    lora_adapter_raw, max_chunk_chars, chunk_pause, paragraph_pause,
    resume_enabled, delete_chunks_after_success=False, preview_only=False, progress=gr.Progress()):
    chunks = _split_long_text(text, int(max_chunk_chars))
    if not chunks:
        raise ValueError("text is required.")
    all_chunks = chunks
    if bool(preview_only):
        chunks = chunks[:2]
    progress(0, desc=f"準備中… {len(chunks)}チャンク" + ("（お試し）" if preview_only else f" / 全{len(all_chunks)}チャンク"))
    if int(num_candidates) != 1:
        raise ValueError("Long Text Mode requires Num Candidates = 1.")

    if SHARE_MODE:
        if str(checkpoint).strip() != str(_default_checkpoint()).strip():
            raise ValueError("共有モードではCheckpointの任意パス指定は許可されていません。")
        if speaker_embedding_path_raw and str(speaker_embedding_path_raw).strip():
            raise ValueError("共有モードではSpeaker Embedding Pathの直接指定は許可されていません。アップロードを使用してください。")

    parsed_num_steps = _parse_optional_int(num_steps, "num_steps")
    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")
    manual_seconds = _parse_optional_float(seconds_raw, "seconds")
    lora_adapter = _parse_optional_str(lora_adapter_raw)

    ref_wavs = _resolve_ref_wavs(uploaded_audio)
    speaker_embedding = _resolve_speaker_embedding(uploaded_speaker_embedding, speaker_embedding_path_raw)
    if ref_wavs and speaker_embedding is not None:
        raise ValueError("Reference audio and speaker embedding are mutually exclusive.")
    no_ref = not ref_wavs and speaker_embedding is None

    # Resolve first: resume identity uses the actual local checkpoint path, not the HF/model label.
    runtime_key = _build_runtime_key(checkpoint, model_device, model_precision, codec_device, codec_precision)
    resolved_checkpoint = str(Path(runtime_key.checkpoint).resolve())
    resume_root = Path("gradio_outputs_large_long") / "resume"
    resume_root.mkdir(parents=True, exist_ok=True)
    removed_old = _cleanup_old_resume_jobs(resume_root, keep_days=30)

    chunk_hashes = [hashlib.sha256(chunk.encode("utf-8")).hexdigest() for chunk, _ in chunks]
    # Persist an automatically chosen seed for this exact input/settings set so an interrupted
    # blank-seed job can be resumed with the same seed on the next launch.
    seed_basis = {
        # Keep the automatic seed stable when only part of the document text changes.
        # Per-chunk text is already included in each chunk cache key.
        "checkpoint": resolved_checkpoint,
        "reference_audio_fingerprints": [_fingerprint_path(path) for path in ref_wavs],
        "speaker_embedding_fingerprint": _fingerprint_path(speaker_embedding),
        "lora_fingerprint": _fingerprint_path(lora_adapter),
        "model_device": str(model_device), "model_precision": str(model_precision),
        "codec_device": str(codec_device), "codec_precision": str(codec_precision),
        "num_steps": parsed_num_steps, "duration_scale": float(duration_scale),
        "seconds": manual_seconds, "cfg_guidance_mode": str(cfg_guidance_mode),
        "cfg_scale_text": float(cfg_scale_text), "cfg_scale_speaker": float(cfg_scale_speaker),
        "cfg_scale": cfg_scale, "cfg_min_t": float(cfg_min_t), "cfg_max_t": float(cfg_max_t),
        "truncation_factor": truncation_factor, "rescale_k": rescale_k, "rescale_sigma": rescale_sigma,
        "context_kv_cache": bool(context_kv_cache), "speaker_kv_scale": speaker_kv_scale,
        "speaker_kv_min_t": speaker_kv_min_t, "speaker_kv_max_layers": speaker_kv_max_layers,
        "t_schedule_mode": str(t_schedule_mode), "sway_coeff": float(sway_coeff),
    }
    if seed is None:
        if bool(resume_enabled):
            seed_key = hashlib.sha256(json.dumps(seed_basis, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            seed_file = resume_root / "auto_seeds" / f"{seed_key}.json"
            seed_file.parent.mkdir(parents=True, exist_ok=True)
            if seed_file.exists():
                seed = int(json.loads(seed_file.read_text(encoding="utf-8"))["seed"])
            else:
                seed = secrets.randbelow(2**31 - 1)
                _atomic_write_json(seed_file, {"seed": seed, "created": datetime.now().isoformat()})
        else:
            seed = secrets.randbelow(2**31 - 1)
        print(f"[large-long] Seed未指定のため自動決定: {seed}", flush=True)

    # Build the actual SamplingRequest arguments once. The resume identity is derived
    # from this same object so it cannot silently drift from the generation call.
    sampling_args = {
        "ref_wav": None, "ref_wavs": ref_wavs or None, "ref_latent": None,
        "ref_embed": speaker_embedding, "no_ref": bool(no_ref),
        "ref_normalize_db": -16.0, "ref_ensure_max": True,
        "num_candidates": 1, "decode_mode": "sequential",
        "seconds": manual_seconds, "duration_scale": float(duration_scale),
        "max_ref_seconds": None, "max_text_len": None, "num_steps": parsed_num_steps,
        "seed": int(seed), "cfg_guidance_mode": str(cfg_guidance_mode),
        "cfg_scale_text": float(cfg_scale_text), "cfg_scale_speaker": float(cfg_scale_speaker),
        "cfg_scale": cfg_scale, "cfg_min_t": float(cfg_min_t), "cfg_max_t": float(cfg_max_t),
        "truncation_factor": truncation_factor, "rescale_k": rescale_k,
        "rescale_sigma": rescale_sigma, "context_kv_cache": bool(context_kv_cache),
        "speaker_kv_scale": speaker_kv_scale, "speaker_kv_min_t": speaker_kv_min_t,
        "speaker_kv_max_layers": speaker_kv_max_layers,
        "t_schedule_mode": str(t_schedule_mode), "sway_coeff": float(sway_coeff),
        "trim_tail": True, "lora_adapter": lora_adapter,
    }
    generation_settings = {}
    for key, value in sampling_args.items():
        if key == "ref_wavs":
            generation_settings[key] = [_fingerprint_path(path) for path in ref_wavs]
        elif key == "ref_embed":
            generation_settings[key] = _fingerprint_path(speaker_embedding)
        elif key == "lora_adapter":
            generation_settings[key] = {"path": str(lora_adapter), "fingerprint": _fingerprint_path(lora_adapter)} if lora_adapter else None
        else:
            generation_settings[key] = value
    generation_settings.update({
        "checkpoint_path": resolved_checkpoint,
        "model_device": str(model_device), "model_precision": str(model_precision),
        "codec_device": str(codec_device), "codec_precision": str(codec_precision),
    })
    # Keep a shared per-chunk cache. The number of chunks and whole-document hash
    # deliberately do not participate: preview chunks can be reused by full runs.
    job_data = {
        "generation_settings": generation_settings,
        "chunk_hashes": chunk_hashes,
        "splitter_version": 2,
    }
    job_id = hashlib.sha256(json.dumps(generation_settings, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_dir = resume_root / "chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = resume_root / "jobs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(manifest_dir / f"{job_id}.json", job_data)
    chunk_files = [cache_dir / f"{_chunk_cache_key(chunk, generation_settings)}.pt" for chunk, _ in chunks]

    runtime, reloaded = get_cached_runtime(runtime_key)
    parts = []
    sample_rate = None
    messages = [f"seed_used: {seed}", f"old_resume_folders_removed: {removed_old}"]
    timings = []
    total = 0.0
    resumed_count = 0

    for idx, (chunk, pbreak) in enumerate(chunks, 1):
        progress((idx - 1) / len(chunks), desc=f"{idx}/{len(chunks)} チャンク処理中…")
        chunk_file = chunk_files[idx - 1]
        resumed = False
        if bool(resume_enabled) and chunk_file.exists():
            try:
                saved = torch.load(chunk_file, map_location="cpu", weights_only=True)
                if not isinstance(saved, dict) or saved.get("text") != chunk or saved.get("cache_key") != chunk_file.stem:
                    raise ValueError("保存データのチャンク本文または設定キーが一致しません")
                audio = saved.get("audio")
                if not isinstance(audio, torch.Tensor):
                    raise ValueError("保存音声がTensorではありません")
                audio = audio.float()
                sr = int(saved["sample_rate"])
                if sr <= 0 or (sample_rate is not None and sr != sample_rate):
                    raise ValueError("サンプルレートが不正または一致しません")
                sample_rate = sr
                resumed_count += 1
                resumed = True
                print(f"[large-long] reuse chunk {idx}/{len(chunks)} ({chunk_file.stem})", flush=True)
                messages.append(f"chunk {idx}: reused cache {chunk_file.stem}")
            except Exception as exc:
                print(f"[large-long] 壊れた保存分を破棄しました: chunk {idx} ({exc})", flush=True)
                try:
                    chunk_file.unlink(missing_ok=True)
                except OSError as unlink_exc:
                    raise RuntimeError(f"チャンク {idx} の破損ファイルを削除できません: {unlink_exc}") from unlink_exc

        if not resumed:
            print(f"[large-long] generate chunk {idx}/{len(chunks)} ({len(chunk)} chars)", flush=True)
            result = runtime.synthesize(
                SamplingRequest(text=chunk, **sampling_args),
                log_fn=lambda msg: print(msg, flush=True))
            audio = result.audios[0].float().detach().cpu()
            sample_rate = int(result.sample_rate)
            messages += [f"chunk {idx}: {m}" for m in result.messages]
            timings += [(f"chunk{idx}:{n}", sec) for n, sec in result.stage_timings]
            total += float(result.total_to_decode)
            if bool(resume_enabled):
                tmp_file = chunk_file.with_suffix(".pt.tmp")
                try:
                    torch.save({"audio": audio, "sample_rate": sample_rate, "text": chunk,
                                "cache_key": chunk_file.stem, "paragraph_break": bool(pbreak)}, tmp_file)
                    tmp_file.replace(chunk_file)
                finally:
                    if tmp_file.exists():
                        tmp_file.unlink()
                print(f"[large-long] saved chunk cache: {chunk_file}", flush=True)

        parts.append(audio)
        if idx < len(chunks):
            pause = float(paragraph_pause if pbreak else chunk_pause)
            if pause > 0:
                parts.append(torch.zeros((*audio.shape[:-1], max(1, round(sample_rate * pause))), dtype=audio.dtype))
        progress(idx / len(chunks), desc=f"{idx}/{len(chunks)} 完了")

    progress(1.0, desc="音声を結合して保存中…")
    combined = torch.cat(parts, dim=-1)
    out_dir = Path("gradio_outputs_large_long")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_prefix = "large_long_preview" if preview_only else "large_long"
    out_path = save_wav(out_dir / f"{output_prefix}_{stamp}.wav", combined, sample_rate)

    # Optional deletion is scoped to the exact per-chunk cache files used by this run
    # and happens only after save_wav returned successfully.
    chunks_deleted = False
    if bool(delete_chunks_after_success) and not bool(preview_only):
        safe_cache_root = cache_dir.resolve()
        for cached_file in chunk_files:
            safe_file = cached_file.resolve()
            if safe_file.parent == safe_cache_root and safe_file.is_file():
                safe_file.unlink()
                chunks_deleted = True

    progress(1.0, desc="完了しました")
    detail = "\n".join([
        "Irodori-TTS v4 Large LONG / Preview" if preview_only else "Irodori-TTS v4 Large LONG / Resume",
        "runtime: reloaded" if reloaded else "runtime: reused",
        f"settings_id: {job_id}", f"seed_used: {seed}",
        f"generated_chunks: {len(chunks)}", f"total_text_chunks: {len(all_chunks)}",
        f"reused_chunks: {resumed_count}", f"deleted_chunks_after_success: {chunks_deleted}",
        f"chunk_cache_folder: {cache_dir}",
        f"saved: {out_path}", *messages,
    ])
    updates = [gr.update(value=str(out_path), visible=True)]
    updates += [gr.update(value=None, visible=False) for _ in range(MAX_GRADIO_CANDIDATES - 1)]
    return (*updates, detail, _format_timings(timings, total))

def _dispatch_with_draft(draft_mode, draft_steps, *args):
    # _dispatch_generation receives num_steps as argument index 10 after long_mode.
    args = list(args)
    if bool(draft_mode):
        steps = _parse_optional_int(str(draft_steps), "下書きSteps")
        if steps is None or steps < 1:
            raise ValueError("下書きStepsは1以上を指定してください。")
        args[10] = str(steps)
    return _dispatch_generation(*args)


def _dispatch_preview_with_draft(draft_mode, draft_steps, *args):
    args = list(args)
    if bool(draft_mode):
        steps = _parse_optional_int(str(draft_steps), "下書きSteps")
        if steps is None or steps < 1:
            raise ValueError("下書きStepsは1以上を指定してください。")
        args[10] = str(steps)
    # preview_only is the final optional argument to _dispatch_generation.
    args.append(True)
    return _dispatch_generation(*args)


def _dispatch_generation(long_mode, checkpoint, model_device, model_precision, codec_device,
    codec_precision, text, uploaded_audio, uploaded_speaker_embedding, speaker_embedding_path_raw,
    num_steps, num_candidates, seed_raw, seconds_raw, duration_scale, t_schedule_mode, sway_coeff,
    cfg_guidance_mode, cfg_scale_text, cfg_scale_speaker, cfg_scale_raw, cfg_min_t, cfg_max_t,
    context_kv_cache, truncation_factor_raw, rescale_k_raw, rescale_sigma_raw,
    speaker_kv_scale_raw, speaker_kv_min_t_raw, speaker_kv_max_layers_raw, lora_adapter_raw,
    max_chunk_chars, chunk_pause, paragraph_pause, resume_enabled, delete_chunks_after_success=False, preview_only=False):
    common=(checkpoint,model_device,model_precision,codec_device,codec_precision,text,uploaded_audio,
        uploaded_speaker_embedding,speaker_embedding_path_raw,num_steps,num_candidates,seed_raw,
        seconds_raw,duration_scale,t_schedule_mode,sway_coeff,cfg_guidance_mode,cfg_scale_text,
        cfg_scale_speaker,cfg_scale_raw,cfg_min_t,cfg_max_t,context_kv_cache,truncation_factor_raw,
        rescale_k_raw,rescale_sigma_raw,speaker_kv_scale_raw,speaker_kv_min_t_raw,
        speaker_kv_max_layers_raw,lora_adapter_raw)
    if bool(long_mode):
        return _run_long_generation(*common, max_chunk_chars, chunk_pause, paragraph_pause,
                                    resume_enabled, delete_chunks_after_success, preview_only)
    if preview_only:
        raise ValueError("お試し生成は長文モードを有効にしてください。")
    return _run_generation(*common)



def _list_voice_presets() -> list[str]:
    root = Path("irodori_voice_presets")
    if not root.exists():
        return []
    names = []
    for path in root.iterdir():
        if path.is_dir() and (path / "preset.json").exists():
            names.append(path.name)
    return sorted(names, key=str.lower)


def _refresh_voice_presets(current: str | None = None):
    choices = _list_voice_presets()
    current = "" if current is None else str(current).strip()
    value = current if current in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value), (
        f"{len(choices)}件のプリセットを検出しました。" if choices else "保存済みプリセットはありません。"
    )


def _safe_preset_name(name: str) -> str:
    name = "" if name is None else str(name).strip()
    if not name:
        raise ValueError("Preset name is required.")
    safe = re.sub(r"[^0-9A-Za-zぁ-んァ-ヶ一-龠_-]+", "_", name).strip("_")
    if not safe:
        raise ValueError("Preset name is invalid.")
    return safe


def _save_voice_preset(
    preset_name: str,
    uploaded_audio: object,
    seed_raw: str,
    max_chunk_chars: int,
    chunk_pause: float,
    paragraph_pause: float,
    duration_scale: float,
) -> str:
    safe = _safe_preset_name(preset_name)
    seed = _parse_optional_int(seed_raw, "seed")
    if seed is None:
        raise ValueError("Seedを入力してからプリセットを保存してください。")

    root = Path("irodori_voice_presets")
    root.mkdir(parents=True, exist_ok=True)
    preset_dir = root / safe
    staging_dir = root / f".{safe}.tmp-{secrets.token_hex(4)}"
    backup_dir = root / f".{safe}.backup-{secrets.token_hex(4)}"
    audio_dir = staging_dir / "reference_audio"
    audio_dir.mkdir(parents=True, exist_ok=False)

    try:
        refs = _resolve_ref_wavs(uploaded_audio)
        copied = []
        for i, raw in enumerate(refs, 1):
            src = Path(raw)
            if not src.is_file():
                raise FileNotFoundError(f"Reference audio not found: {src}")
            target = audio_dir / f"{i:02d}_{src.name}"
            shutil.copy2(src, target)
            copied.append(str((preset_dir / "reference_audio" / target.name).resolve()))

        data = {
            "name": safe,
            "seed": int(seed),
            "max_chunk_chars": int(max_chunk_chars),
            "chunk_pause": float(chunk_pause),
            "paragraph_pause": float(paragraph_pause),
            "duration_scale": float(duration_scale),
            "reference_audio": copied,
        }
        _atomic_write_json(staging_dir / "preset.json", data)

        # Keep the old preset intact until the complete new preset has been staged.
        if preset_dir.exists():
            preset_dir.replace(backup_dir)
        try:
            staging_dir.replace(preset_dir)
        except Exception:
            if backup_dir.exists() and not preset_dir.exists():
                backup_dir.replace(preset_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if backup_dir.exists() and preset_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

    choices = _list_voice_presets()
    return (
        f"保存しました: {safe}\nReference Audio: {len(copied)} file(s)\nSeed: {seed}",
        gr.update(choices=choices, value=safe),
    )


def _load_voice_preset(preset_name: str):
    safe = _safe_preset_name(preset_name)
    path = Path("irodori_voice_presets") / safe / "preset.json"
    if not path.exists():
        raise ValueError(f"Preset not found: {safe}")
    data = json.loads(path.read_text(encoding="utf-8"))
    refs = [p for p in data.get("reference_audio", []) if Path(p).exists()]
    return (
        gr.update(value=refs),
        str(data["seed"]),
        int(data.get("max_chunk_chars", 120)),
        float(data.get("chunk_pause", 0.30)),
        float(data.get("paragraph_pause", 0.70)),
        float(data.get("duration_scale", 1.0)),
        f"読み込みました: {safe}\nReference Audio: {len(refs)} file(s)\nSeed: {data['seed']}",
    )


def _run_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    text: str,
    uploaded_audio: object,
    uploaded_speaker_embedding: object,
    speaker_embedding_path_raw: str,
    num_steps: str | None,
    num_candidates: int,
    seed_raw: str,
    seconds_raw: str,
    duration_scale: float,
    t_schedule_mode: str,
    sway_coeff: float,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    speaker_kv_scale_raw: str,
    speaker_kv_min_t_raw: str,
    speaker_kv_max_layers_raw: str,
    lora_adapter_raw: str,
) -> tuple[object, ...]:
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    if SHARE_MODE:
        if str(checkpoint).strip() != str(_default_checkpoint()).strip():
            raise ValueError("共有モードではCheckpointの任意パス指定は許可されていません。")
        if speaker_embedding_path_raw and str(speaker_embedding_path_raw).strip():
            raise ValueError("共有モードではSpeaker Embedding Pathの直接指定は許可されていません。アップロードを使用してください。")

    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
    )

    if str(text).strip() == "":
        raise ValueError("text is required.")
    requested_candidates = int(num_candidates)
    if requested_candidates <= 0:
        raise ValueError("num_candidates must be >= 1.")
    if requested_candidates > MAX_GRADIO_CANDIDATES:
        raise ValueError(f"num_candidates must be <= {MAX_GRADIO_CANDIDATES}.")

    parsed_num_steps = _parse_optional_int(num_steps, "num_steps")
    if parsed_num_steps is not None and parsed_num_steps < 1:
        raise ValueError("num_steps must be >= 1 or blank.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")
    manual_seconds = _parse_optional_float(seconds_raw, "seconds")
    lora_adapter = _parse_optional_str(lora_adapter_raw)

    ref_wavs = _resolve_ref_wavs(uploaded_audio)
    speaker_embedding = _resolve_speaker_embedding(
        uploaded_embedding=uploaded_speaker_embedding,
        speaker_embedding_path_raw=speaker_embedding_path_raw,
    )
    if ref_wavs and speaker_embedding is not None:
        raise ValueError("Reference audio and speaker embedding are mutually exclusive.")
    no_ref = not ref_wavs and speaker_embedding is None
    ref_normalize_db = -16.0
    ref_ensure_max = True

    runtime, reloaded = get_cached_runtime(runtime_key)
    stdout_log(f"[gradio] runtime: {'reloaded' if reloaded else 'reused'}")
    stdout_log(
        (
            "[gradio] request: model_device={} model_precision={} codec_device={} codec_precision={} "
            "mode={} schedule={} sway_coeff={} seconds={} duration_scale={} steps={} seed={} no_ref={} candidates={}"
        ).format(
            model_device,
            model_precision,
            codec_device,
            codec_precision,
            cfg_guidance_mode,
            t_schedule_mode,
            sway_coeff,
            "auto" if manual_seconds is None else manual_seconds,
            duration_scale,
            num_steps,
            "random" if seed is None else seed,
            no_ref,
            requested_candidates,
        )
    )
    if speaker_embedding is not None:
        stdout_log(f"[gradio] speaker_embedding: {speaker_embedding}")
    elif ref_wavs:
        stdout_log(f"[gradio] reference clips: {len(ref_wavs)}")

    result = runtime.synthesize(
        SamplingRequest(
            text=str(text),
            ref_wav=None,
            ref_wavs=ref_wavs or None,
            ref_latent=None,
            ref_embed=speaker_embedding,
            no_ref=bool(no_ref),
            ref_normalize_db=ref_normalize_db,
            ref_ensure_max=bool(ref_ensure_max),
            num_candidates=requested_candidates,
            decode_mode="sequential",
            seconds=manual_seconds,
            duration_scale=float(duration_scale),
            max_ref_seconds=None,
            max_text_len=None,
            num_steps=parsed_num_steps,
            seed=None if seed is None else int(seed),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            t_schedule_mode=str(t_schedule_mode),
            sway_coeff=float(sway_coeff),
            trim_tail=True,
            lora_adapter=lora_adapter,
        ),
        log_fn=stdout_log,
    )

    out_dir = Path("gradio_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_paths: list[str] = []
    for i, audio in enumerate(result.audios, start=1):
        out_path = save_wav(
            out_dir / f"sample_{stamp}_{i:03d}.wav",
            audio.float(),
            result.sample_rate,
        )
        out_paths.append(str(out_path))

    runtime_msg = "runtime: reloaded" if reloaded else "runtime: reused"
    detail_lines = [
        runtime_msg,
        f"seed_used: {result.used_seed}",
        f"candidates: {len(result.audios)}",
        *[f"saved[{i}]: {path}" for i, path in enumerate(out_paths, start=1)],
        *result.messages,
    ]
    detail_text = "\n".join(detail_lines)
    timing_text = _format_timings(result.stage_timings, result.total_to_decode)
    stdout_log(f"[gradio] saved {len(out_paths)} candidates")

    audio_updates: list[object] = []
    for i in range(MAX_GRADIO_CANDIDATES):
        if i < len(out_paths):
            audio_updates.append(gr.update(value=out_paths[i], visible=True))
        else:
            audio_updates.append(gr.update(value=None, visible=False))
    return (*audio_updates, detail_text, timing_text)


def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "cleared loaded model from memory"


def build_ui(share_mode: bool = False) -> gr.Blocks:
    default_checkpoint = _default_checkpoint()
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    with gr.Blocks(title="Irodori-TTS Gradio") as demo:
        gr.Markdown("# Irodori-TTS v4 Large LONG - Voice Stable")
        if share_mode:
            gr.Markdown("⚠️ **共有モード警告：この画面はインターネット経由で第三者がアクセスできる可能性があります。Checkpointの直接指定とサーバー側Speaker Embedding Pathは無効化されています。信頼できる相手にのみ共有してください。**")
        gr.Markdown(
            "Reference-audio cloning / Speaker Inversion UI. "
            "Irodori-TTS-v4 Large Quantized (int4-weight-only) is used by default when no local checkpoint is found; unchanged settings reuse the cached runtime."
        )

        with gr.Row():
            checkpoint = gr.Textbox(
                label="Checkpoint (.pt/.safetensors or HF repo id)",
                value=default_checkpoint,
                scale=4,
                visible=not share_mode,
                interactive=not share_mode,
            )
            model_device = gr.Dropdown(
                label="Model Device",
                choices=device_choices,
                value=default_model_device,
                scale=1,
            )
            model_precision = gr.Dropdown(
                label="Model Precision",
                choices=model_precision_choices,
                value=model_precision_choices[0],
                scale=1,
            )
            codec_device = gr.Dropdown(
                label="Codec Device",
                choices=device_choices,
                value=default_codec_device,
                scale=1,
            )
            codec_precision = gr.Dropdown(
                label="Codec Precision",
                choices=codec_precision_choices,
                value=codec_precision_choices[0],
                scale=1,
            )

        with gr.Row():
            load_model_btn = gr.Button("Load Model")
            clear_cache_btn = gr.Button("Unload Model")
            clear_cache_msg = gr.Textbox(label="Model Status", interactive=False)

        with gr.Column():
            text = gr.Textbox(
                label="Text",
                lines=6,
                elem_id="irodori-text-input",
            )
            build_emoji_palette(text, open=False)
        with gr.Tabs():
            with gr.Tab("Reference Audio"):
                gr.Markdown(
                    "**Long-reference tip:** Upload multiple clean, shorter clips from the same "
                    "speaker and arrange them in the desired order. This matches v4-Small "
                    "training. A single uninterrupted long recording is accepted but has not "
                    "been evaluated."
                )
                uploaded_audio = gr.File(
                    label=("Reference Audio Uploads (optional; concatenated in displayed order)"),
                    type="filepath",
                    file_count="multiple",
                    file_types=["audio"],
                    allow_reordering=True,
                )
            with gr.Tab("Speaker Embedding"):
                with gr.Row():
                    uploaded_speaker_embedding = gr.File(
                        label="Speaker Embedding Upload (.speaker.safetensors, optional)",
                        type="filepath",
                        file_count="single",
                        scale=1,
                    )
                    speaker_embedding_path_raw = gr.Textbox(
                        label="Speaker Embedding Path (.speaker.safetensors, optional)",
                        value="",
                        scale=1,
                        visible=not share_mode,
                        interactive=not share_mode,
                    )

        with gr.Accordion("Voice Preset / 声プリセット", open=False):
            gr.Markdown("Reference Audio と Seed、長文設定を名前付きで保存・再読込します。")
            initial_presets = _list_voice_presets()
            preset_name = gr.Dropdown(
                label="Preset / 保存済みプリセット",
                choices=initial_presets,
                value=(initial_presets[0] if initial_presets else None),
                allow_custom_value=True,
                info="保存済みプリセットを一覧から選択できます。新規保存時は名前を直接入力できます。",
            )
            with gr.Row():
                save_preset_btn = gr.Button("Save Preset / 保存")
                load_preset_btn = gr.Button("Load Preset / 読込")
                refresh_preset_btn = gr.Button("Refresh List / 一覧更新")
            preset_status = gr.Textbox(label="Preset Status", interactive=False)

        with gr.Accordion("Long Text Mode / 長文モード", open=True):
            long_mode = gr.Checkbox(label="Enable Long Text Mode / 長文モードを有効化", value=True)
            with gr.Row():
                max_chunk_chars = gr.Slider(label="Max characters per chunk / 最大文字数", minimum=40, maximum=300, value=120, step=10)
                chunk_pause = gr.Slider(label="Pause between chunks (sec) / 分割間", minimum=0.0, maximum=2.0, value=0.30, step=0.05)
                paragraph_pause = gr.Slider(label="Pause after paragraph (sec) / 段落間", minimum=0.0, maximum=3.0, value=0.70, step=0.05)
            preview_btn = gr.Button("Preview Split / 分割を確認")
            split_preview = gr.Textbox(label="Split Preview / 分割プレビュー", lines=10, interactive=False)
            resume_enabled = gr.Checkbox(
                label="途中保存・途中再開を有効化 / Resume",
                value=True,
                info="各チャンク生成後に自動保存。中断後も同じ入力・設定なら続きから再開します。",
            )
            delete_chunks_after_success = gr.Checkbox(
                label="本番生成の保存成功後に使用チャンクを削除",
                value=False,
                info="既定ではチャンクキャッシュを残し、お試し生成分も再利用します。オンにすると本番で使用したチャンクを削除します。",
            )
            gr.Markdown(
                "長文モードでは候補数を1に固定します。Seedが空欄の場合は自動で決定し、ログに表示します。 "
                "間の長さだけを変更した場合は、保存済みチャンクを再利用して結合し直します。 "
                "Generate中は現在のチャンク数と進捗バーを表示します。30日以上未更新の再開データは自動整理されます。"
            )

        with gr.Accordion("Sampling", open=True):
            with gr.Row():
                num_steps = gr.Textbox(
                    label="Num Steps",
                    value="",
                    placeholder="Auto (RF: 40, MeanFlow: 4)",
                    info="Blank: model default (RF: 40, MeanFlow: 4).",
                )
                draft_mode = gr.Checkbox(
                    label="下書きモード（少ないStepsで高速確認）",
                    value=False,
                    info="オンにすると下書き用Stepsを使用します。オフなら通常のSteps設定です。",
                )
                draft_steps = gr.Number(label="下書きSteps", value=12, precision=0, minimum=1, maximum=100, visible=True)
                num_candidates = gr.Slider(
                    label="Num Candidates",
                    minimum=1,
                    maximum=MAX_GRADIO_CANDIDATES,
                    value=1,
                    step=1,
                    interactive=False,
                )
                seed_raw = gr.Textbox(label="Seed (blank=random)", value="")
                seconds_raw = gr.Textbox(label="Seconds (blank=auto)", value="")
                duration_scale = gr.Slider(
                    label="Duration Scale",
                    minimum=0.5,
                    maximum=1.5,
                    value=1.0,
                    step=0.01,
                )

            with gr.Row():
                t_schedule_mode = gr.Dropdown(
                    label="Time Schedule",
                    choices=["linear", "sway"],
                    value="linear",
                )
                sway_coeff = gr.Slider(
                    label="Sway Coeff",
                    minimum=-1.0,
                    maximum=1.5,
                    value=-1.0,
                    step=0.1,
                    interactive=False,
                )

            with gr.Row():
                cfg_guidance_mode = gr.Dropdown(
                    label="CFG Guidance Mode",
                    choices=["independent", "joint", "alternating"],
                    value="independent",
                )
                cfg_scale_text = gr.Slider(
                    label="CFG Scale Text",
                    minimum=0.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.1,
                )
                cfg_scale_speaker = gr.Slider(
                    label="CFG Scale Speaker",
                    minimum=0.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.1,
                )

        with gr.Accordion("Advanced (Optional)", open=False):
            cfg_scale_raw = gr.Textbox(label="CFG Scale Override (optional)", value="")
            with gr.Row():
                cfg_min_t = gr.Number(label="CFG Min t", value=0.5)
                cfg_max_t = gr.Number(label="CFG Max t", value=1.0)
                context_kv_cache = gr.Checkbox(label="Context KV Cache", value=True)
            with gr.Row():
                truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")
            with gr.Row():
                speaker_kv_scale_raw = gr.Textbox(label="Speaker KV Scale (optional)", value="")
                speaker_kv_min_t_raw = gr.Textbox(label="Speaker KV Min t (optional)", value="0.9")
                speaker_kv_max_layers_raw = gr.Textbox(
                    label="Speaker KV Max Layers (optional)", value=""
                )
            lora_adapter_raw = gr.Textbox(label="LoRA Adapter Directory (optional)", value="")

        with gr.Row():
            generate_btn = gr.Button("Generate / 本番生成", variant="primary")
            preview_generate_btn = gr.Button("お試し生成（最初の1〜2チャンク）")

        out_audios: list[gr.Audio] = []
        num_rows = (
            MAX_GRADIO_CANDIDATES + GRADIO_AUDIO_COLS_PER_ROW - 1
        ) // GRADIO_AUDIO_COLS_PER_ROW
        with gr.Column():
            for row_idx in range(num_rows):
                with gr.Row():
                    for col_idx in range(GRADIO_AUDIO_COLS_PER_ROW):
                        i = row_idx * GRADIO_AUDIO_COLS_PER_ROW + col_idx
                        if i >= MAX_GRADIO_CANDIDATES:
                            break
                        out_audios.append(
                            gr.Audio(
                                label=f"Generated Audio {i + 1}",
                                type="filepath",
                                interactive=False,
                                visible=(i == 0),
                                min_width=160,
                            )
                        )
        out_log = gr.Textbox(label="Run Log", lines=8)
        out_timing = gr.Textbox(label="Timing", lines=8)

        save_preset_btn.click(
            _save_voice_preset,
            inputs=[
                preset_name, uploaded_audio, seed_raw, max_chunk_chars,
                chunk_pause, paragraph_pause, duration_scale,
            ],
            outputs=[preset_status, preset_name],
        )
        refresh_preset_btn.click(
            _refresh_voice_presets,
            inputs=[preset_name],
            outputs=[preset_name, preset_status],
        )
        load_preset_btn.click(
            _load_voice_preset,
            inputs=[preset_name],
            outputs=[
                uploaded_audio, seed_raw, max_chunk_chars,
                chunk_pause, paragraph_pause, duration_scale, preset_status,
            ],
        )

        preview_btn.click(_preview_long_text, inputs=[text, max_chunk_chars], outputs=[split_preview])
        long_mode.change(_on_long_mode_change, inputs=[long_mode], outputs=[num_candidates])

        generation_inputs = [
                long_mode,
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                text,
                uploaded_audio,
                uploaded_speaker_embedding,
                speaker_embedding_path_raw,
                num_steps,
                num_candidates,
                seed_raw,
                seconds_raw,
                duration_scale,
                t_schedule_mode,
                sway_coeff,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
                speaker_kv_scale_raw,
                speaker_kv_min_t_raw,
                speaker_kv_max_layers_raw,
                lora_adapter_raw,
                max_chunk_chars,
                chunk_pause,
                paragraph_pause,
                resume_enabled,
                delete_chunks_after_success,
            ]
        generate_btn.click(
            _dispatch_with_draft,
            inputs=[draft_mode, draft_steps, *generation_inputs],
            outputs=[*out_audios, out_log, out_timing],
        )
        preview_generate_btn.click(
            _dispatch_preview_with_draft,
            inputs=[draft_mode, draft_steps, *generation_inputs],
            outputs=[*out_audios, out_log, out_timing],
        )
        model_device.change(
            _on_model_device_change, inputs=[model_device], outputs=[model_precision]
        )
        codec_device.change(
            _on_codec_device_change, inputs=[codec_device], outputs=[codec_precision]
        )
        t_schedule_mode.change(
            _on_t_schedule_mode_change, inputs=[t_schedule_mode], outputs=[sway_coeff]
        )

        load_model_btn.click(
            _load_model,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
            ],
            outputs=[clear_cache_msg],
        )
        clear_cache_btn.click(_clear_runtime_cache, outputs=[clear_cache_msg])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app for Irodori-TTS with cached runtime.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7867)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    global SHARE_MODE
    SHARE_MODE = bool(args.share)
    if SHARE_MODE:
        print("\n[SECURITY WARNING] --share is enabled: the Gradio app may be reachable from the public internet.", flush=True)
        print("[SECURITY WARNING] Checkpoint path editing and direct Speaker Embedding Path input are disabled; only the default checkpoint is accepted.", flush=True)

    demo = build_ui(share_mode=SHARE_MODE)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
        css=EMOJI_PALETTE_CSS,
    )


if __name__ == "__main__":
    main()
