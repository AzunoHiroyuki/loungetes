#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import hashlib
import json
import shutil

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



def _split_long_text(text: str, max_chars: int) -> list[tuple[str, bool]]:
    raw = "" if text is None else str(text).strip()
    if not raw:
        return []
    max_chars = max(20, int(max_chars))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    chunks = []
    for pi, paragraph in enumerate(paragraphs):
        sentences = [x.strip() for x in re.findall(r".+?(?:[。！？!?]+[」』】）)]*|$)", paragraph, flags=re.S) if x.strip()] or [paragraph]
        pc, current = [], ""
        for sentence in sentences:
            pieces=[sentence[i:i+max_chars] for i in range(0,len(sentence),max_chars)]
            for piece in pieces:
                if current and len(current)+len(piece)>max_chars:
                    pc.append(current.strip()); current=piece
                else:
                    current += piece
        if current.strip(): pc.append(current.strip())
        for ci, chunk in enumerate(pc):
            chunks.append((chunk, ci==len(pc)-1 and pi<len(paragraphs)-1))
    return chunks

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
    resume_enabled, progress=gr.Progress()):
    chunks=_split_long_text(text,int(max_chunk_chars))
    if not chunks: raise ValueError("text is required.")
    progress(0, desc=f"準備中… 全{len(chunks)}チャンク")
    if int(num_candidates)!=1: raise ValueError("Long Text Mode requires Num Candidates = 1.")

    parsed_num_steps=_parse_optional_int(num_steps,"num_steps")
    cfg_scale=_parse_optional_float(cfg_scale_raw,"cfg_scale")
    truncation_factor=_parse_optional_float(truncation_factor_raw,"truncation_factor")
    rescale_k=_parse_optional_float(rescale_k_raw,"rescale_k")
    rescale_sigma=_parse_optional_float(rescale_sigma_raw,"rescale_sigma")
    speaker_kv_scale=_parse_optional_float(speaker_kv_scale_raw,"speaker_kv_scale")
    speaker_kv_min_t=_parse_optional_float(speaker_kv_min_t_raw,"speaker_kv_min_t")
    speaker_kv_max_layers=_parse_optional_int(speaker_kv_max_layers_raw,"speaker_kv_max_layers")
    seed=_parse_optional_int(seed_raw,"seed")
    manual_seconds=_parse_optional_float(seconds_raw,"seconds")
    lora_adapter=_parse_optional_str(lora_adapter_raw)

    if bool(resume_enabled) and seed is None:
        raise ValueError("途中再開を使う場合は Seed を固定してください（例: 12345）。")

    ref_wavs=_resolve_ref_wavs(uploaded_audio)
    speaker_embedding=_resolve_speaker_embedding(uploaded_speaker_embedding,speaker_embedding_path_raw)
    if ref_wavs and speaker_embedding is not None:
        raise ValueError("Reference audio and speaker embedding are mutually exclusive.")
    no_ref=not ref_wavs and speaker_embedding is None

    # Same text + key settings = same resumable job.
    job_data = {
        "text": str(text),
        "checkpoint": str(checkpoint),
        "max_chunk_chars": int(max_chunk_chars),
        "seed": seed,
        "duration_scale": float(duration_scale),
        "num_steps": parsed_num_steps,
        "cfg_guidance_mode": str(cfg_guidance_mode),
        "cfg_scale_text": float(cfg_scale_text),
        "cfg_scale_speaker": float(cfg_scale_speaker),
        "chunk_count": len(chunks),
    }
    job_id = hashlib.sha256(
        json.dumps(job_data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    work_dir = Path("gradio_outputs_large_long") / "resume" / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "job.json").write_text(
        json.dumps(job_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    runtime_key=_build_runtime_key(checkpoint,model_device,model_precision,codec_device,codec_precision)
    runtime,reloaded=get_cached_runtime(runtime_key)

    parts=[]; sample_rate=None; messages=[]; timings=[]; total=0.0
    resumed_count=0

    for idx,(chunk,pbreak) in enumerate(chunks,1):
        progress(
            (idx - 1) / len(chunks),
            desc=f"{idx}/{len(chunks)} チャンク処理中…",
        )
        chunk_file = work_dir / f"chunk_{idx:04d}.pt"

        if bool(resume_enabled) and chunk_file.exists():
            saved = torch.load(chunk_file, map_location="cpu", weights_only=False)
            audio = saved["audio"].float()
            sr = int(saved["sample_rate"])
            if sample_rate is not None and sr != sample_rate:
                raise ValueError(f"Saved chunk sample rate mismatch at chunk {idx}.")
            sample_rate = sr
            resumed_count += 1
            print(f"[large-long] resume chunk {idx}/{len(chunks)}", flush=True)
            messages.append(f"chunk {idx}: resumed from saved checkpoint")
        else:
            print(f"[large-long] generate chunk {idx}/{len(chunks)} ({len(chunk)} chars)",flush=True)
            result=runtime.synthesize(SamplingRequest(
                text=chunk, ref_wav=None, ref_wavs=ref_wavs or None, ref_latent=None,
                ref_embed=speaker_embedding, no_ref=bool(no_ref), ref_normalize_db=-16.0,
                ref_ensure_max=True, num_candidates=1, decode_mode="sequential",
                seconds=manual_seconds, duration_scale=float(duration_scale),
                max_ref_seconds=None, max_text_len=None, num_steps=parsed_num_steps,
                seed=None if seed is None else int(seed),
                cfg_guidance_mode=str(cfg_guidance_mode), cfg_scale_text=float(cfg_scale_text),
                cfg_scale_speaker=float(cfg_scale_speaker), cfg_scale=cfg_scale,
                cfg_min_t=float(cfg_min_t), cfg_max_t=float(cfg_max_t),
                truncation_factor=truncation_factor, rescale_k=rescale_k,
                rescale_sigma=rescale_sigma, context_kv_cache=bool(context_kv_cache),
                speaker_kv_scale=speaker_kv_scale, speaker_kv_min_t=speaker_kv_min_t,
                speaker_kv_max_layers=speaker_kv_max_layers, t_schedule_mode=str(t_schedule_mode),
                sway_coeff=float(sway_coeff), trim_tail=True, lora_adapter=lora_adapter),
                log_fn=lambda msg: print(msg,flush=True))
            audio=result.audios[0].float().detach().cpu()
            sample_rate=int(result.sample_rate)
            messages += [f"chunk {idx}: {m}" for m in result.messages]
            timings += [(f"chunk{idx}:{n}",sec) for n,sec in result.stage_timings]
            total += float(result.total_to_decode)

            # Save immediately after each successful chunk.
            torch.save({
                "audio": audio,
                "sample_rate": sample_rate,
                "text": chunk,
                "paragraph_break": bool(pbreak),
            }, chunk_file)
            print(f"[large-long] saved checkpoint: {chunk_file}", flush=True)

        parts.append(audio)
        if idx<len(chunks):
            pause=float(paragraph_pause if pbreak else chunk_pause)
            if pause>0:
                parts.append(torch.zeros(
                    (*audio.shape[:-1],max(1,round(sample_rate*pause))),
                    dtype=audio.dtype
                ))
        progress(
            idx / len(chunks),
            desc=f"{idx}/{len(chunks)} 完了",
        )

    progress(1.0, desc="音声を結合して保存中…")
    combined=torch.cat(parts,dim=-1)
    out_dir=Path("gradio_outputs_large_long")
    out_dir.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path=save_wav(out_dir/f"large_long_{stamp}.wav",combined,sample_rate)
    progress(1.0, desc="完了しました")

    detail="\n".join([
        "Irodori-TTS v4 Large LONG / Resume",
        "runtime: reloaded" if reloaded else "runtime: reused",
        f"job_id: {job_id}",
        f"chunks: {len(chunks)}",
        f"resumed_chunks: {resumed_count}",
        f"checkpoint_folder: {work_dir}",
        f"saved: {out_path}",
        *messages
    ])
    updates=[gr.update(value=str(out_path),visible=True)]
    updates += [gr.update(value=None,visible=False) for _ in range(MAX_GRADIO_CANDIDATES-1)]
    return (*updates,detail,_format_timings(timings,total))

def _dispatch_generation(long_mode, checkpoint, model_device, model_precision, codec_device,
    codec_precision, text, uploaded_audio, uploaded_speaker_embedding, speaker_embedding_path_raw,
    num_steps, num_candidates, seed_raw, seconds_raw, duration_scale, t_schedule_mode, sway_coeff,
    cfg_guidance_mode, cfg_scale_text, cfg_scale_speaker, cfg_scale_raw, cfg_min_t, cfg_max_t,
    context_kv_cache, truncation_factor_raw, rescale_k_raw, rescale_sigma_raw,
    speaker_kv_scale_raw, speaker_kv_min_t_raw, speaker_kv_max_layers_raw, lora_adapter_raw,
    max_chunk_chars, chunk_pause, paragraph_pause, resume_enabled):
    common=(checkpoint,model_device,model_precision,codec_device,codec_precision,text,uploaded_audio,
        uploaded_speaker_embedding,speaker_embedding_path_raw,num_steps,num_candidates,seed_raw,
        seconds_raw,duration_scale,t_schedule_mode,sway_coeff,cfg_guidance_mode,cfg_scale_text,
        cfg_scale_speaker,cfg_scale_raw,cfg_min_t,cfg_max_t,context_kv_cache,truncation_factor_raw,
        rescale_k_raw,rescale_sigma_raw,speaker_kv_scale_raw,speaker_kv_min_t_raw,
        speaker_kv_max_layers_raw,lora_adapter_raw)
    return _run_long_generation(*common,max_chunk_chars,chunk_pause,paragraph_pause,resume_enabled) if bool(long_mode) else _run_generation(*common)



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
    preset_dir = root / safe
    audio_dir = preset_dir / "reference_audio"
    preset_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Replace old copied references for this preset.
    for old in audio_dir.iterdir():
        if old.is_file():
            old.unlink()

    refs = _resolve_ref_wavs(uploaded_audio)
    copied = []
    for i, raw in enumerate(refs, 1):
        src = Path(raw)
        if not src.exists():
            continue
        target = audio_dir / f"{i:02d}_{src.name}"
        shutil.copy2(src, target)
        copied.append(str(target.resolve()))

    data = {
        "name": safe,
        "seed": int(seed),
        "max_chunk_chars": int(max_chunk_chars),
        "chunk_pause": float(chunk_pause),
        "paragraph_pause": float(paragraph_pause),
        "duration_scale": float(duration_scale),
        "reference_audio": copied,
    }
    (preset_dir / "preset.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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


def build_ui() -> gr.Blocks:
    default_checkpoint = _default_checkpoint()
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    with gr.Blocks(title="Irodori-TTS Gradio") as demo:
        gr.Markdown("# Irodori-TTS v4 Large LONG - Voice Stable")
        gr.Markdown(
            "Reference-audio cloning / Speaker Inversion UI. "
            "Irodori-TTS-v4.1-Small is used by default; unchanged settings reuse the cached runtime."
        )

        with gr.Row():
            checkpoint = gr.Textbox(
                label="Checkpoint (.pt/.safetensors or HF repo id)",
                value=default_checkpoint,
                scale=4,
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
                info="各チャンク生成後に自動保存。同じ文章・設定・Seedで再実行すると続きから再開します。",
            )
            gr.Markdown(
                "長文モードでは **Num Candidates = 1**。途中再開を使う場合は **Seedを固定**してください。  "
                "Generate中は画面に **現在のチャンク / 全チャンク** と進捗バーを表示します。"
            )

        with gr.Accordion("Sampling", open=True):
            with gr.Row():
                num_steps = gr.Textbox(
                    label="Num Steps",
                    value="",
                    placeholder="Auto (RF: 40, MeanFlow: 4)",
                    info="Blank: model default (RF: 40, MeanFlow: 4).",
                )
                num_candidates = gr.Slider(
                    label="Num Candidates",
                    minimum=1,
                    maximum=MAX_GRADIO_CANDIDATES,
                    value=1,
                    step=1,
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

        generate_btn = gr.Button("Generate", variant="primary")

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

        generate_btn.click(
            _dispatch_generation,
            inputs=[
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
            ],
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

    demo = build_ui()
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
