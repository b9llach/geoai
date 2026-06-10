"""`geoai-train-stage1` — train the hierarchical S2-cell classifier.

Single-script multi-GPU via 🤗 Accelerate:

    .venv/bin/accelerate launch -m geoai.stage1.train [args]

(For single-GPU, plain `python -m geoai.stage1.train` works too.)

Default config follows PLAN.md §"Phase 4 — Training Configuration", adjusted
to fit on dual 24 GB 4090s under DDP + Accelerate's bf16 wrapping:
    - SigLIP2-SO400M-patch14-384 backbone, 4-view concat pool
    - bf16 mixed precision + gradient checkpointing
    - AdamW lr=2e-5 wd=0.01, cosine schedule, 5% warmup
    - 15 epochs (early-stop on val loss)
    - per-GPU batch 2, grad accum 32 → effective 128 across 2 GPUs
    - haversine sigmas {3:2000, 6:500, 9:100, 12:20} km
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Reduce VRAM fragmentation under DDP — the SigLIP encoder allocates many
# differently-sized tensors and the default allocator strands ~1-2 GB
# per GPU as "reserved but unallocated."
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import typer
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

import wandb
from geoai.config import PROCESSED_DIR
from geoai.config import METADATA_DB
from geoai.stage1.cells import PRUNED_LABEL, CellVocab
from geoai.stage1.country_vocab import CountryVocab
from geoai.stage1.dataset import PanoDataset, collate
from geoai.stage1.eval import aggregate_metrics, evaluate_batch
from geoai.stage1.loss import DEFAULT_SIGMAS_KM, country_loss, hierarchical_loss
from geoai.stage1.model import DEFAULT_BACKBONE, HierarchicalGeocellClassifier

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


@dataclass
class TrainConfig:
    backbone: str = DEFAULT_BACKBONE
    levels: tuple = (3, 6, 9, 12)
    min_count_l3: int = 1
    min_count_l6: int = 1
    min_count_l9: int = 2
    min_count_l12: int = 5
    epochs: int = 15
    per_gpu_batch: int = 2
    grad_accum_steps: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_workers: int = 4
    log_every: int = 25
    val_every_steps: int = 0    # 0 = end of epoch only
    out_dir: Path = PROCESSED_DIR / "checkpoints" / "stage1"
    wandb_project: str = "geoai-stage1"
    wandb_run_name: Optional[str] = None
    resume_from: Optional[Path] = None
    # Warm-start: load backbone + cell heads (+ country head remapped by code)
    # from a prior checkpoint, but with FRESH optimizer/scheduler/epoch counter.
    # Use this (not resume_from) when the country vocab size changed — resume
    # would crash on the country_head shape mismatch. Mutually exclusive with
    # resume_from.
    init_from: Optional[Path] = None
    # Old cells parquet that the init_from checkpoint was trained on. When set,
    # cell heads + embeds are remapped BY cell_id_str. When None, the cell heads
    # load verbatim — only safe when the cell vocab is unchanged. Required when
    # cells_parquet differs from what init_from was trained on.
    init_from_cells_parquet: Optional[Path] = None
    seed: int = 42
    compile: bool = False
    country_loss_weight: float = 0.3
    border_lambda: float = 0.0   # 0 = disabled (V1). Try 0.5 for V2 ablation.
    hardneg_weights_path: Optional[Path] = None  # JSON of per-country sampling weights
    cells_parquet: Optional[Path] = None         # override PROCESSED_DIR/cells.parquet
    # Haversine-smoothing sigma (km) for the FINE levels. Defaults = the
    # original {9:100, 12:20}, which are 5.5×/8.7× the cell side — so broad
    # that "correct" means a ~100 km blob, not the cell (oracle showed cells
    # are fine; the model just never learns to nail them). Tighten (e.g.
    # 9:30, 12:5) to force fine-cell discrimination — the V4 precision ablation.
    sigma_l9: float = 100.0
    sigma_l12: float = 20.0

    def min_count_dict(self) -> dict[int, int]:
        return {3: self.min_count_l3, 6: self.min_count_l6,
                9: self.min_count_l9, 12: self.min_count_l12}

    def sigmas_dict(self) -> dict[int, float]:
        """Per-level smoothing sigmas. L3/L6 stay at the originals; L9/L12 are
        tunable for the precision ablation."""
        return {3: 2000.0, 6: 500.0, 9: self.sigma_l9, 12: self.sigma_l12}


# Optional training-progress notifications. Set GEOAI_DISCORD_WEBHOOK to a
# Discord webhook URL to enable; empty (default) disables.
_DISCORD_WEBHOOK = os.environ.get("GEOAI_DISCORD_WEBHOOK", "")


def _post_discord_embed(title: str, fields: dict, color: int = 0x5865F2) -> None:
    """Fire-and-forget Discord webhook. Network errors are logged, not raised —
    a training run must never die because of a notification glitch.
    Set GEOAI_DISCORD_WEBHOOK='' to disable.
    """
    if not _DISCORD_WEBHOOK:
        return
    try:
        import urllib.request
        from datetime import datetime, timezone
        payload = {
            "embeds": [
                {
                    "title": title,
                    "color": color,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fields": [
                        {"name": k, "value": str(v), "inline": True}
                        for k, v in fields.items()
                    ],
                }
            ]
        }
        req = urllib.request.Request(
            _DISCORD_WEBHOOK,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                # Discord rejects requests without a User-Agent (403).
                "User-Agent": "geoai-train/0.1 (+https://github.com/b9llach/geoai)",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        log.warning(f"discord webhook failed: {e}")


def _build_hardneg_sampler(
    train_ds: PanoDataset,
    weights_path: Path,
    seed: int = 42,
) -> WeightedRandomSampler:
    """WeightedRandomSampler over training panos, with per-country multipliers
    loaded from JSON. Unlisted countries default to 1.0. Sampler is sized to
    `len(train_ds)` so an "epoch" remains the same number of steps as before.
    """
    cfg = json.loads(Path(weights_path).read_text())
    weights_map: dict[str, float] = cfg.get("weights", cfg)
    # Reject the leading "_comment" key if author put weights at top level.
    weights_map = {k: float(v) for k, v in weights_map.items() if not k.startswith("_")}

    # Per-pano weight = country multiplier (default 1.0)
    per_pano_w = torch.tensor(
        [weights_map.get(r["country_code"] or "", 1.0) for r in train_ds.rows],
        dtype=torch.double,
    )
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=per_pano_w, num_samples=len(train_ds),
        replacement=True, generator=g,
    )


def _warm_start_from(
    model,
    init_from: Path,
    new_countries: CountryVocab | None,
    new_cells: CellVocab | None,
    init_cells_parquet: Path | None,
    min_count: dict[int, int],
    log,
) -> None:
    """Load weights from a prior checkpoint into a freshly-built model.

    Loads backbone + projection verbatim. Cell heads + cell embeds can be
    remapped BY CELL ID when `init_cells_parquet` is given (the old training's
    cells.parquet) — needed when the cell vocab grew because new panos opened
    new cells. CellVocab.from_parquet sorts by cell_id_str so adding cells
    shifts indices; positional copy would scramble every learned cell. Each
    shared cell's row is copied to its new index; newcomers stay fresh-init.
    If `init_cells_parquet` is None the cell heads load verbatim — only safe
    when the cell vocab is unchanged.

    The country head is always remapped BY CODE (sorted alphabetically, same
    shift-on-insert problem). Shared codes copy across; newcomers stay fresh.

    Optimizer / scheduler / RNG are NOT touched (caller keeps them fresh) —
    this is a warm-start, not a resume.

    Must run on the RAW (uncompiled, unwrapped) model: checkpoint keys carry a
    torch.compile `_orig_mod.` prefix which we strip to match.
    """
    from safetensors.torch import load_file
    raw = load_file(str(Path(init_from) / "model.safetensors"))
    sd = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}

    # ---- Cell heads + cell embeds: remap by cell_id_str -------------------
    if init_cells_parquet is not None and new_cells is not None:
        old_cells = CellVocab.from_parquet(
            path=init_cells_parquet,
            levels=new_cells.levels,
            min_count=min_count,
        )
        for i, lvl in enumerate(new_cells.levels):
            old_map = old_cells.by_level[lvl].cell_to_index
            new_size = new_cells.vocab_size(lvl)
            old_size = old_cells.vocab_size(lvl)

            # Heads (Linear): weight [V_new, in_dim], bias [V_new]
            hw_key, hb_key = f"heads.{i}.weight", f"heads.{i}.bias"
            old_hw = sd.pop(hw_key, None)
            old_hb = sd.pop(hb_key, None)
            if old_hw is not None and old_hb is not None:
                new_hw = model.heads[i].weight.data.clone()
                new_hb = model.heads[i].bias.data.clone()
                if new_hw.shape[1] != old_hw.shape[1]:
                    log.warning(
                        f"[warm-start] L{lvl} head in_dim mismatch "
                        f"(old={old_hw.shape[1]} new={new_hw.shape[1]}); "
                        f"keeping fresh init for this head"
                    )
                else:
                    copied = 0
                    for new_idx in range(new_size):
                        cid = new_cells.cell_id_at(lvl, new_idx)
                        old_idx = old_map.get(cid, PRUNED_LABEL)
                        if 0 <= old_idx < old_hw.shape[0]:
                            new_hw[new_idx] = old_hw[old_idx]
                            new_hb[new_idx] = old_hb[old_idx]
                            copied += 1
                    log.info(
                        f"[warm-start] L{lvl} cell head: copied {copied}/{new_size} "
                        f"rows by cell_id (old vocab={old_size})"
                    )
                sd[hw_key] = new_hw
                sd[hb_key] = new_hb

            # Cell embeds (Embedding): only for levels that condition a successor.
            if i < len(new_cells.levels) - 1:
                emb_key = f"cell_embeds.{i}.weight"
                old_emb = sd.pop(emb_key, None)
                if old_emb is not None:
                    new_emb = model.cell_embeds[i].weight.data.clone()
                    if new_emb.shape[1] != old_emb.shape[1]:
                        log.warning(
                            f"[warm-start] L{lvl} cell_embed dim mismatch "
                            f"(old={old_emb.shape[1]} new={new_emb.shape[1]}); "
                            f"keeping fresh init"
                        )
                    else:
                        copied = 0
                        for new_idx in range(new_size):
                            cid = new_cells.cell_id_at(lvl, new_idx)
                            old_idx = old_map.get(cid, PRUNED_LABEL)
                            if 0 <= old_idx < old_emb.shape[0]:
                                new_emb[new_idx] = old_emb[old_idx]
                                copied += 1
                        log.info(
                            f"[warm-start] L{lvl} cell_embed: copied "
                            f"{copied}/{new_size} rows by cell_id"
                        )
                    sd[emb_key] = new_emb

    # ---- Country head: remap by code (existing logic) ---------------------
    ch_w = sd.pop("country_head.weight", None)
    ch_b = sd.pop("country_head.bias", None)
    if new_countries is not None and model.country_head is not None and ch_w is not None:
        old_vocab = CountryVocab.from_json(Path(init_from) / "country_vocab.json")
        new_w = model.country_head.weight.data.clone()   # fresh-init [N_new, D]
        new_b = model.country_head.bias.data.clone()
        copied = 0
        for new_idx, code in enumerate(new_countries.codes):
            old_idx = old_vocab.index(code)
            if old_idx >= 0:
                new_w[new_idx] = ch_w[old_idx]
                new_b[new_idx] = ch_b[old_idx]
                copied += 1
        sd["country_head.weight"] = new_w
        sd["country_head.bias"] = new_b
        log.info(f"[warm-start] country head: copied {copied}/"
                 f"{len(new_countries.codes)} rows by code, "
                 f"{len(new_countries.codes) - copied} fresh-init "
                 f"(old vocab n={old_vocab.size})")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    log.info(f"[warm-start] loaded {init_from}: "
             f"{len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        log.warning(f"[warm-start] missing keys (first 5): {list(missing)[:5]}")
    if unexpected:
        log.warning(f"[warm-start] unexpected keys (first 5): {list(unexpected)[:5]}")


def _build_cell_to_country(
    cells: CellVocab,
    countries: CountryVocab,
    db_path: Path = METADATA_DB,
    split: str = "train",
) -> dict[int, torch.Tensor]:
    """For each level, build a [vocab_size] long tensor mapping cell_idx →
    modal country_idx among that cell's training panos. Unknown / unmapped
    cells get PRUNED_LABEL (-1) so the border penalty masks them out.

    One-time scan of the catalog at startup. Result lives on CPU; copied to
    GPU per-step inside `hierarchical_loss`.
    """
    import sqlite3
    from geoai.stage1.cells import PRUNED_LABEL

    out: dict[int, torch.Tensor] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        for lvl in cells.levels:
            rows = conn.execute(
                f"""
                SELECT s2_l{lvl}, country_code, COUNT(*) AS n
                FROM panos
                WHERE split = ?
                  AND s2_l{lvl} IS NOT NULL
                  AND country_code IS NOT NULL
                GROUP BY s2_l{lvl}, country_code
                ORDER BY s2_l{lvl}, n DESC
                """,
                (split,),
            ).fetchall()
            modal: dict[str, str] = {}
            for cell_id_str, cc, _ in rows:
                modal.setdefault(cell_id_str, cc)  # first row per cell = highest-count

            mapping = torch.full(
                (cells.vocab_size(lvl),), PRUNED_LABEL, dtype=torch.long
            )
            for cell_id_str, cc in modal.items():
                cell_idx = cells.index(lvl, cell_id_str)
                if cell_idx == PRUNED_LABEL:
                    continue  # cell pruned out of vocab
                country_idx = countries.index(cc)
                if country_idx != PRUNED_LABEL:
                    mapping[cell_idx] = country_idx
            out[lvl] = mapping
    finally:
        conn.close()
    return out


def _move_batch(batch: dict, device) -> dict:
    return {
        "pano_ids": batch["pano_ids"],
        "country_codes": batch["country_codes"],
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "latlng": batch["latlng"].to(device, non_blocking=True),
        "cell_indices": {lvl: t.to(device, non_blocking=True)
                         for lvl, t in batch["cell_indices"].items()},
        "country_idx": batch["country_idx"].to(device, non_blocking=True),
    }


def _save_ckpt(
    accelerator: Accelerator, path: Path, model, optimizer, scheduler,
    step: int, epoch: int,
    countries: CountryVocab | None = None,
):
    if accelerator.is_main_process:
        path.parent.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(path))
    if accelerator.is_main_process:
        meta = {"step": step, "epoch": epoch}
        (path / "meta.json").write_text(json.dumps(meta))
        if countries is not None:
            countries.to_json(path / "country_vocab.json")


@app.command()
def main(
    backbone: str = typer.Option(DEFAULT_BACKBONE),
    epochs: int = typer.Option(15),
    per_gpu_batch: int = typer.Option(2),
    grad_accum_steps: int = typer.Option(32),
    lr: float = typer.Option(2e-5),
    weight_decay: float = typer.Option(0.01),
    warmup_ratio: float = typer.Option(0.05),
    num_workers: int = typer.Option(4),
    out_dir: Path = typer.Option(PROCESSED_DIR / "checkpoints" / "stage1"),
    wandb_project: str = typer.Option("geoai-stage1"),
    wandb_run_name: Optional[str] = typer.Option(None),
    resume_from: Optional[Path] = typer.Option(None),
    init_from: Optional[Path] = typer.Option(
        None,
        help="Warm-start from a checkpoint: load backbone + cell heads, remap the "
             "country head by country code, fresh optimizer/scheduler/epoch. Use "
             "this (not --resume-from) when the country vocab size changed.",
    ),
    init_from_cells_parquet: Optional[Path] = typer.Option(
        None,
        help="Old cells parquet that --init-from was trained on. Required when the "
             "cell vocab grew (new panos opened new cells). When set, cell heads + "
             "embeds are remapped BY cell_id_str; shared cells copy across, newcomers "
             "stay fresh. When None, cell heads load verbatim — only safe if the cell "
             "vocab is unchanged.",
    ),
    train_subset: int = typer.Option(0, help="Limit training set size (0 = full). For dry-runs."),
    val_subset: int = typer.Option(0, help="Limit val set size (0 = full)."),
    log_every: int = typer.Option(25),
    seed: int = typer.Option(42),
    compile: bool = typer.Option(
        False, "--compile/--no-compile",
        help="Wrap model with torch.compile for ~1.5× speedup. First optimizer step is slow (~5–10 min compile).",
    ),
    country_loss_weight: float = typer.Option(
        0.3, help="Weight on auxiliary country-head CE loss. 0.0 disables the head entirely (V1 behavior)."
    ),
    border_lambda: float = typer.Option(
        0.0, help="Cross-country prediction penalty multiplier on per-level CE. 0=off (V1). Try 0.5 for V2."
    ),
    hardneg_weights_path: Optional[Path] = typer.Option(
        None,
        help="Optional JSON of per-country sampling weights. Builds a WeightedRandomSampler "
             "over training panos in place of plain shuffle. See scripts/analysis/hardneg_weights.json.",
    ),
    cells_parquet: Optional[Path] = typer.Option(
        None,
        help="Path to cells.parquet for the cell vocab. Defaults to PROCESSED_DIR/cells.parquet. "
             "Use a V2 path (e.g. cells_v2.parquet) to keep V1 vocab intact for serve/inference.",
    ),
    sigma_l9: float = typer.Option(
        100.0, help="Haversine-smoothing sigma (km) at L9. Default 100 is ~5.5× the "
                    "18km cell (over-smooth → vague). Try 30 to force fine-cell precision."
    ),
    sigma_l12: float = typer.Option(
        20.0, help="Haversine-smoothing sigma (km) at L12. Default 20 is ~8.7× the "
                   "2.3km cell. Try 5 for the precision ablation."
    ),
) -> None:
    cfg = TrainConfig(
        backbone=backbone, epochs=epochs, per_gpu_batch=per_gpu_batch,
        grad_accum_steps=grad_accum_steps, lr=lr, weight_decay=weight_decay,
        warmup_ratio=warmup_ratio, num_workers=num_workers, out_dir=out_dir,
        wandb_project=wandb_project, wandb_run_name=wandb_run_name,
        resume_from=resume_from, init_from=init_from,
        init_from_cells_parquet=init_from_cells_parquet,
        log_every=log_every, seed=seed,
        compile=compile, country_loss_weight=country_loss_weight,
        border_lambda=border_lambda, hardneg_weights_path=hardneg_weights_path,
        cells_parquet=cells_parquet, sigma_l9=sigma_l9, sigma_l12=sigma_l12,
    )
    if resume_from is not None and init_from is not None:
        raise typer.BadParameter("--resume-from and --init-from are mutually exclusive")

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg.grad_accum_steps,
        log_with="wandb",
    )
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    torch.manual_seed(cfg.seed)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.wandb_project,
            init_kwargs={"wandb": {"name": cfg.wandb_run_name, "config": asdict(cfg)}},
        )

    # ---- guard: refuse to silently clobber existing checkpoints ----------
    # Without this, launching a fresh V2 run into V1's out_dir would overwrite
    # epoch_00, epoch_01, ... as V2 epochs complete. --resume-from is the
    # only legitimate way to write into a populated dir.
    if accelerator.is_main_process and cfg.resume_from is None and cfg.out_dir.exists():
        existing = sorted(p.name for p in cfg.out_dir.glob("epoch_*") if p.is_dir())
        if existing:
            raise FileExistsError(
                f"out_dir already contains {len(existing)} checkpoint(s): "
                f"{existing[:3]}{'...' if len(existing) > 3 else ''} at {cfg.out_dir}. "
                f"Either pass a fresh --out-dir, --resume-from <ckpt>, or move the "
                f"existing checkpoints aside. Refusing to overwrite."
            )

    # ---- vocab + datasets ---------------------------------------------------
    cells_path = cfg.cells_parquet or (PROCESSED_DIR / "cells.parquet")
    cells = CellVocab.from_parquet(
        path=cells_path, min_count=cfg.min_count_dict(), levels=cfg.levels,
    )
    log.info(f"Cell vocab: {cells} (from {cells_path})")

    countries: CountryVocab | None = None
    cell_to_country: dict[int, torch.Tensor] | None = None
    if cfg.country_loss_weight > 0 or cfg.border_lambda > 0:
        countries = CountryVocab.from_catalog(METADATA_DB, split="train")
        log.info(f"Country vocab: {countries}")
        if cfg.border_lambda > 0:
            log.info(f"building cell→country lookup (lambda={cfg.border_lambda}) ...")
            cell_to_country = _build_cell_to_country(cells, countries)
            for lvl, t in cell_to_country.items():
                covered = (t != PRUNED_LABEL).sum().item()
                log.info(f"  L{lvl}: {covered}/{len(t)} cells country-mapped "
                         f"({100*covered/max(len(t),1):.1f}%)")

    train_ds = PanoDataset(cells, split="train", countries=countries)
    val_ds = PanoDataset(cells, split="val", augment=False, countries=countries)
    if train_subset > 0:
        train_ds.rows = train_ds.rows[:train_subset]
    if val_subset > 0:
        val_ds.rows = val_ds.rows[:val_subset]
    log.info(f"train={len(train_ds):,}  val={len(val_ds):,}")

    sampler = None
    if cfg.hardneg_weights_path is not None:
        sampler = _build_hardneg_sampler(train_ds, cfg.hardneg_weights_path, seed=cfg.seed)
        log.info(f"hardneg sampler enabled from {cfg.hardneg_weights_path}")
    train_loader = DataLoader(
        train_ds, batch_size=cfg.per_gpu_batch,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.per_gpu_batch * 2, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True,
    )

    # ---- model + optim ------------------------------------------------------
    model = HierarchicalGeocellClassifier(
        cells=cells, backbone_name=cfg.backbone,
        num_countries=countries.size if countries is not None else 0,
    )
    # Warm-start (weights only) must happen on the raw model, before compile/DDP
    # wrap — checkpoint keys carry the compile prefix we strip in the helper.
    if cfg.init_from is not None:
        _warm_start_from(
            model, cfg.init_from, countries, cells,
            cfg.init_from_cells_parquet, cfg.min_count_dict(), log,
        )
    if cfg.compile:
        # Compile BEFORE accelerate wraps the model in DDP — dynamo handles
        # DDP-wrapped modules less well. mode='default' is the safest pairing
        # with gradient checkpointing; 'reduce-overhead' uses CUDA graphs that
        # conflict with checkpointing's re-forward.
        log.info("compiling model with torch.compile — first optimizer step will be slow (~5-10 min)")
        model = torch.compile(model, mode="default", dynamic=False)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Centroids per level — kept on the same device as logits.
    centroids = {
        lvl: cells.centroids_tensor(lvl, device=accelerator.device)
        for lvl in cells.levels
    }

    if cfg.resume_from is not None:
        accelerator.load_state(str(cfg.resume_from))
        meta = json.loads((cfg.resume_from / "meta.json").read_text())
        start_step, start_epoch = meta["step"], meta["epoch"]
        log.info(f"resumed at epoch {start_epoch} step {start_step}")
    else:
        start_step, start_epoch = 0, 0

    # ---- loop ---------------------------------------------------------------
    global_step = start_step
    import time as _time
    for epoch in range(start_epoch, cfg.epochs):
        epoch_t0 = _time.time()
        if accelerator.is_main_process:
            _post_discord_embed(
                title=f"{cfg.wandb_run_name or 'geoai-stage1'} — epoch {epoch} starting",
                fields={
                    "epoch": f"{epoch}/{cfg.epochs - 1}",
                    "step": f"{global_step:,}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "out_dir": str(cfg.out_dir),
                },
                color=0xFAA61A,  # amber — "in progress"
            )

        model.train()
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}",
            disable=not accelerator.is_main_process,
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in pbar:
            batch = _move_batch(batch, accelerator.device)
            with accelerator.accumulate(model):
                use_country = countries is not None
                out = model(
                    batch["pixel_values"],
                    teacher_forcing=batch["cell_indices"],
                    return_country=use_country,
                )
                if use_country:
                    logits, country_logits = out
                else:
                    logits, country_logits = out, None

                loss_h, per_lvl = hierarchical_loss(
                    logits, batch["latlng"], batch["cell_indices"], centroids,
                    sigmas_km=cfg.sigmas_dict(),
                    country_idx=batch["country_idx"] if use_country else None,
                    cell_to_country=cell_to_country,
                    border_lambda=cfg.border_lambda,
                )
                if country_logits is not None:
                    loss_c = country_loss(country_logits, batch["country_idx"])
                    loss = loss_h + cfg.country_loss_weight * loss_c
                else:
                    loss_c = None
                    loss = loss_h

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    pbar.set_postfix(
                        loss=f"{loss.item():.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        step=global_step,
                    )
                if global_step % cfg.log_every == 0 and accelerator.is_main_process:
                    payload = {
                        "train/loss": loss.item(),
                        "train/loss_hier": loss_h.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }
                    payload.update({f"train/loss_L{lvl}": v.item() for lvl, v in per_lvl.items()})
                    if loss_c is not None:
                        payload["train/loss_country"] = loss_c.item()
                    accelerator.log(payload, step=global_step)
                    log.info(f"step {global_step} loss={loss.item():.4f}")

        # ---- validation ----
        model.eval()
        per_batch = []
        use_country = countries is not None
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"val (epoch {epoch})",
                disable=not accelerator.is_main_process,
                unit="batch", dynamic_ncols=True,
            ):
                batch = _move_batch(batch, accelerator.device)
                out = model(
                    batch["pixel_values"], teacher_forcing=None,
                    return_country=use_country,
                )
                if use_country:
                    val_logits, val_country_logits = out
                else:
                    val_logits, val_country_logits = out, None
                per_batch.append(evaluate_batch(
                    val_logits, batch["cell_indices"], batch["latlng"], cells,
                    country_logits=val_country_logits,
                    country_idx=batch["country_idx"] if use_country else None,
                ))
        if accelerator.is_main_process:
            val_metrics = aggregate_metrics(per_batch, cells.levels)
            payload = {f"val/{k}": v for k, v in val_metrics.items()}
            accelerator.log(payload, step=global_step)
            country_str = (
                f" country_top1={val_metrics.get('country_top1_acc', 0):.3f}"
                if "country_top1_acc" in val_metrics else ""
            )
            log.info(
                f"epoch {epoch} val "
                f"median={val_metrics.get('median_km'):.1f}km "
                + " ".join(f"L{lvl}_top1={val_metrics.get(f'L{lvl}_top1_acc', 0):.3f}"
                           for lvl in cells.levels)
                + country_str
            )

            elapsed_min = (_time.time() - epoch_t0) / 60.0
            embed_fields = {
                "epoch": f"{epoch}/{cfg.epochs - 1}",
                "median_km": f"{val_metrics.get('median_km', 0):.1f}",
                "mean_km": f"{val_metrics.get('mean_km', 0):.1f}",
                "within_25km": f"{100 * val_metrics.get('within_25km', 0):.1f}%",
                "within_200km": f"{100 * val_metrics.get('within_200km', 0):.1f}%",
            }
            for lvl in cells.levels:
                embed_fields[f"L{lvl}_top1"] = f"{val_metrics.get(f'L{lvl}_top1_acc', 0):.3f}"
            if "country_top1_acc" in val_metrics:
                embed_fields["country_top1"] = f"{val_metrics['country_top1_acc']:.3f}"
            embed_fields["epoch_minutes"] = f"{elapsed_min:.0f}"
            embed_fields["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
            _post_discord_embed(
                title=f"{cfg.wandb_run_name or 'geoai-stage1'} — epoch {epoch} complete",
                fields=embed_fields,
                color=0x3BA55D,  # green — "done"
            )

        ckpt = cfg.out_dir / f"epoch_{epoch:02d}"
        _save_ckpt(
            accelerator, ckpt, model, optimizer, scheduler,
            global_step, epoch + 1, countries=countries,
        )

    accelerator.end_training()


if __name__ == "__main__":
    app()
