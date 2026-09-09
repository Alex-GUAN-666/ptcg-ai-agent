# Authorized reference implementation; not an end-to-end runnable training release.
# See training_reference/README.md and NOTICE.md.
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "v67"), str(ROOT / "v40"), str(ROOT / "v20"), str(ROOT / "v14")]

from features_v40 import TACTIC_DIM, feature_mask_for
from model_v67 import V67PlanTransformer, policy_state_dict

spec = importlib.util.spec_from_file_location("v40_train_for_v67", ROOT / "v40/train.py")
base = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(base)

DATASET = ROOT / "v66/artifacts/structured_v40_history.npz"
DATA_META = ROOT / "v66/artifacts/metadata.json"
PLAN_TARGETS = ROOT / "v67/artifacts/plan_targets.npz"
SOURCE_RUNS = {
    "strong": ROOT / "v40/artifacts/deck_3121746f2b28_v40_anti_bcfc_strong",
    "mild": ROOT / "v40/artifacts/deck_3121746f2b28_v40_anti_bcfc_mild",
}
ARTIFACT_ROOT = ROOT / "v67/artifacts"
TARGET_DECK = "3121746f2b28"


class PlanCollator(base.Collator):
    def __init__(self, data, weights, plans, feature_mask):
        super().__init__(data, weights, feature_mask)
        self.plans = plans

    def __call__(self, decisions):
        result = super().__call__(decisions)
        idx = np.asarray(decisions, np.int64)
        result["plan_multi"] = torch.from_numpy(self.plans["plan_multi"][idx].astype(np.float32))
        result["next_type"] = torch.from_numpy(self.plans["next_type"][idx].astype(np.int64))
        result["next_card"] = torch.from_numpy(self.plans["next_card"][idx].astype(np.int64))
        result["atomic_weight"] = torch.from_numpy(self.plans["atomic_weight"][idx].astype(np.float32))
        return result


def make_loader(data, weights, plans, split, batch_size, shuffle, feature_mask):
    indices = np.flatnonzero((data["split"] == split) & (weights > 0))
    if not len(indices):
        raise RuntimeError(f"empty unchanged split={split}")
    return DataLoader(base.Decisions(indices), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, collate_fn=PlanCollator(data, weights, plans, feature_mask))


def forward(model, batch):
    return model(batch["state"], batch["cards"], batch["zones"], batch["entity_cards"], batch["entity_zones"], batch["entity_nums"], batch["state_nums"], batch["history_nums"], batch["resource_nums"], batch["action_features"], batch["action_cards"], batch["action_nums"], batch["tactic_nums"], batch["deck"], batch["action_mask"])


def load_v40(model, source_run: Path, target_meta: dict) -> dict:
    source = torch.load(source_run / "policy_v40.pt", map_location="cpu", weights_only=True)
    source_meta = json.loads((source_run / "metadata.json").read_text(encoding="utf-8"))
    target = model.state_dict(); copied = partial = 0
    for key, value in source.items():
        if key not in target:
            continue
        if target[key].shape == value.shape:
            target[key] = value.clone(); copied += 1
        elif key == "base.deck.weight":
            current = target[key].clone(); current[:] = value.mean(0, keepdim=True)
            source_map = {deck_id: i for i, deck_id in enumerate(source_meta["deck_ids"])}
            for i, deck_id in enumerate(target_meta["deck_ids"]):
                if deck_id in source_map:
                    current[i] = value[source_map[deck_id]]
            target[key] = current; partial += 1
    model.load_state_dict(target)
    return {"source": str(source_run), "copied": copied, "partial": partial}


def plan_loss(plans, batch):
    multi_logits, type_logits, card_logits = plans
    weight = batch["atomic_weight"]
    weight = weight / weight.mean().clamp_min(1e-6)
    multi = F.binary_cross_entropy_with_logits(multi_logits.float(), batch["plan_multi"], reduction="none").mean(1)
    type_loss = F.cross_entropy(type_logits.float(), batch["next_type"], reduction="none")
    card_loss = F.cross_entropy(card_logits.float(), batch["next_card"], reduction="none")
    return ((multi + 0.35 * type_loss + 0.12 * card_loss) * weight).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); hits = base_hits = count_hits = items = 0; loss_sum = plan_sum = 0.0; type_hits = type_items = 0
    for raw in loader:
        batch = base.move(raw, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            logits, base_logits, _, _, count_logits, plans = forward(model, batch)
        logp = torch.log_softmax(logits.float(), 1)
        loss_sum += (-(batch["targets"] * logp).sum(1)).sum().item()
        best = logits.argmax(1); base_best = base_logits.argmax(1)
        hits += batch["expert_actions"].gather(1, best[:, None]).sum().item()
        base_hits += batch["expert_actions"].gather(1, base_best[:, None]).sum().item()
        count_hits += (count_logits.masked_fill(~batch["count_mask"], -1e4).argmax(1) == batch["count_label"]).sum().item()
        plan_sum += plan_loss(plans, batch).item() * len(best)
        mask = batch["next_type"] > 0
        type_hits += (plans[1].argmax(1)[mask] == batch["next_type"][mask]).sum().item(); type_items += mask.sum().item()
        items += len(best)
    return {"top1": hits/max(items,1), "base_top1": base_hits/max(items,1), "count_acc": count_hits/max(items,1), "listwise_loss": loss_sum/max(items,1), "plan_loss": plan_sum/max(items,1), "next_type_acc": type_hits/max(type_items,1), "decisions": items}


def run_epoch(model, loader, optimizer, device, plan_weight, policy_scale):
    model.train(); scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"); total = hits = items = 0
    for raw in loader:
        batch = base.move(raw, device); optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            logits, base_logits, residual, gate, count_logits, plans = forward(model, batch)
            logp = torch.log_softmax(logits.float(), 1); base_logp = torch.log_softmax(base_logits.float(), 1)
            per = -(batch["targets"] * logp).sum(1); base_per = -(batch["targets"] * base_logp).sum(1)
            bc = (per * batch["weights"]).sum() / batch["weights"].sum().clamp_min(1e-6)
            base_bc = (base_per * batch["weights"]).sum() / batch["weights"].sum().clamp_min(1e-6)
            pair = base.pairwise_loss(logits.float(), batch["expert_actions"], batch["action_mask"])
            cnt = base.count_loss(count_logits.float(), batch["count_mask"], batch["count_label"])
            aux = plan_loss(plans, batch)
            policy = bc + 0.30 * base_bc + 0.07 * pair + 0.10 * cnt + 0.001 * ((residual.square()*batch["action_mask"]).sum()/batch["action_mask"].sum()) + 0.0005 * ((gate*batch["action_mask"]).sum()/batch["action_mask"].sum())
            loss = policy_scale * policy + plan_weight * aux
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
        hits += batch["expert_actions"].gather(1, logits.argmax(1)[:,None]).sum().item(); items += len(logits); total += loss.item()*len(logits)
    return total/max(items,1), hits/max(items,1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="deck_3121746f2b28_v67_plan_atomic")
    parser.add_argument("--anchor", choices=("strong", "mild"), default="strong")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--joint-epochs", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=6767)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output = ARTIFACT_ROOT / args.run_name; output.mkdir(parents=True, exist_ok=True)
    data_archive = np.load(DATASET, mmap_mode="r"); data = {k:data_archive[k] for k in data_archive.files}
    plan_archive = np.load(PLAN_TARGETS, mmap_mode="r"); plans = {k:plan_archive[k] for k in plan_archive.files}
    if len(data["split"]) != len(plans["source_split"]) or not np.array_equal(data["split"], plans["source_split"]):
        raise RuntimeError("plan targets do not preserve the dataset split")
    meta = json.loads(DATA_META.read_text(encoding="utf-8")); meta["target_deck"] = TARGET_DECK; meta["target_deck_index"] = int(meta["deck_ids"].index(TARGET_DECK))
    feature_mask = feature_mask_for("card", TACTIC_DIM); meta["feature_variant"] = "card"; meta["feature_mask"] = feature_mask.tolist()
    weights = base.weights_for(data, meta, "anti_bcfc_strong" if args.anchor == "strong" else "anti_bcfc_mild")
    loaders = [make_loader(data, weights, plans, split, args.batch_size if split == 0 else args.batch_size*2, split == 0, feature_mask) for split in range(3)]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = V67PlanTransformer(len(meta["deck_ids"])); init = load_v40(model, SOURCE_RUNS[args.anchor], meta); model.to(device)
    initial = evaluate(model, loaders[1], device); history=[]; torch.save(model.state_dict(), output/"epoch_-1.pt")
    print("V67_INITIAL="+json.dumps({"split_unchanged":True,"anchor":args.anchor,"init":init,"valid":initial}), flush=True)
    epoch = 0
    for phase, epochs, lr, heads_only, plan_weight, policy_scale in (("plan_head",args.head_epochs,2e-4,True,0.30,0.0),("joint",args.joint_epochs,5e-5,False,0.16,1.0)):
        for p in model.parameters(): p.requires_grad = (not heads_only) or any(name.startswith("plan_") and p is value for name,value in model.named_parameters())
        params=[p for p in model.parameters() if p.requires_grad]; optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=6e-4)
        for _ in range(epochs):
            train_loss, train_top1 = run_epoch(model,loaders[0],optimizer,device,plan_weight,policy_scale)
            valid=evaluate(model,loaders[1],device); row={"epoch":epoch,"phase":phase,"lr":lr,"train_loss":train_loss,"train_top1":train_top1,**{"valid_"+k:v for k,v in valid.items()}}
            history.append(row); torch.save(model.state_dict(),output/f"epoch_{epoch:02d}.pt"); print(json.dumps(row),flush=True); epoch+=1
    initial_top=float(initial["top1"]); eligible=[r for r in history if float(r["valid_top1"]) >= initial_top-0.002]
    selected=max(eligible,key=lambda r:(float(r["valid_top1"]),float(r["valid_next_type_acc"]))) if eligible else {"epoch":-1,**{"valid_"+k:v for k,v in initial.items()}}
    checkpoint=output/("epoch_-1.pt" if int(selected["epoch"])<0 else f"epoch_{int(selected['epoch']):02d}.pt")
    state=torch.load(checkpoint,map_location="cpu",weights_only=True); model.load_state_dict(state); policy=policy_state_dict(state)
    torch.save(policy,output/"policy_v40.pt"); np.savez_compressed(output/"policy_v40.npz",**{k:v.cpu().numpy() for k,v in policy.items()})
    test=evaluate(model,loaders[2],device); meta["v67_plan"]={"anchor":args.anchor,"split_unchanged":True,"selected_epoch":int(selected["epoch"]),"plan_targets":str(PLAN_TARGETS)}
    (output/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); (output/"variant_config.json").write_text(json.dumps({"feature_variant":"card","feature_mask":feature_mask.tolist()},indent=2),encoding="utf-8")
    result={"run_name":args.run_name,"initial":initial,"selected":selected,"test":test,"history":history,"split_unchanged":True}; (output/"training.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("FINAL_V67_PLAN="+json.dumps({"run_name":args.run_name,"selected":selected,"test":test,"split_unchanged":True}),flush=True)


if __name__ == "__main__":
    main()
