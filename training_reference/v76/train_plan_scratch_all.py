# Authorized reference implementation; not an end-to-end runnable training release.
# See training_reference/README.md and NOTICE.md.
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/p) for p in ("v67","v40","v20","v14")]
import train_plan_bc as v67
from features_v40 import TACTIC_DIM,feature_mask_for
from model_v67 import V67PlanTransformer,policy_state_dict
DATASET=ROOT/"v76/artifacts/structured_v40_history.npz";META=ROOT/"v76/artifacts/metadata.json";PLANS=ROOT/"v76/artifacts/plan_targets.npz";OUT=ROOT/"v76/artifacts"

def main():
 p=argparse.ArgumentParser();p.add_argument("--run-name",default="deck_3121746f2b28_v76_plan_scratch_latest");p.add_argument("--batch-size",type=int,default=128);p.add_argument("--phase1-epochs",type=int,default=20);p.add_argument("--phase2-epochs",type=int,default=8);p.add_argument("--device",choices=("cuda","cpu"),default="cuda");p.add_argument("--seed",type=int,default=7601);a=p.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);output=OUT/a.run_name;output.mkdir(parents=True,exist_ok=True)
 da=np.load(DATASET,mmap_mode="r");data={k:da[k] for k in da.files};pa=np.load(PLANS,mmap_mode="r");plans={k:pa[k] for k in pa.files}
 if len(data["split"])!=len(plans["source_split"]) or not np.array_equal(data["split"],plans["source_split"]):raise RuntimeError("plan target split mismatch")
 meta=json.loads(META.read_text(encoding="utf-8"));meta["target_deck"]="3121746f2b28";meta["target_deck_index"]=int(meta["deck_ids"].index(meta["target_deck"]));mask=feature_mask_for("card",TACTIC_DIM);meta["feature_variant"]="card";meta["feature_mask"]=mask.tolist()
 weights=v67.base.weights_for(data,meta,"anti_bcfc_strong");loaders=[v67.make_loader(data,weights,plans,s,a.batch_size if s==0 else a.batch_size*2,s==0,mask) for s in range(3)]
 device=torch.device(a.device if a.device=="cpu" or torch.cuda.is_available() else "cpu");model=V67PlanTransformer(len(meta["deck_ids"])).to(device);initial=v67.evaluate(model,loaders[1],device)
 cfg={"scratch":True,"warm_start":False,"all_weights_trainable":True,"parameters":sum(x.numel() for x in model.parameters()),"device":str(device),"train":int(((data["split"]==0)&(weights>0)).sum()),"valid":int(((data["split"]==1)&(weights>0)).sum()),"test":int(((data["split"]==2)&(weights>0)).sum()),"initial":initial};print("V76_CONFIG="+json.dumps(cfg),flush=True)
 history=[];epoch=0
 for phase,epochs,lr,pw in (("scratch_fast",a.phase1_epochs,3e-4,.18),("scratch_refine",a.phase2_epochs,8e-5,.12)):
  for parameter in model.parameters():parameter.requires_grad=True
  optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=6e-4)
  for _ in range(epochs):
   loss,top1=v67.run_epoch(model,loaders[0],optimizer,device,pw,1.0);valid=v67.evaluate(model,loaders[1],device)
   row={"epoch":epoch,"phase":phase,"lr":lr,"plan_weight":pw,"train_loss":loss,"train_top1":top1,**{"valid_"+k:v for k,v in valid.items()}};history.append(row)
   cpu={k:v.detach().cpu() for k,v in model.state_dict().items()};torch.save(cpu,output/f"epoch_{epoch:02d}.pt");(output/"history_partial.json").write_text(json.dumps(history,indent=2),encoding="utf-8");print(json.dumps(row),flush=True);epoch+=1
 selected=max(history,key=lambda r:(float(r["valid_top1"]),-float(r["valid_listwise_loss"]),float(r["valid_next_type_acc"])));state=torch.load(output/f"epoch_{int(selected['epoch']):02d}.pt",map_location="cpu",weights_only=True);model.load_state_dict(state);model.to(device);policy=policy_state_dict(state)
 torch.save(policy,output/"policy_v40.pt");np.savez_compressed(output/"policy_v40.npz",**{k:v.detach().cpu().numpy() for k,v in policy.items()});test=v67.evaluate(model,loaders[2],device)
 meta["v76_plan"]={"scratch":True,"warm_start":False,"all_weights_trainable":True,"selected_epoch":int(selected["epoch"]),"plan_targets":str(PLANS)};(output/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8");(output/"variant_config.json").write_text(json.dumps({"feature_variant":"card","feature_mask":mask.tolist()},indent=2),encoding="utf-8")
 result={"run_name":a.run_name,"config":cfg,"selected":selected,"test":test,"history":history};(output/"training.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print("FINAL_V76="+json.dumps({"run_name":a.run_name,"scratch":True,"selected":selected,"test":test}),flush=True)
if __name__=="__main__":main()
