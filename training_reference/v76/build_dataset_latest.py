# Authorized reference implementation; not an end-to-end runnable training release.
# See training_reference/README.md and NOTICE.md.
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUTPUT=ROOT/"v76/artifacts"; TARGET="3121746f2b28"
KEEP_CONFIG={
 "2026-07-20":1000,"2026-07-21":1000,"2026-07-22":1000,"2026-07-23":1000,"2026-07-24":1000,
 "2026-07-25":1000,"2026-07-26":1000,"2026-07-27":1000,"2026-07-28":1000,
 "2026-07-29":100000,"2026-07-30":100000,"2026-07-31":100000,
 "2026-08-01":100000,"2026-08-02":100000,"2026-08-03":100000,
}
def main():
 p=argparse.ArgumentParser();p.add_argument("--max-episodes",type=int,default=0);a=p.parse_args()
 source=ROOT/"v40/build_dataset.py";spec=importlib.util.spec_from_file_location("v76_builder",source)
 builder=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(builder)
 builder.OUTPUT=OUTPUT;builder.KEEP_CONFIG=KEEP_CONFIG
 sys.argv=[str(source),"--target-deck",TARGET,"--target-only","--max-episodes",str(a.max_episodes)];builder.main()
if __name__=="__main__":main()
