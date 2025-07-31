import os
import json


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

VS_CODE_SET_JSON_PATH = f"/{CURRENT_DIR}/.vscode"

SETTINGS_PATH = os.path.join(VS_CODE_SET_JSON_PATH, "settings.json")


os.makedirs(VS_CODE_SET_JSON_PATH, exist_ok=True)

ISAACSIM_ROOT = os.path.expanduser("~/isaacsim")  
EXTS_DIR = os.path.join(ISAACSIM_ROOT, "exts")
EXTS_DEPR_DIR = os.path.join(ISAACSIM_ROOT, "extsDeprecated")
EXTS_PHY_DIR = os.path.join(ISAACSIM_ROOT, "extsPhysics")
EXTS_CACHE_DIR = os.path.join(ISAACSIM_ROOT, "extscache")
KIT = os.path.join(ISAACSIM_ROOT, "kit")

paths = [EXTS_DIR, EXTS_DEPR_DIR, EXTS_PHY_DIR, EXTS_CACHE_DIR, KIT]


all_path = []
for path in paths:
    for root, dir, files in os.walk(path):
        for d in dir:
            fp = os.path.join(path, d)
            all_path.append(fp)
            
        break
    
# for root, dir, files in os.walk(KIT):
#     for d in dir:
#         fp = os.path.join(KIT, d)
#         all_path.append(fp)
        

settings = {
    "python.analysis.extraPaths": all_path
}

with open(SETTINGS_PATH, "w") as f:
    json.dump(settings, f, indent=4)

print("settings.json generated.")


