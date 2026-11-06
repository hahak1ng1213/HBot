import sys,json,collections
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from holoboxing.data.contract import CompetitionAssets
from holoboxing.data.robot import load_scene_model
from stance_alignment_control import geometry
config=json.loads((ROOT/'alignment.json').read_text())
assets=CompetitionAssets.load(Path('/home/zhanghan/datasets/hbot-assets'));model=load_scene_model(assets);data=mujoco.MjData(model)
names=['pelvis','left_wrist_yaw_link','right_wrist_yaw_link','left_ankle_roll_link','right_ankle_roll_link']
ids=np.array([[model.body(prefix+n).id for n in names] for prefix in ['p1_','p2_']])
old=ROOT.parent/'HBot-standard-five-range060-20261106'
def summarize(folder,baseline=False):
    report=json.loads((folder/'report.json').read_text());decisions=json.loads((folder/'decisions.json').read_text())
    metrics=json.loads((folder/'metrics.json').read_text());states=np.load(folder/'states.npz')
    attacks=[r for r in decisions if r['skill'] in ['straight_lr','uppercut_lr']]
    values=[]
    for r in attacks:
        if 'alignment' in r:g=r['alignment']
        else:
            data.qpos[:]=states['qpos'][r['frame']-1];mujoco.mj_kinematics(model,data)
            g=geometry(data.xpos[ids],data.xquat[ids],[3,4],config)
        values.append(g)
    result=dict(seconds=report['simulation_seconds'],hits=report['hits'],falls=report['first_fall_seconds'],invalid=report['invalid'],
                decision_counts=dict(collections.Counter(r['skill'] for r in decisions)),attack_decisions=len(attacks),
                alignment_at_attack={})
    for k in ['pelvis_error_deg','feet_error_deg']:
        v=np.abs([g[k] for g in values])
        result['alignment_at_attack'][k]=dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(np.max(v))) if len(v) else None
    result['attack_meeting_new_angle_gate']=sum(g['aligned'] for g in values)
    cr=old/'impulse-check/contacts.json' if baseline else folder/'contacts.json'
    rows=json.loads(cr.read_text());result['max_20ms_impulse_ns']=[max([r.get('normal_impulse_ns',0) for r in rows if r['actor']==a and r['hand'] in ['left','right']],default=0) for a in [0,1]]
    result['effective_hits']=[r for r in rows if r['valid']]
    result['attack_starts']=[dict(time_s=r['time_s'],skill=r['skill'],**g) for r,g in zip(attacks,values)]
    return result
result=dict(baseline=summarize(old/'demo',True),candidate=summarize(ROOT/sys.argv[1]))
(ROOT/sys.argv[1]/'comparison.json').write_text(json.dumps(result,indent=2))
for k,v in result.items(): print(k,json.dumps({a:b for a,b in v.items() if a not in ['attack_starts','effective_hits']}))
