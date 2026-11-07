"""Physical reach/speed diagnostics, independent of the teacher's phase labels."""
from student_common import setup,write_json,sha
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
a=p.parse_args();setup(a.bundle)
import numpy as np,mujoco
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as R
from holoboxing.data.contract import CompetitionAssets
from holoboxing.data.robot import load_scene_model
s=json.loads((a.source/'status.json').read_text());assert s['state']=='completed'
z=np.load(a.source/'states.npz');assert z['qpos'].shape[1]==1
m=load_scene_model(CompetitionAssets.load(Path('/home/zhanghan/datasets/hbot-assets')));d=mujoco.MjData(m)
root=m.body('p1_pelvis').id;other=m.body('p2_pelvis').id
gloves=[m.geom('p1_'+h+'_glove_collision').id for h in ['left','right']]
arms=[[m.body('p1_'+h+'_'+j+'_link').id for j in ['shoulder_roll','elbow','wrist_yaw']] for h in ['left','right']]
world=[];relative=[];elbows=[];distances=[];tilts=[]
for q in z['qpos'][:,0]:
    d.qpos[:]=q;mujoco.mj_kinematics(m,d)
    rot=R.from_quat(d.xquat[root][[1,2,3,0]]);yaw=rot.as_euler('xyz')[2]+.04432690669811633
    relative.append((d.geom_xpos[gloves]-d.xpos[root])@R.from_euler('z',yaw).as_matrix());world.append(d.geom_xpos[gloves].copy())
    distances.append(np.linalg.norm(d.xpos[root,:2]-d.xpos[other,:2]));tilts.append(np.degrees(np.arccos(np.clip(rot.as_matrix()[2,2],-1,1))))
    angles=[]
    for sh,el,wr in arms:
        u=d.xpos[sh]-d.xpos[el];v=d.xpos[wr]-d.xpos[el]
        angles.append(np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))
    elbows.append(angles)
relative=np.array(relative);world=np.array(world);elbows=np.array(elbows);distances=np.array(distances)
speed=np.linalg.norm(np.gradient(world,.02,axis=0),axis=-1)
fall=s['first_fall_seconds'][0][0];usable=np.arange(len(world))*.02<(fall if fall is not None else 1e9)
close=(distances<.75)&usable;rows=[]
for hand in [0,1]:
    reach=relative[:,hand,0];peaks,_=find_peaks(reach,prominence=.12,distance=8,height=.30)
    for pk in peaks:
        if not close[pk]:continue
        begin=max(0,pk-15);end=min(len(world),pk+16)
        rows.append(dict(hand=['left','right'][hand],peak_s=(pk+1)*.02,forward_reach_m=float(reach[pk]),elbow_degrees=float(elbows[pk,hand]),outgoing_peak_speed_m_s=float(speed[begin:pk+1,hand].max()),return_reach_m=float(reach[pk:end].min())))
def summarize(values):
    return dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),max=float(np.max(values))) if len(values) else None
r=dict(source=str(a.source),states_sha256=sha(a.source/'states.npz'),duration_s=s['simulation_seconds'],hits=s['hits'],falls=s['first_fall_seconds'],close_range_seconds=float(close.sum()*.02),detected_extension_peaks=len(rows),criterion='Glove forward peak prominence >=0.12m, reach >=0.30m, spacing >=0.16s, distance <0.75m; descriptive only, not official hits',peak_elbow_degrees=summarize([v['elbow_degrees'] for v in rows]),peak_reach_m=summarize([v['forward_reach_m'] for v in rows]),outgoing_peak_speed_m_s=summarize([v['outgoing_peak_speed_m_s'] for v in rows]),near_wrist_speed_m_s=summarize(speed[close].flatten()),tilt_degrees=summarize(np.array(tilts)[usable]),extension_events=rows)
write_json(a.source/'motion-audit.json',r);print(json.dumps({k:v for k,v in r.items() if k!='extension_events'}),flush=True)
