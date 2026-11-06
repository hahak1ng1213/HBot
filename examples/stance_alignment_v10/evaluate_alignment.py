"""Measured-state selector -> motion primitives -> frozen lower vs seed44."""
import os
os.environ.update(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='1',CUDA_VISIBLE_DEVICES='GPU-2a18a801-bbc4-8c59-9f5f-ce5839904141',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LP_NUM_THREADS='1')
import sys,json,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),'/home/zhanghan/tools/hbot-video']
import torch
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R,Slerp
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from holoboxing.data.contract import CompetitionAssets
from holoboxing.training.resources import resolve_resources
from holoboxing.training.export import load_policy
from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
from holoboxing.training.tasks.combat.config.sandbag import BODY_NAMES,POINTS
from holoboxing.training.tasks.combat.mdp.geometry import encode_output

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def yaw(q):return R.from_quat(np.asarray(q)[[1,2,3,0]]).as_euler('xyz')[2]
def feature(raw,offset):
    p=raw['position'][0,:,0].cpu().numpy();q=raw['quaternion'][0,:,0].cpu().numpy();v=raw['velocity'][0,1,0].cpu().numpy()
    h=yaw(q[0])+offset;c,s=np.cos(h),np.sin(h);rot=np.array([[c,s,0],[-s,c,0],[0,0,1]])
    delta=rot@(p[1]-p[0]);velocity=rot@v;d=np.linalg.norm(delta[:2]);a=np.arctan2(delta[1],delta[0]);oh=yaw(q[1])-h
    return np.r_[delta,velocity,d,np.sin(a),np.cos(a),np.sin(oh),np.cos(oh)],d,a

assert sha(ROOT/'holoboxing/training/envs/control.py')=='2f2099828bc470a6e4d69278a84c74c90c46d7bef5cee9cf88aced5e50b20056', 'Require original standard five-point controller'
torch.set_num_threads(1);torch.manual_seed(273);np.random.seed(273)
parser=argparse.ArgumentParser();parser.add_argument('--adaptive',action='store_true');parser.add_argument('--output',default='demo');args=parser.parse_args();out=ROOT/args.output;out.mkdir(exist_ok=False)
blue_path=Path('/home/zhanghan/runs/hbot/approach-strike-node02-20261101-001552/seed44/train/model_1536.pt')
assert sha(blue_path)=='b5af327c5fc4754d9d3586f5ae96b00f36f5f8f7619753f6e4cb22fb96f9cdc5'
data=ROOT/'candidate_50hz';integrity=json.loads((data/'integrity.json').read_text())
for filename,expected in integrity.items():assert sha(data/filename)==expected,filename
rules=None
if (data/'rules.json').exists():
    sys.path.insert(0,str(data))
    from boxing_skill_control import enforce_selection,validate_rules
    rules=json.loads((data/'rules.json').read_text());validate_rules(rules)
saved=torch.load(data/'selector.pt',map_location='cpu',weights_only=True)
model=torch.nn.Sequential(torch.nn.Linear(11,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,4))
model.load_state_dict(saved['state_dict'],strict=True);model.eval().requires_grad_(False)
blue=load_policy(blue_path).to('cuda:0').eval().requires_grad_(False)
blue_tensors={k:v.clone() for k,v in blue.state_dict().items()}
skills=['forward_short','turn_left','turn_right','straight_lr','uppercut_lr'];bank={k:dict(np.load(data/'motions'/(k+'.npz'))) for k in skills}
offset=json.loads((data/'features.json').read_text())['heading_offset_from_pelvis_rad']
ref=bank['straight_lr'];point_ids=[ref['body_names'].tolist().index(n) for n in POINTS];actual_ids=[BODY_NAMES.index(n) for n in POINTS]
assert tuple(POINTS)==('pelvis','left_wrist_yaw_link','right_wrist_yaw_link','left_ankle_roll_link','right_ankle_roll_link')
assert len(point_ids)==len(actual_ids)==5
if args.adaptive:
    for path in (ROOT/'approach').glob('*.npz'):bank[path.stem]=dict(np.load(path))
    hold={k:v.copy() for k,v in ref.items()}
    for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
        hold[k]=np.repeat(ref[k][:1],7,axis=0)
        if 'vel' in k:hold[k][:]=0
    bank['settle_guard']=hold
from stance_alignment_control import geometry,choose
alignment_config=json.loads((ROOT/'alignment.json').read_text())
feet_ids=[BODY_NAMES.index(s+'_ankle_roll_link') for s in ['left','right']]
from holoboxing.evaluation.runtime.rules import MatchStateMachine
_observe=MatchStateMachine.observe_frame
contact_rows=[]
def observed(self,*args,**kwargs):
    result=_observe(self,*args,**kwargs)
    contact_rows.extend(dict(r) for r in self.contact_diagnostics)
    return result
MatchStateMachine.observe_frame=observed
origin=ref['body_pos_w'][0,0].astype(float)
assets_path=Path('/home/zhanghan/datasets/hbot-assets');lower_path=Path('/home/zhanghan/datasets/holomotion-v2')
assets=CompetitionAssets.load(assets_path);_,paths,resources=resolve_resources(assets_path,lower_path)
report=dict(state='initializing',seed=273,deterministic=True,device='cuda:0 on node01 physical GPU1',render='none',candidate_sha256=sha(data/'selector.pt'),
    blue_checkpoint=str(blue_path),blue_sha256=sha(blue_path),resources=resources,mid_episode_resets=False,
    official_qualification=False,physics='native MuJoCo CPU with official contact referee',
    initialization='training environment seeded reset; 1s smooth entry to ready; blue active throughout',
    decision='Measured pelvis + feet alignment, adaptive 5/10/20deg turns, no retraining',
    research_extra_elbow_targets=False,research_extra_body_targets=[],adaptive_approach=args.adaptive,standard_C225_only=True,controller_rules=rules,motion_version='v8-roomy-guard',motion_hashes={k:sha(data/'motions'/(k+'.npz')) for k in skills},approach_hashes={p.name:sha(p) for p in (ROOT/'approach').glob('*.npz')},
    scheduling='decision at completed clip boundaries; fixed world reference composed across clips; last lookahead clamped at clip end',
    measured_inputs='Original 11 selector features plus measured own/opponent pelvis and ankles for alignment; standard five-point targets only',max_simulation_seconds=60.,
    source_files={str(p.relative_to(ROOT)):sha(p) for folder in ['holoboxing','holomotion'] for p in (ROOT/folder).rglob('*.py')})
def save(): (out/'report.json').write_text(json.dumps(report,indent=2))
save();env=None;renderer=None;writer=None;rows=[];velrows=[];decisions=[];metrics=[];features=[];target_rows=[];tracking=[];hits=[0,0];first_fall=[None,None];started=time.monotonic()
try:
    env=MujocoSandbagEnv(assets,paths,1,'cuda:0',ROOT/'runtime',273,official_rules=True,capture_transition_state=True,physics_threads=1)
    assert all(not hasattr(c,'extra_reference') for c in env.controllers.values())
    obs=env.reset();env.next_push[:]=float('inf');raw=env.physics.state()
    position=raw['position'][0,0,0].cpu().numpy().copy();position[2]=origin[2]+env.physics.floor_height
    delta=raw['position'][0,1,0,:2].cpu().numpy()-position[:2];attack_heading=np.arctan2(delta[1],delta[0])
    rotation=R.from_euler('z',attack_heading-(yaw(ref['body_quat_w'][0,0])+offset))
    def targets(z,indices):
        p=z['body_pos_w'][indices][:,point_ids].astype(float);q=z['body_quat_w'][indices][:,point_ids].astype(float)
        p=rotation.apply((p-origin).reshape(-1,3)).reshape(p.shape)+position
        q=(rotation*R.from_quat(q.reshape(-1,4)[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]].reshape(q.shape)
        return p,q
    endp,endq=targets(ref,np.array([0]));startp=raw['position'][0,0,actual_ids].cpu().numpy();startq=raw['quaternion'][0,0,actual_ids].cpu().numpy()
    entry_p=[];entry_q=[]
    for f in range(51):
        u=f/50;u=u**3*(10-15*u+6*u*u);entry_p.append(startp*(1-u)+endp[0]*u)
        entry_q.append(np.stack([Slerp([0,1],R.from_quat(np.stack([startq[j],endq[0,j]])[:,[1,2,3,0]]))([u]).as_quat()[0,[3,0,1,2]] for j in range(len(point_ids))]))
    entry_p=np.array(entry_p);entry_q=np.array(entry_q)
    physics_model=env.physics.models[0];physics_model.vis.global_.offwidth=960;physics_model.vis.global_.offheight=640
    display=mujoco.MjData(physics_model)
    def capture(base):
        d=base.physics.native[0];return dict(demo_qpos=torch.from_numpy(d.qpos.copy())[None],demo_qvel=torch.from_numpy(d.qvel.copy())[None])
    env.transition_measurement=capture
    phase=0;active=None;previous_dist=None;report['state']='running';save()
    with torch.inference_mode():
        for f in range(3000):
            raw=env.physics.state();alignment=geometry(raw['position'][0].cpu().numpy(),raw['quaternion'][0].cpu().numpy(),feet_ids,alignment_config);x,dist,bearing=feature(raw,offset);features.append(x);closing=0. if previous_dist is None else np.clip((previous_dist-dist)/.02,0,1.);previous_dist=dist
            if f<50:
                indices=np.minimum(f+np.arange(1,6),50);tp,tq=entry_p[indices],entry_q[indices];label='ready-entry'
            else:
                if active is None or phase>=len(bank[active]['joint_pos'])-1:
                    finished=active
                    if active is not None:
                        z=bank[active];p,q=targets(z,np.array([len(z['joint_pos'])-1]));position=p[0,0].copy();rotation=R.from_quat(q[0,0,[1,2,3,0]])*R.from_quat(ref['body_quat_w'][0,0,[1,2,3,0]]).inv()
                    logits=model((torch.tensor(x,dtype=torch.float32)-saved['mean'])/saved['std']);probs=logits.softmax(-1).numpy();choice=int(np.argmax(probs))
                    proposed=choice;overridden=False
                    reference_heading=yaw((rotation*R.from_quat(ref['body_quat_w'][0,0,[1,2,3,0]])).as_quat()[[3,0,1,2]])+offset
                    choice,active,reference_error=choose(alignment,reference_heading,alignment_config,rules)
                    overridden=choice!=proposed;phase=0
                    if args.adaptive:
                        if finished is not None and finished.startswith('forward') and dist<=rules['settle_after_forward_max_m']:
                            active='settle_guard'
                        elif choice==0 and dist<rules['short_step_start_m']:
                            budget=max(.03,min(.20,float(dist)-rules['approach_target_m']-float(closing)*.20))
                            length=max(cm for cm in [3,5,8,12,20] if cm/100<=budget+1e-9)
                            active='forward_short' if length==20 else f'forward_{length:02}cm'
                    decisions.append(dict(frame=f,time_s=f*.02,class_id=(-1 if active=='settle_guard' else choice),model_class_id=proposed,rule_overridden=overridden,skill=active,distance_m=float(dist),bearing_deg=float(np.rad2deg(bearing)),closing_speed_est_m_s=float(closing),too_close=bool(dist<(rules['too_close_m'] if rules else .35)),probabilities=probs.tolist(),alignment=alignment,reference_error_deg=reference_error))
                z=bank[active];indices=np.minimum(phase+np.arange(1,6),len(z['joint_pos'])-1);tp,tq=targets(z,indices);label=active;phase+=1
            assert tp.shape==(5,5,3) and tq.shape==(5,5,4)
            red=encode_output(raw['position'][:,0,actual_ids[:5]],raw['quaternion'][:,0,actual_ids[:5]],torch.tensor(tp[:,:5],dtype=torch.float32,device=env.device)[None],torch.tensor(tq[:,:5],dtype=torch.float32,device=env.device)[None])
            assert red.shape==(1,225)
            obs,_,done,info=env.step({'robot_A':red,'robot_B':blue(obs['robot_B']['task_policy'])})
            transition=env.last_transition_state;display.qpos[:]=transition['demo_qpos'][0].numpy();display.qvel[:]=transition['demo_qvel'][0].numpy();mujoco.mj_forward(physics_model,display)
            rows.append(display.qpos.copy());velrows.append(display.qvel.copy());m={'time_s':(f+1)*.02,'distance_m':float(dist),'bearing_deg':float(np.rad2deg(bearing)),'invalid':bool(info['invalid_envs'][0]),'skill':label}
            for role,a in enumerate(['robot_A','robot_B']):
                stat=env.metrics[a];hits[role]+=int(stat['official_hits'][0]);m[a+'_score']=float(stat['official_score'][0]);m[a+'_fallen']=bool(stat['fallen'][0])
                if m[a+'_fallen'] and first_fall[role] is None:first_fall[role]=(f+1)*.02
            actual=transition['position'][0,0,actual_ids].cpu().numpy()
            errors=np.linalg.norm(actual-tp[0],axis=-1)
            m['target_error_pelvis_m']=float(errors[0]);m['target_error_left_wrist_m']=float(errors[1]);m['target_error_right_wrist_m']=float(errors[2])
            target_rows.append(tp[0].copy());tracking.append(errors)
            m.update(alignment=alignment);metrics.append(m)
            if (f+1)%100==0:
                report.update(simulation_seconds=(f+1)*.02,hits=hits,first_fall_seconds=first_fall,wall_seconds=time.monotonic()-started);save();print(json.dumps({k:report[k] for k in ['simulation_seconds','hits','wall_seconds']}),flush=True)
            if bool(done[0]) or m['invalid']:break
    writer=None
    (out/'contacts.json').write_text(json.dumps(contact_rows,indent=2))
    np.savez_compressed(out/'states.npz',qpos=np.stack(rows),qvel=np.stack(velrows),selector_input=np.stack(features),target_positions=np.stack(target_rows),tracking_error_m=np.stack(tracking))
    (out/'decisions.json').write_text(json.dumps(decisions,indent=2));(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    assert all(torch.equal(v,blue.state_dict()[k]) for k,v in blue_tensors.items());assert sha(blue_path)==report['blue_sha256']
    report.update(state='completed',simulation_seconds=len(rows)*.02,hits=hits,first_fall_seconds=first_fall,score=[metrics[-1]['robot_A_score'],metrics[-1]['robot_B_score']],
                  invalid=metrics[-1]['invalid'],terminal=bool(done[0]),decision_count=len(decisions),wall_seconds=time.monotonic()-started,video_sha256=None,alignment_config=alignment_config,blue_weights_and_normalization_unchanged=True)
    save();print(json.dumps({k:v for k,v in report.items() if k!='source_files'}),flush=True)
except BaseException as exc:
    report.update(state='failed',error=repr(exc),wall_seconds=time.monotonic()-started);save();raise
finally:
    if writer is not None:writer.close()
    if renderer is not None:renderer.close()
    if env is not None:env.close()
