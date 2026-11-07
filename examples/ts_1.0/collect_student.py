"""Real teacher/DAgger data; student receives only the public 1097-vector."""
from student_common import setup,sha,write_json,clean_observation,Teacher
import argparse,json,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worlds',type=int,default=8);p.add_argument('--seconds',type=float,default=12);p.add_argument('--seed',type=int,default=71101)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--onnx',type=Path);p.add_argument('--teacher-fraction',type=float,default=1.)
    p.add_argument('--evaluate',action='store_true');p.add_argument('--native-reset',action='store_true')
    p.add_argument('--guard-start',action='store_true',help='training-only guard pose and close-range initial states')
    a=p.parse_args();assert 0<=a.teacher_fraction<=1
    bundle,manifest,Executor=setup(a.bundle)
    import numpy as np,torch
    from scipy.spatial.transform import Rotation as R
    from holoboxing.data.contract import CompetitionAssets
    from holoboxing.training.resources import resolve_resources
    from holoboxing.training.export import load_policy
    from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
    from holoboxing.training.tasks.combat.config.sandbag import BODY_NAMES,POINTS
    from holoboxing.training.tasks.combat.mdp.geometry import encode_output,project_xz,rot_xz
    from student_policy import StandardPolicy,load
    torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report=dict(state='initializing',args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                bundle_sha256=sha(bundle/'manifest.json'),teacher_sha256=sha(bundle/'state_aware_control.py'),
                public_input='clean history10x108 + opponent5x3 + context2',target='C225 in current full pelvis frame',
                student_external_state=False,teacher_phase_is_not_student_input=True,official_qualification=False,
                collector_sha256=sha(__file__))
    def status(**kw):report.update(kw,wall_seconds=time.monotonic()-start);write_json(a.output/'status.json',report)
    status();env=None
    try:
        bank={f.stem:dict(np.load(f)) for f in (bundle/'motions').glob('*.npz')}
        ref=bank['straight_lr'];hold={k:v.copy() for k,v in ref.items()}
        for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
            hold[k]=np.repeat(ref[k][:1],7,axis=0)
            if 'vel' in k:hold[k][:]=0
        bank['settle_guard']=hold
        pi=[ref['body_names'].tolist().index(n) for n in POINTS];ai=[BODY_NAMES.index(n) for n in POINTS]
        axis=manifest['axis'];offset=float(np.arctan2(axis[1],axis[0])-R.from_quat(ref['body_quat_w'][0,0,[1,2,3,0]]).as_euler('xyz')[2])
        blue=load_policy(manifest['opponent']).cuda().eval().requires_grad_(False)
        frozen={k:v.detach().clone() for k,v in blue.state_dict().items()}
        assert not (a.checkpoint and a.onnx)
        student=load(a.checkpoint,'cuda:0') if a.checkpoint else None
        session=None
        if a.onnx:
            from holoboxing.evaluation.runtime.onnx_device import create_device_session,verify_cuda_execution
            from holoboxing.evaluation.runtime.model_contract import check_model,check_contract
            graph,_=check_model(a.onnx);check_contract(graph);session=create_device_session(a.onnx)
            report['onnx_sha256']=sha(a.onnx)
        assert a.teacher_fraction==1 or student is not None or session is not None
        assets_path=Path('/home/zhanghan/datasets/hbot-assets');assets=CompetitionAssets.load(assets_path)
        _,paths,resources=resolve_resources(assets_path,Path('/home/zhanghan/datasets/holomotion-v2'))
        env=MujocoSandbagEnv(assets,paths,a.worlds,'cuda:0',a.output/'runtime',a.seed,official_rules=True,capture_transition_state=True,physics_threads=min(4,a.worlds))
        if a.evaluate:
            def capture(base):return dict(record_qpos=torch.from_numpy(np.stack([d.qpos.copy() for d in base.physics.native])),record_qvel=torch.from_numpy(np.stack([d.qvel.copy() for d in base.physics.native])))
            env.transition_measurement=capture
        obs=env.reset();env.next_push[:]=float('inf')
        scenarios=[(.42,0),(.56,0),(.9,-20),(1.4,25),(.58,-18),(.43,10),(.44,-12),(1.1,18)]
        assert not (a.native_reset and a.guard_start)
        if a.guard_start:scenarios=[(.34,0),(.38,0),(.41,-8),(.44,10),(.36,-10),(.40,8),(.35,-6),(.42,6)]
        if not a.native_reset:
            qa,qb=env.physics.root_qi;rng=np.random.default_rng(a.seed)
            for i in range(a.worlds):
                distance,angle=scenarios[i%8];distance+=rng.uniform(-.015,.015);angle+=rng.uniform(-3,3)
                direction=(env.physics.data.qpos[i,qb:qb+2]-env.physics.data.qpos[i,qa:qa+2]).cpu().numpy();direction/=np.linalg.norm(direction)
                direction=R.from_euler('z',angle,degrees=True).apply(np.r_[direction,0])[:2]
                env.physics.data.qpos[i,qb:qb+2]=env.physics.data.qpos[i,qa:qa+2]+torch.tensor(direction*distance,device=env.device)
                if a.guard_start:
                    # Explicit episode initialization, before any sampled transition.
                    # The original motions, dynamics, referee and blue weights are unchanged.
                    heading=float(np.arctan2(direction[1],direction[0]))
                    m=env.physics.models[i]
                    for role,root_address in enumerate([qa,qb]):
                        prefix='p1_' if role==0 else 'p2_'
                        joint_ids=[int(m.joint(prefix+str(n)).qposadr[0]) for n in ref['joint_names']]
                        env.physics.data.qpos[i,joint_ids]=torch.tensor(ref['joint_pos'][0],device=env.device,dtype=env.physics.data.qpos.dtype)
                        env.physics.data.qpos[i,root_address+2]=float(ref['body_pos_w'][0,0,2])+env.physics.floor_height
                        q=R.from_euler('z',heading+role*np.pi-offset).as_quat()[[3,0,1,2]]
                        env.physics.data.qpos[i,root_address+3:root_address+7]=torch.tensor(q,device=env.device)
            env.physics.data.qvel.zero_();env.physics.forward()
            env.previous_p=env.previous_q=env.history=None;env.actor_noise.reset_history()
            obs=env._read(torch.ones(a.worlds,dtype=torch.bool,device=env.device))
        def sliced(raw,i):return {k:v[i:i+1] for k,v in raw.items() if k in ['position','quaternion','velocity']}
        raw=env.physics.state();teachers=[Teacher(Executor,bank,pi,ai,offset,env.physics.floor_height) for _ in range(a.worlds)]
        for i,t in enumerate(teachers):
            t.reset(sliced(raw,i))
            if a.guard_start:t.frame=50
        x=clean_observation(env)
        hold_p=raw['position'][:,0,ai][:,None].expand(-1,5,-1,-1);hold_q=raw['quaternion'][:,0,ai][:,None].expand(-1,5,-1,-1)
        expected=encode_output(raw['position'][:,0,ai],raw['quaternion'][:,0,ai],hold_p,hold_q)
        baseline_error=float((StandardPolicy.baseline(x)-expected).abs().max());assert baseline_error<2e-5,baseline_error
        arrays={k:[] for k in ['obs','target','world','episode','healthy','skill']};records=[];states=[];velocities=[]
        healthy=np.ones(a.worlds,bool);episodes=np.zeros(a.worlds,int);hits=np.zeros((a.worlds,2),int);falls=np.full((a.worlds,2),np.nan)
        status(state='running',resources=resources,baseline_frame_error=baseline_error,teacher_settings=teachers[0].planner.settings)
        with torch.inference_mode():
            for frame in range(round(a.seconds*50)):
                raw=env.physics.state();result=[t.step(sliced(raw,i)) for i,t in enumerate(teachers)]
                tp=torch.tensor(np.stack([v[0] for v in result]),dtype=torch.float32,device=env.device)
                tq=torch.tensor(np.stack([v[1] for v in result]),dtype=torch.float32,device=env.device)
                target=encode_output(raw['position'][:,0,ai],raw['quaternion'][:,0,ai],tp,tq);x=clean_observation(env)
                assert x.shape==(a.worlds,1097) and target.shape==(a.worlds,225)
                assert torch.isfinite(x).all() and torch.isfinite(target).all()
                arrays['obs'].append(x.cpu().numpy());arrays['target'].append(target.cpu().numpy());arrays['world'].append(np.arange(a.worlds))
                arrays['episode'].append(episodes.copy());arrays['healthy'].append(healthy.copy());arrays['skill'].append([v[2] for v in result])
                action=target
                if student is not None or session is not None:
                    if session is not None:
                        prediction=torch.tensor(np.concatenate([session.run(['target_pose'],{'obs':row[None]})[0] for row in x.cpu().numpy()]),device=env.device)
                        if frame==0:verify_cuda_execution(session)
                    else:prediction=student(x)
                    v=(a.teacher_fraction*target+(1-a.teacher_fraction)*prediction).reshape(-1,5,5,9)
                    action=torch.cat((v[...,:3],rot_xz(project_xz(v[...,3:]))),-1).flatten(1)
                obs,_,done,info=env.step({'robot_A':action,'robot_B':blue(obs['robot_B']['task_policy'])})
                if a.evaluate:
                    states.append(env.last_transition_state['record_qpos'].cpu().numpy())
                    velocities.append(env.last_transition_state['record_qvel'].cpu().numpy())
                row=dict(time_s=(frame+1)*.02,skills=[v[2] for v in result],invalid=info['invalid_envs'].cpu().tolist(),done=done.cpu().tolist())
                for role,name in enumerate(['robot_A','robot_B']):
                    hits[:,role]+=env.metrics[name]['official_hits'].cpu().numpy().astype(int)
                    fallen=env.metrics[name]['fallen'].cpu().numpy().astype(bool);mask=fallen&np.isnan(falls[:,role]);falls[mask,role]=(frame+1)*.02
                    row[name+'_fallen']=fallen.tolist();row[name+'_score']=env.metrics[name]['official_score'].cpu().tolist()
                    if role==0:healthy&=~fallen
                bad=info['invalid_envs'].cpu().numpy();healthy&=~bad;arrays['healthy'][-1]&=~bad;records.append(row)
                if (frame+1)%100==0:status(simulation_seconds=(frame+1)*.02,hits=hits.tolist(),rows=len(records)*a.worlds)
                completed=done.cpu().numpy()
                if a.evaluate and completed.any():break
                if completed.any():
                    raw=env.physics.state()
                    for i in np.flatnonzero(completed):episodes[i]+=1;teachers[i].reset(sliced(raw,i));healthy[i]=True
                    env.next_push[:]=float('inf')
        data={k:np.concatenate(v) if k not in ['skill','healthy'] else np.asarray(v).reshape(-1) for k,v in arrays.items()}
        np.savez_compressed(a.output/'samples.npz',**data)
        if states:np.savez_compressed(a.output/'states.npz',qpos=np.stack(states),qvel=np.stack(velocities))
        write_json(a.output/'metrics.json',records)
        assert all(torch.equal(v,blue.state_dict()[k]) for k,v in frozen.items())
        assert sha(manifest['opponent'])==manifest['opponent_sha256']
        status(state='completed',simulation_seconds=len(records)*.02,rows=len(data['obs']),usable_rows=int(data['healthy'].sum()),hits=hits.tolist(),
               first_fall_seconds=[[None if np.isnan(v) else float(v) for v in row] for row in falls],
               invalid_frames=sum(sum(r['invalid']) for r in records),blue_unchanged=True,samples_sha256=sha(a.output/'samples.npz'))
        print(json.dumps(report),flush=True)
    except BaseException as exc:status(state='failed',error=repr(exc));raise
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
