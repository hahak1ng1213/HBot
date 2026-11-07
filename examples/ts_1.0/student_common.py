"""Immutable teacher/runtime loading and the public observation boundary."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','GPU-416cb80b-45b7-b6a7-2786-b0be58ec88d1')
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LP_NUM_THREADS']:
    os.environ[name]='1'
import sys,json,hashlib,importlib.util
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)

def setup(bundle):
    bundle=Path(bundle).resolve();manifest=json.loads((bundle/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if sha(bundle/name)!=expected:raise ValueError('Changed bundle: '+name)
    if sha(manifest['opponent'])!=manifest['opponent_sha256']:raise ValueError('Changed opponent')
    sys.path.insert(0,str(bundle/'runtime'))
    spec=importlib.util.spec_from_file_location('snapshot_teacher',bundle/'state_aware_control.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return bundle,manifest,module.StateAwareScheduler

def clean_observation(env):
    import torch
    context=torch.as_tensor(env.referees.context(),device=env.device).flatten(0,1)
    return torch.cat((env.measured['history'],env.measured['task'][:,-15:],context),-1)[::2].clone()

class Teacher:
    def __init__(self,executor,bank,point_ids,actual_ids,offset,floor):
        self.executor=executor;self.bank=bank;self.pi=point_ids;self.ai=actual_ids
        self.offset=offset;self.floor=floor

    def reset(self,raw):
        import numpy as np
        from scipy.spatial.transform import Rotation as R,Slerp
        self.planner=self.executor(self.bank,self.pi,self.ai,self.offset,self.floor)
        ref=self.bank['straight_lr'];p=raw['position'][0].cpu().numpy();q=raw['quaternion'][0].cpu().numpy()
        origin=ref['body_pos_w'][0,0].astype(float);position=p[0,0].copy();position[2]=origin[2]+self.floor
        delta=p[1,0,:2]-p[0,0,:2]
        yaw=R.from_quat(ref['body_quat_w'][0,0,[1,2,3,0]]).as_euler('xyz')[2]
        rotation=R.from_euler('z',np.arctan2(delta[1],delta[0])-yaw-self.offset)
        gp=rotation.apply(ref['body_pos_w'][0,self.pi]-origin)+position
        gq=(rotation*R.from_quat(ref['body_quat_w'][0,self.pi][:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
        sp,sq=p[0,self.ai],q[0,self.ai];self.entry_p=[];self.entry_q=[]
        for f in range(51):
            u=f/50;u=u**3*(10-15*u+6*u*u)
            self.entry_p.append(sp*(1-u)+gp*u)
            self.entry_q.append(np.stack([Slerp([0,1],R.from_quat(np.stack([sq[j],gq[j]])[:,[1,2,3,0]]))([u]).as_quat()[0,[3,0,1,2]] for j in range(5)]))
        self.entry_p=np.array(self.entry_p);self.entry_q=np.array(self.entry_q);self.frame=0

    def step(self,raw):
        import numpy as np
        if self.frame<50:
            ix=np.minimum(self.frame+np.arange(1,6),50)
            result=self.entry_p[ix],self.entry_q[ix],'ready-entry',{}
        else:result=self.planner.step(raw,self.frame)
        self.frame+=1
        return result
