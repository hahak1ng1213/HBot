import json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from stance_alignment_control import geometry,choose,wrap
root=Path(__file__).resolve().parent;c=json.loads((root/'alignment.json').read_text());rules=json.loads((root/'candidate_50hz/rules.json').read_text())
z=np.load(root/'candidate_50hz/motions/straight_lr.npz');b=z['body_names'].tolist();ids=[b.index(x) for x in ['pelvis','left_wrist_yaw_link','right_wrist_yaw_link','left_ankle_roll_link','right_ankle_roll_link']]
p0=z['body_pos_w'][0,ids];q0=z['body_quat_w'][0,ids];qroot=R.from_quat(q0[0,[1,2,3,0]]);h=qroot.as_euler('xyz')[2]+c['pelvis_heading_offset_rad'];count=0
for distance in [.4,.55,.8,1.5]:
 for angle in [-170,-31,-16,-13,-9,-5,0,5,9,13,16,31,170]:
    a=h+math.radians(angle);delta=np.array([math.cos(a)*distance,math.sin(a)*distance,0.]);p=np.stack([p0,p0+delta]);q=np.stack([q0,q0])
    g=geometry(p,q,[3,4],c);assert abs(wrap(math.radians(g['pelvis_error_deg']-angle)))<1e-5
    choice,skill,error=choose(g,h,c,rules)
    if not g['aligned']:assert choice==1
    elif distance>.6:assert choice==0
    elif distance>.5:assert choice==2
    else:assert choice==3
    rot=R.from_euler('z',1.23);pp=rot.apply(p.reshape(-1,3)).reshape(p.shape)+[2.,-1.,.3]
    qq=(rot*R.from_quat(q.reshape(-1,4)[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]].reshape(q.shape)
    gg=geometry(pp,qq,[3,4],c)
    np.testing.assert_allclose([g['pelvis_error_deg'],g['feet_error_deg']],[gg['pelvis_error_deg'],gg['feet_error_deg']],atol=1e-4)
    assert choose(gg,h+1.23,c,rules)[:2]==(choice,skill);count+=1
g=dict(aligned=False,pelvis_error_deg=-15.,feet_error_deg=-6.,zone='attack',pelvis_heading_rad=0.)
assert choose(g,math.radians(-15),c,rules)[1]=='turn_right_10'
assert choose(g,math.radians(-25),c,rules)[1]=='settle_guard'
result=dict(status='PASS',geometry_and_decision_cases=count,global_yaw_translation_invariance=True,persistent_tracking_bias_compensation=True,bounded_reference_lead=True)
(root/'control-preflight.json').write_text(json.dumps(result,indent=2));print(result)
