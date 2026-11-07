"""Check the neural student's ONNX parity on actual rollout observations."""
from student_common import setup,sha,write_json
from pathlib import Path
import argparse,time
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();setup(a.bundle)
import numpy as np,torch,onnxruntime as ort
from student_policy import load
from holoboxing.evaluation.runtime.onnx_device import create_device_session,verify_cuda_execution
from holoboxing.evaluation.runtime.upper import check_target_pose
from holoboxing.training.export import check_export_parity
torch.set_num_threads(1);net=load(a.checkpoint);z=np.load(a.source/'samples.npz');x=z['obs']
options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
cpu=ort.InferenceSession(str(a.model),options,providers=['CPUExecutionProvider']);gpu=create_device_session(a.model)
r=dict(state='running',source=str(a.source),model_sha256=sha(a.model),source_sha256=sha(a.source/'samples.npz'),samples=len(x),checked=0,failures=[],maxima={},deterministic_active_repeats=0);start=time.time()
with torch.inference_mode():
    inputs=torch.from_numpy(x)
    # Select a near/aligned observation for repeated deterministic inference.
    opp=inputs[:,1080:1095].reshape(-1,5,3)[:,0,:2];radius=opp.norm(dim=-1)
    candidates=torch.nonzero((radius>.47)&(radius<.56)&(opp[:,1].abs()<.12)).flatten()
    active_index=int(candidates[len(candidates)//2]) if len(candidates) else 0
    r['near_aligned_probe_rows']=len(candidates);r['repeat_probe_index']=active_index
    for i,row in enumerate(x):
        expected=net(torch.from_numpy(row[None])).numpy();feed={'obs':row[None]}
        for label,session in [('cpu',cpu),('cuda',gpu)]:
            actual=session.run(['target_pose'],feed)[0];check_target_pose(actual)
            pos=np.linalg.norm((actual-expected).reshape(1,5,5,9)[...,:3],axis=-1).max()
            r['maxima'][label+'_position_m']=max(r['maxima'].get(label+'_position_m',0),float(pos))
            try:metrics=check_export_parity(actual,expected)
            except ValueError as exc:
                r['failures'].append(dict(row=i,provider=label,error=str(exc),position_error_m=float(pos)));continue
            for k,v in metrics.items():r['maxima'][label+'_'+k]=max(r['maxima'].get(label+'_'+k,0),v)
        r.update(checked=i+1,wall_seconds=time.time()-start)
        if i%100==0:write_json(a.output,r)
    verify_cuda_execution(gpu);feed={'obs':x[active_index:active_index+1]};reference=gpu.run(None,feed)[0]
    for _ in range(20):assert np.array_equal(gpu.run(None,feed)[0],reference)
r.update(state='completed',deterministic_active_repeats=20,wall_seconds=time.time()-start);write_json(a.output,r);print(r,flush=True)

