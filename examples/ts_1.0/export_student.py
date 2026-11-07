"""ONNX contract, all-CUDA execution and real-observation numerical parity."""
from student_common import setup,sha,write_json
from pathlib import Path
import argparse,json,time

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--measure-upper-timing',action='store_true');a=p.parse_args();bundle,manifest,_=setup(a.bundle)
    import numpy as np,torch,onnx,onnxruntime as ort
    from student_policy import load
    from holoboxing.evaluation.runtime.model_contract import CONTRACT,check_model,check_contract
    from holoboxing.evaluation.runtime.upper import check_target_pose,neutral_observation
    from holoboxing.evaluation.runtime.onnx_device import create_device_session,verify_cuda_execution
    from holoboxing.training.export import check_export_parity
    torch.set_num_threads(1);net=load(a.checkpoint);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert saved['bundle_sha256']==sha(bundle/'manifest.json')
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'student.onnx';z=np.load(a.data/'samples.npz')
    valid=z['healthy']&np.isin(z['world'],saved['validation_worlds']);x=z['obs'][valid]
    assert len(x)>0
    probes=np.concatenate([x[::max(1,len(x)//48)][:48],neutral_observation(),np.zeros((1,1097),np.float32)]).astype(np.float32)
    torch.onnx.export(net,torch.from_numpy(probes[:1]),path,input_names=['obs'],output_names=['target_pose'],opset_version=17,dynamo=False,external_data=False)
    graph=onnx.load(path);onnx.helper.set_model_props(graph,dict(policy_contract=CONTRACT,teacher_sha256=sha(bundle/'state_aware_control.py'),
        external_state='none',lower_controller_included='false',motion_rate='4 punches per second teacher',
        straight_max_m='0.60',uppercut_max_m='0.45',policy_kind='learned public-observation student'))
    onnx.save(graph,path);graph,size=check_model(path);check_contract(graph)
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    cpu=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider']);gpu=create_device_session(path)
    maxima={};timings=[]
    with torch.inference_mode():
        for row in probes:
            feed={'obs':row[None]};expected=net(torch.from_numpy(row[None])).numpy()
            for name,session in [('cpu',cpu),('cuda',gpu)]:
                actual=session.run(['target_pose'],feed)[0];check_target_pose(actual)
                for key,value in check_export_parity(actual,expected).items():maxima[name+'_'+key]=max(maxima.get(name+'_'+key,0.),value)
        verify_cuda_execution(gpu)
        feed={'obs':probes[:1]};reference=gpu.run(None,feed)[0]
        for _ in range(10):assert np.array_equal(gpu.run(None,feed)[0],reference)
        if a.measure_upper_timing:
            for _ in range(100):
                t=time.perf_counter();gpu.run(None,feed);timings.append((time.perf_counter()-t)*1000)
    report=dict(state='completed',onnx_sha256=sha(path),bytes=size,input=dict(name='obs',shape=[1,1097],dtype='float32'),
                output=dict(name='target_pose',shape=[1,225],dtype='float32'),embedded_weights=True,external_state=False,checkpoint_sha256=sha(a.checkpoint),
                bundle_sha256=sha(bundle/'manifest.json'),parity=maxima,probe_count=len(probes),all_compute_nodes_cuda=True,
                deterministic_repeats=10,upper_only_cuda_ms_mean=float(np.mean(timings)) if timings else None,upper_only_cuda_ms_p95=float(np.percentile(timings,95)) if timings else None,
                complete_chain_5090_qualification='not performed',closed_loop_acceptance='separate evaluation required')
    write_json(a.output/'export-report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()

