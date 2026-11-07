"""Fit a public-observation student; select whole-scenario validation loss."""
from student_common import setup,sha,write_json
from pathlib import Path
import argparse,time,json

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--data',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=160);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--warmstart',type=Path,help='Explicit weights-only initialization from a compatible previous teacher bundle')
    p.add_argument('--validation-worlds',type=int,nargs='+',default=[6,7]);p.add_argument('--seed',type=int,default=71110)
    a=p.parse_args();bundle,manifest,_=setup(a.bundle)
    import numpy as np,torch
    from student_policy import StandardPolicy
    torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report=dict(state='initializing',bundle_sha256=sha(bundle/'manifest.json'),epochs_requested=a.epochs,
                input=[1,1097],output=[1,225],external_state=False,validation_worlds=a.validation_worlds,
                initialization='new direct pose network; existing blue checkpoint has an old-motion frozen latent decoder',
                checkpoint=str(a.checkpoint) if a.checkpoint else None,
                source_hashes={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('student_policy.py')]})
    def status(**kw):report.update(kw,wall_seconds=time.monotonic()-start);write_json(a.output/'status.json',report)
    status()
    try:
        keys=['obs','target','world','healthy','skill'];parts={k:[] for k in keys};provenance=[]
        for folder in a.data:
            source=json.loads((folder/'status.json').read_text());assert source['state']=='completed'
            assert source['bundle_sha256']==report['bundle_sha256'] and source['teacher_sha256']==sha(bundle/'state_aware_control.py')
            file=folder/'samples.npz';assert sha(file)==source['samples_sha256'];z=np.load(file)
            for k in keys:parts[k].append(z[k])
            provenance.append(dict(path=str(file.resolve()),sha256=sha(file),rows=len(z['obs'])))
        data={k:np.concatenate(v) for k,v in parts.items()}
        good=data['healthy']&np.isfinite(data['obs']).all(1)&np.isfinite(data['target']).all(1)
        valid=good&np.isin(data['world'],a.validation_worlds);train=good&~np.isin(data['world'],a.validation_worlds)
        assert train.sum()>=500 and valid.sum()>=100,(int(train.sum()),int(valid.sum()))
        x=torch.tensor(data['obs'][train],device='cuda');y=torch.tensor(data['target'][train],device='cuda')
        vx=torch.tensor(data['obs'][valid],device='cuda');vy=torch.tensor(data['target'][valid],device='cuda')
        net=StandardPolicy().cuda()
        if a.checkpoint or a.warmstart:
            saved=torch.load(a.checkpoint or a.warmstart,map_location='cuda',weights_only=False)
            assert saved['format']=='hbot_four_punch_student_v1'
            if a.checkpoint:assert saved['bundle_sha256']==report['bundle_sha256']
            report['weights_lineage']=dict(path=str(a.checkpoint or a.warmstart),sha256=sha(a.checkpoint or a.warmstart),previous_bundle_sha256=saved['bundle_sha256'],optimizer_restored=False)
            net.load_state_dict(saved['model'],strict=True)
        else:
            with torch.no_grad():
                net.input_mean.copy_(x[:,:1095].mean(0));net.input_scale.copy_(x[:,:1095].std(0).clamp_min(.05))
                residual=y-net.baseline(x);net.output_mean.copy_(residual.mean(0));net.output_scale.copy_(residual.std(0).clamp_min(.02))
        def family(name):
            return 'forward' if name.startswith('forward') else 'turn' if name.startswith('turn') else name
        labels=np.array([family(str(v)) for v in data['skill'][train]]);vlabels=np.array([family(str(v)) for v in data['skill'][valid]])
        names,counts=np.unique(labels,return_counts=True);count=dict(zip(names,counts))
        for family_name in ['forward','turn','straight_lr','uppercut_lr']:
            if count.get(family_name,0)<50:
                raise ValueError('Insufficient teacher coverage for '+family_name+': '+str(count.get(family_name,0)))
            if np.count_nonzero(vlabels==family_name)<10:
                raise ValueError('Insufficient held-out coverage for '+family_name)
        weights=torch.tensor([len(x)/len(names)/count[v] for v in labels],device='cuda',dtype=torch.float32).clamp(.25,4)
        scale=torch.tensor([10,10,10,1,1,1,1,1,1],device='cuda').repeat(25)
        opt=torch.optim.AdamW(net.parameters(),lr=2e-4 if a.checkpoint else 4e-4,weight_decay=1e-5)
        best=float('inf');best_epoch=0;history=[];updates=0
        def save(name,epoch):
            torch.save(dict(format='hbot_four_punch_student_v1',model={k:v.detach().cpu() for k,v in net.state_dict().items()},
                            optimizer=opt.state_dict(),epoch=epoch,updates=updates,bundle_sha256=report['bundle_sha256'],
                            provenance=provenance,validation_worlds=a.validation_worlds),a.output/name)
        status(state='training',train_rows=len(x),validation_rows=len(vx),by_family_train={str(k):int(v) for k,v in zip(names,counts)},
               by_family_validation={str(k):int(np.count_nonzero(vlabels==k)) for k in np.unique(vlabels)},provenance=provenance)
        for epoch in range(1,a.epochs+1):
            net.train();total=0.
            for ix in torch.randperm(len(x),device='cuda').split(512):
                pred=net.unprojected(x[ix]);loss=((((pred-y[ix])*scale).square().mean(-1))*weights[ix]).mean()
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                total+=float(loss)*len(ix);updates+=1
            net.eval()
            with torch.no_grad():
                pred=torch.cat([net(v) for v in vx.split(512)])
                err=((pred-vy)*scale).square().mean(-1)
                # Select equal-family performance, not a guard-dominated global average.
                val=float(torch.stack([err[torch.tensor(vlabels==k,device='cuda')].mean() for k in np.unique(vlabels)]).mean())
                pos=(pred-vy).reshape(-1,5,5,9)[...,:3].norm(dim=-1)
                row=dict(epoch=epoch,updates=updates,training_loss=total/len(x),validation_loss=val,
                         position_rmse_m=float(pos.square().mean().sqrt()),position_p95_m=float(torch.quantile(pos.flatten(),.95)),
                         wrist_rmse_m=float(pos[:,:,1:3].square().mean().sqrt()))
            history.append(row)
            if val<best:best=val;best_epoch=epoch;save('best.pt',epoch)
            if epoch==1 or epoch%10==0:
                save('latest.pt',epoch);status(**row,best_epoch=best_epoch,best_validation_loss=best);write_json(a.output/'history.json',history);print(json.dumps(row),flush=True)
            if epoch in [80,120]:
                for group in opt.param_groups:group['lr']*=.4
            if epoch>=80 and epoch-best_epoch>=30:break
        save('latest.pt',epoch);write_json(a.output/'history.json',history)
        status(state='completed',**history[-1],best_epoch=best_epoch,best_validation_loss=best,best_checkpoint_sha256=sha(a.output/'best.pt'))
    except BaseException as exc:status(state='failed',error=repr(exc));raise

if __name__=='__main__':main()
