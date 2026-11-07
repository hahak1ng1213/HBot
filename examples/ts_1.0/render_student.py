"""Render actual pure-student rollout, without stepping or teacher labels."""
from student_common import setup,sha,write_json
from pathlib import Path
import argparse,os,sys,json
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();setup(a.bundle);sys.path.insert(0,'/home/zhanghan/tools/hbot-video')
import numpy as np,mujoco,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from holoboxing.data.contract import CompetitionAssets
from holoboxing.data.robot import load_scene_model
r=json.loads((a.source/'status.json').read_text());assert r['state']=='completed' and r['args']['teacher_fraction']==0 and r['args']['onnx']
z=np.load(a.source/'states.npz');metrics=json.loads((a.source/'metrics.json').read_text());assert z['qpos'].shape[1]==1
a.output.mkdir(exist_ok=False,parents=True)
m=load_scene_model(CompetitionAssets.load(Path('/home/zhanghan/datasets/hbot-assets')))
for gid in range(m.ngeom):
    if m.geom(gid).name.startswith(('smpl_ring_wall','smpl_ring_corner')):m.geom_rgba[gid,3]=.12
m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
d=mujoco.MjData(m);renderer=mujoco.Renderer(m,width=1280,height=720);cam=mujoco.MjvCamera()
roots=[m.body('p1_pelvis').id,m.body('p2_pelvis').id]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
w=imageio_ffmpeg.write_frames(str(a.output/'motion.mp4'),(1280,720),fps=25,codec='libx264',quality=8,output_params=['-threads','1','-movflags','+faststart']);w.send(None)
indices=list(range(1,len(metrics),2));picks=set(indices[i] for i in np.linspace(0,len(indices)-1,6).astype(int));stills=[]
try:
    for f in indices:
        d.qpos[:]=z['qpos'][f,0];d.qvel[:]=z['qvel'][f,0];mujoco.mj_forward(m,d)
        distance=np.linalg.norm(d.xpos[roots[0],:2]-d.xpos[roots[1],:2])
        cam.distance=max(3.3,1.8+1.3*distance);cam.azimuth=135;cam.elevation=-28;cam.lookat[:]=d.xpos[roots].mean(0);cam.lookat[2]=.7
        renderer.update_scene(d,camera=cam);pic=Image.fromarray(renderer.render());dr=ImageDraw.Draw(pic);row=metrics[f]
        dr.rectangle((0,0,1280,85),fill='#101927')
        lines=[f'STUDENT ONNX | RED vs frozen BLUE seed44 | {row["time_s"]:.2f}s',
            f'Official score {row["robot_A_score"][0]:g} : {row["robot_B_score"][0]:g} | distance {distance:.2f}m',
            'Actual native physics | normal speed | NO teacher assistance']
        for k,line in enumerate(lines):dr.text((12,5+25*k),line,font=font,fill='white')
        w.send(np.asarray(pic))
        if f in picks:stills.append(pic.resize((640,360)))
finally:w.close();renderer.close()
sheet=Image.new('RGB',(1280,360*((len(stills)+1)//2)))
for i,pic in enumerate(stills):sheet.paste(pic,((i%2)*640,(i//2)*360))
sheet.save(a.output/'overview.jpg',quality=92)
frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(a.output/'motion.mp4'));assert frames==len(indices)
write_json(a.output/'video-report.json',dict(frames=frames,seconds=seconds,source=str(a.source.resolve()),source_sha256=sha(a.source/'states.npz'),onnx_sha256=r['onnx_sha256'],video_sha256=sha(a.output/'motion.mp4'),resimulated=False,teacher_assistance=False,render_only_wall_opacity=.12))
print('VIDEO_DONE',frames,seconds,flush=True)
